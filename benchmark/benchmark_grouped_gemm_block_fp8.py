# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Benchmark the XE2 cutlass grouped GEMM kernel across quantization recipes.

Compares three weight recipes for a single grouped GEMM (one expert-batched
matmul, the same op used by gemm1/gemm2 inside the fused MoE):

* ``bf16``       - 16-bit weights, no scales (``cutlass_grouped_gemm_xe2``).
* ``fp8``        - per-tensor / per-expert fp8 weights (W8A16, 1D scales).
* ``block_fp8``  - block-wise fp8 weights (W8A16, 128x128 fp32 block scales).

Latency is reported in microseconds along with achieved TFLOP/s so the recipes
can be compared at matched problem sizes.

Example:
    python benchmark/benchmark_grouped_gemm_block_fp8.py
    python benchmark/benchmark_grouped_gemm_block_fp8.py --check
    python benchmark/benchmark_grouped_gemm_block_fp8.py \
        --save-path ./grouped_gemm_bench
"""

# isort: off
import argparse
import csv
import gc

import torch

try:
    import triton
    import triton.testing
    HAS_TRITON = True
except ImportError:
    triton = None
    HAS_TRITON = False

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)
from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.utils import seed_everything
from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2
# isort: on

DEVICE = "xpu"
BLOCK_SIZE = 128

# (m, n, k) - shapes for Llama-4-scout style MoE grouped GEMMs.
MNK_FACTORS = [
    (1, 5120, 8192),
    (4, 5120, 8192),
    (16, 5120, 8192),
    (64, 5120, 8192),
    (256, 5120, 8192),
    (1024, 5120, 8192),
    (8192, 5120, 8192),
]
NUM_EXPERTS = 16
TOPK = 1

# DeepSeek-V3: hidden 7168, moe_intermediate 2048, 256 routed experts, top-8,
# 128x128 block scales. gate_up fuses gate+up, so its N is 2 * 2048.
# Expert count is per rank = 256 / EP; 64 matches the TP=4 + EP runs.
DSV3_M = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
PRESETS = {
    "llama4": (MNK_FACTORS, 16, 1),
    "dsv3-gateup": ([(m, 4096, 7168) for m in DSV3_M], 64, 8),
    "dsv3-down": ([(m, 7168, 2048) for m in DSV3_M], 64, 8),
}

RECIPES = ["bf16", "fp8", "block_fp8"]
RECIPE_STYLES = {
    "bf16": ("blue", "-"),
    "fp8": ("green", "-"),
    "block_fp8": ("red", "-"),
}


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def init_rows_for_experts(tokens, topk, num_rows_per_expert):
    """Uniform-random routing of tokens*topk rows across experts.

    Mirrors tests/fused_moe/test_grouped_gemm.py; inlined so the benchmark does
    not import the test module (which requires pytest).
    """
    if num_rows_per_expert.shape[0] == 1:
        num_rows_per_expert[0] = tokens * topk
        return
    n_experts = num_rows_per_expert.numel()
    rand = torch.rand(tokens, n_experts, device=num_rows_per_expert.device)
    topk_idx = torch.topk(rand, topk, dim=1).indices
    num_rows_per_expert += torch.bincount(topk_idx.flatten(),
                                          minlength=n_experts)


def calculate_tflops(m, n, k, latency_us):
    # The kernel processes m * TOPK rows: each token is routed to TOPK experts.
    flops = 2.0 * m * TOPK * n * k
    return flops / (latency_us * 1e-6) / 1e12


def make_grouped_gemm_input(m, n, k, recipe, x_dtype, fp8_dtype, has_bias):
    """Build inputs for ``cutlass_grouped_gemm_xe2`` for the given recipe.

    Returns a kwargs dict ready to splat into ``cutlass_grouped_gemm_xe2``.
    """
    num_experts = NUM_EXPERTS
    total_m = m * TOPK

    input_A = torch.randn((total_m, k), dtype=x_dtype,
                          device=DEVICE).contiguous()
    # Weights are stored row-major [num_experts, K, N] for the W8A16 paths.
    input_B = torch.randn((num_experts, k, n), dtype=x_dtype, device=DEVICE)

    if has_bias:
        bias = torch.randn((num_experts, n), dtype=x_dtype, device=DEVICE)
    else:
        bias = None

    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, TOPK, num_rows_per_expert)
    output = torch.empty((total_m, n), dtype=x_dtype, device=DEVICE)

    if recipe == "bf16":
        scales = None
        is_B_fp8block = False
    elif recipe == "fp8":
        # per-expert (per-tensor) fp8 quantization, 1D scales [num_experts].
        random_exponents = torch.randint(-3, 4, (num_experts, ),
                                         device=DEVICE)
        scales = torch.pow(2.0, random_exponents.float())
        input_B_fp8 = torch.empty_like(input_B, dtype=fp8_dtype)
        for i in range(num_experts):
            input_B_fp8[i], _ = scaled_fp8_quant(input_B[i],
                                                 scales[i].to(torch.float32),
                                                 False,
                                                 False,
                                                 fp8_dtype=fp8_dtype)
        input_B = input_B_fp8
        is_B_fp8block = False
    elif recipe == "block_fp8":
        assert n % BLOCK_SIZE == 0 and k % BLOCK_SIZE == 0, (
            "block_fp8 requires N and K divisible by 128")
        # block-wise fp8 weights, 128x128 fp32 block scales
        # [num_experts, K // 128, N // 128].
        num_k_blocks = k // BLOCK_SIZE
        num_n_blocks = n // BLOCK_SIZE
        random_exponents = torch.randint(
            -3, 4, (num_experts, num_k_blocks, num_n_blocks), device=DEVICE)
        scales = torch.pow(2.0, random_exponents.float()).contiguous()
        input_B = input_B.to(fp8_dtype)
        is_B_fp8block = True
    else:
        raise ValueError(f"Unknown recipe: {recipe}")

    return dict(
        input_A=input_A,
        input_B=input_B,
        scales=scales,
        bias=bias,
        output=output,
        num_rows_per_expert=num_rows_per_expert,
        n=n,
        k=k,
        num_experts=num_experts,
        is_B_int4=False,
        is_B_mxfp4=False,
        is_B_fp8block=is_B_fp8block,
    )


def run_grouped_gemm(kwargs):
    # The recipe is inferred from the dtype/shape of input_B and scales.
    cutlass_grouped_gemm_xe2(
        kwargs["input_A"],
        kwargs["input_B"],
        kwargs["scales"],
        kwargs["bias"],
        kwargs["output"],
        kwargs["num_rows_per_expert"],
        kwargs["n"],
        kwargs["k"],
        kwargs["num_experts"],
    )


def time_ms(fn, warmup=25, rep=100):
    """Median latency in milliseconds using XPU events (triton-free)."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    times = []
    for _ in range(rep):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def get_benchmark(x_dtype, fp8_dtype, has_bias):
    line_vals = []
    line_names = []
    styles = []
    for recipe in RECIPES:
        line_vals.append(recipe)
        line_names.append(f"{recipe}(us)")
        styles.append(RECIPE_STYLES[recipe])
    for recipe in RECIPES:
        line_vals.append(f"{recipe}_tflops")
        line_names.append(f"{recipe}(TFLOP/s)")
        color, _ = RECIPE_STYLES[recipe]
        styles.append((color, "--"))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k"],
            x_vals=MNK_FACTORS,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            styles=styles,
            ylabel="Latency (us)",
            plot_name=f"grouped-gemm-block-fp8-{x_dtype}".replace(
                "torch.", ""),
            args={},
        ))
    def benchmark(m, n, k, provider):
        clear_xpu_cache()
        recipe = provider.replace("_tflops", "")
        report_tflops = provider.endswith("_tflops")

        kwargs = make_grouped_gemm_input(m, n, k, recipe, x_dtype, fp8_dtype,
                                         has_bias)

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_grouped_gemm(kwargs),
            quantiles=quantiles,
            warmup=200,
            rep=1000,
        )

        if report_tflops:
            return (
                calculate_tflops(m, n, k, ms * 1000),
                calculate_tflops(m, n, k, max_ms * 1000),
                calculate_tflops(m, n, k, min_ms * 1000),
            )
        # do_bench returns milliseconds; report microseconds.
        return ms * 1000, min_ms * 1000, max_ms * 1000

    return benchmark


def bench_us(kwargs):
    """Median latency in microseconds, preferring triton's do_bench."""
    def fn():
        run_grouped_gemm(kwargs)

    if HAS_TRITON:
        return triton.testing.do_bench(fn, warmup=200, rep=1000) * 1000
    return time_ms(fn) * 1000


def run_plain(x_dtype, fp8_dtype, has_bias, csv_path=None):
    """Print a latency/TFLOP-s table per problem size, optionally saving CSV."""
    header = f"{'m':>6} {'n':>6} {'k':>6}"
    for recipe in RECIPES:
        header += f" | {recipe + ' us':>14} {recipe + ' TFLOP/s':>16}"
    print(header)
    print("-" * len(header))

    rows = []
    for m, n, k in MNK_FACTORS:
        line = f"{m:>6} {n:>6} {k:>6}"
        for recipe in RECIPES:
            clear_xpu_cache()
            kwargs = make_grouped_gemm_input(m, n, k, recipe, x_dtype,
                                             fp8_dtype, has_bias)
            us = bench_us(kwargs)
            tflops = calculate_tflops(m, n, k, us)
            line += f" | {us:>14.2f} {tflops:>16.2f}"
            rows.append((m, n, k, NUM_EXPERTS, TOPK, recipe, us, tflops))
        print(line, flush=True)

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["m", "n", "k", "experts", "topk", "recipe",
                        "latency_us", "tflops"])
            for r in rows:
                w.writerow([*r[:6], f"{r[6]:.2f}", f"{r[7]:.3f}"])
        print(f"\nwrote {csv_path}")


def check_correctness(x_dtype, fp8_dtype, has_bias):
    """Sanity check each recipe against an fp32 reference grouped GEMM."""
    seed_everything(7)
    n, k = 5120, 8192
    # Dispatch keys on A_avg_M = m*TOPK/NUM_EXPERTS, so sweep m to cover all
    # three policies; the large-M one folds scales into A instead of using a
    # second accumulator and is otherwise never exercised here.
    for avg_target in (4, 16, 64):
        m = max(1, round(avg_target * NUM_EXPERTS / TOPK))
        for recipe in RECIPES:
            check_one(m, n, k, recipe, x_dtype, fp8_dtype, has_bias)


def check_one(m, n, k, recipe, x_dtype, fp8_dtype, has_bias):
        avg_m = m * TOPK / NUM_EXPERTS
        kwargs = make_grouped_gemm_input(m, n, k, recipe, x_dtype, fp8_dtype,
                                         has_bias)
        run_grouped_gemm(kwargs)
        output = kwargs["output"]

        # Build an fp32 dequantized reference.
        input_A = kwargs["input_A"]
        input_B = kwargs["input_B"]
        scales = kwargs["scales"]
        bias = kwargs["bias"]
        num_rows_per_expert = kwargs["num_rows_per_expert"]

        ref = []
        pre = 0
        for i in range(NUM_EXPERTS):
            cur = int(num_rows_per_expert[i])
            if cur == 0:
                continue
            a = input_A[pre:pre + cur, :].to(torch.float32)
            if recipe == "bf16":
                w = input_B[i].to(torch.float32)
            elif recipe == "fp8":
                w = input_B[i].to(torch.float32) * scales[i]
            else:  # block_fp8
                scale_full = scales[i].repeat_interleave(
                    BLOCK_SIZE, dim=0).repeat_interleave(BLOCK_SIZE, dim=1)
                w = input_B[i].to(torch.float32) * scale_full
            out = a @ w
            if has_bias:
                out += bias[i]
            ref.append(out.to(x_dtype))
            pre += cur
        ref = torch.cat(ref, dim=0)

        try:
            torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-1)
            print(f"✅ {recipe:10s} m={m:<5d} avg_M={avg_m:<6.1f} matches reference")
        except AssertionError as err:
            print(f"❌ {recipe:10s} m={m:<5d} avg_M={avg_m:<6.1f} DIFFERS -> {err}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark XE2 grouped GEMM across fp8 recipes.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify each recipe against an fp32 reference before timing.")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to save benchmark CSV/plots. Defaults to no save.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Activation/output dtype.")
    parser.add_argument(
        "--fp8-dtype",
        type=str,
        choices=["e4m3", "e5m2"],
        default="e4m3",
        help="FP8 weight dtype for the fp8 / block_fp8 recipes.")
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Include a per-expert bias.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="llama4",
        help="Problem-size preset: shapes, expert count and topk.")
    parser.add_argument(
        "--experts",
        type=int,
        default=None,
        help="Override expert count (per rank, i.e. 256/EP for DeepSeek-V3).")
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Override experts-per-token.")
    parser.add_argument(
        "--m-values",
        type=str,
        default=None,
        help="Comma-separated token counts to sweep, overriding the preset.")
    parser.add_argument(
        "--recipes",
        type=str,
        default=None,
        help="Comma-separated subset of bf16,fp8,block_fp8.")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print a table instead of triton's perf_report (no matplotlib).")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Write results to this CSV path (implies --plain).")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(7)

    global MNK_FACTORS, NUM_EXPERTS, TOPK, RECIPES
    MNK_FACTORS, NUM_EXPERTS, TOPK = PRESETS[args.preset]
    if args.experts is not None:
        NUM_EXPERTS = args.experts
    if args.topk is not None:
        TOPK = args.topk
    if args.m_values:
        _, n, k = MNK_FACTORS[0]
        MNK_FACTORS = [(int(m), n, k) for m in args.m_values.split(",")]
    if args.recipes:
        RECIPES = [r.strip() for r in args.recipes.split(",")]
    print(f"[config] preset={args.preset} experts={NUM_EXPERTS} topk={TOPK} "
          f"recipes={RECIPES} shapes={len(MNK_FACTORS)}")

    x_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    fp8_dtype = {
        "e4m3": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2,
    }[args.fp8_dtype]
    has_bias = args.bias

    if args.check:
        check_correctness(x_dtype, fp8_dtype, has_bias)

    if not HAS_TRITON:
        print("[info] triton not available; using XPU-event timing fallback.")
    if args.plain or args.csv or not HAS_TRITON:
        run_plain(x_dtype, fp8_dtype, has_bias, args.csv)
        return

    benchmark = get_benchmark(x_dtype, fp8_dtype, has_bias)
    if args.save_path:
        ensure_save_path_exists(args.save_path)
        benchmark.run(print_data=True,
                      show_plots=False,
                      save_path=args.save_path)
    else:
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
