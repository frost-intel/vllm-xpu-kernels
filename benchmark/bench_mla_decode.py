# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the XPU paged-decode kernel on DeepSeek-V3 MLA shapes.

Reproduces what ``XPUMLAImpl.forward_mqa`` issues per layer per decode step:
one query token per sequence, 128 query heads, 1 KV head, head_size_qk 576
(kv_lora_rank 512 + qk_rope_head_dim 64) and head_size_vo 512, against a paged
KV cache.

Purpose
-------
Production traces show this call at ~306 us against a ~5 us bandwidth
roofline.  Two hypotheses:

  H1 (parallelism starvation).  The MLA backend pins block_size=16, which
     selects ``decode_policy_qpacked_head<q, head, _16>`` with
     ``SubgroupLayoutQK = Layout<Shape<_1,_1,_1>>`` -- one subgroup per
     work-group.  At batch=1 that leaves very few hardware threads resident.
     Prediction: latency is nearly flat as batch grows, then rises once the
     machine finally fills.

  H2 (KV-split never engaged).  ``num_splits_kv`` only takes effect when
     ``host_kv_lens`` is also supplied; otherwise the split plan is never
     built and the argument is silently ignored (see
     flash_attn_interface.py).  Prediction: forcing splits collapses
     latency at batch=1.

The sweeps below are designed to separate the two.  block_size is swept even
though production pins 16, because ``mha_varlen_fwd`` TORCH_CHECKs an SLM
budget when head_size_qk > 512 -- some combinations are expected to be
rejected outright, and those are reported as FAIL rather than hidden.

Usage:
    python benchmark/bench_mla_decode.py
    python benchmark/bench_mla_decode.py --batches 1 --splits 0 8 --block-sizes 16
"""

import argparse
import ctypes
import itertools
import os
import sys
import tempfile

import torch

from vllm_xpu_kernels.flash_attn_interface import (
    _infer_num_xe_cores,
    _kv_tile_from_block_size,
    build_decode_split_plan,
    flash_attn_varlen_func,
)

DEVICE = "xpu"

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM

# Subgroups per work-group implied by SubgroupLayoutQK in
# csrc/xpu/attn/xe_2/fmha_utils.hpp (decode_policy_qpacked_head).
_SGS_PER_WG = {16: 1, 32: 2, 64: 4}


def make_seq_lens(batch, seq_len, ragged, fixed_max=False):
    """Per-sequence KV lengths with spread ``ragged``.

    Two modes, which together discriminate *why* ragged batches are slow:

    ``fixed_max=False`` (default): ``seq_len`` is the mean and the batch total
    is held fixed, so a ragged batch moves the same KV bytes as the uniform one
    and only the distribution differs.

    ``fixed_max=True`` (inverted test): ``seq_len`` is the *max* and stays
    pinned while the mean falls to ``seq_len * (1 - ragged)``.  Total work
    therefore drops with ``ragged``.  If latency stays flat anyway, the longest
    sequence owns the critical path (tail effect); if latency falls with the
    mean, the kernel is work-bound and the penalty is elsewhere.
    """
    if ragged <= 0 or batch == 1:
        return [seq_len] * batch
    if fixed_max:
        # Same distribution shape as below, scaled by 1/(1+ragged) so the max
        # lands exactly on seq_len instead of the mean.
        lo = max(1, int(round(seq_len * (1.0 - ragged) / (1.0 + ragged))))
        return [int(round(lo + (seq_len - lo) * i / (batch - 1)))
                for i in range(batch)]
    lo = max(1, int(round(seq_len * (1.0 - ragged))))
    hi = int(round(seq_len * (1.0 + ragged)))
    lens = [int(round(lo + (hi - lo) * i / (batch - 1))) for i in range(batch)]
    lens[-1] = max(1, lens[-1] + (seq_len * batch - sum(lens)))
    return lens


def build_inputs(batch, seq_lens, num_heads_q, block_size, dtype, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn(
        batch, num_heads_q, HEAD_QK, dtype=dtype, device=DEVICE, generator=g
    )
    blocks_per_seq = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = blocks_per_seq * batch + 8
    cache = torch.randn(
        num_blocks,
        block_size,
        1,
        HEAD_QK,
        dtype=dtype,
        device=DEVICE,
        generator=g,
    )
    k_cache = cache
    # Same non-contiguous narrow view the MLA backend passes for V.
    v_cache = cache.narrow(-1, 0, KV_LORA_RANK)
    cu_seqlens_q = torch.arange(
        batch + 1, dtype=torch.int32, device=DEVICE
    )
    # NOTE: host_kv_lens and seqused_k are MUTUALLY EXCLUSIVE.
    # flash_attn_interface.py:454-457 derives seqused_k from host_kv_lens and
    # raises if both are given.  This matters: the compact-grid split plan at
    # line 537 requires host_kv_lens, so a caller that passes seqused_k (as
    # vLLM's XPUMLAImpl.forward_mqa does) can NEVER get the split plan --
    # num_splits_kv is silently ignored on the Python side.
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    host_kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
    block_table = (
        torch.arange(blocks_per_seq * batch, dtype=torch.int32, device=DEVICE)
        .view(batch, blocks_per_seq)
        .contiguous()
    )
    return (q, k_cache, v_cache, cu_seqlens_q, seqused_k, host_kv_lens,
            block_table)


def run_once(q, k_cache, v_cache, cu_seqlens_q, seqused_k, host_kv_lens,
             block_table, max_seqlen_k, num_splits_kv, path):
    """path='seqused' mirrors vLLM today; path='hostlens' enables split plan."""
    kw = {}
    if path == "seqused":
        kw["seqused_k"] = seqused_k
    else:
        kw["host_kv_lens"] = host_kv_lens
    return flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=1,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        softmax_scale=HEAD_QK**-0.5,
        causal=False,
        return_softmax_lse=True,
        num_splits_kv=num_splits_kv,
        fa_version=2,
        **kw,
    )


_LIBC = ctypes.CDLL(None)


def call_capturing_stdout(fn):
    """Rejected configs are reported by a C++ printf on stdout, and the call
    then returns an unwritten buffer rather than raising."""
    sys.stdout.flush()
    saved = os.dup(1)
    tmp = tempfile.TemporaryFile()
    try:
        os.dup2(tmp.fileno(), 1)
        out = fn()
        torch.xpu.synchronize()
        _LIBC.fflush(None)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
    tmp.seek(0)
    msg = tmp.read().decode("utf-8", "replace")
    tmp.close()
    return out, msg


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 48])
    ap.add_argument("--seq-len", type=int, default=1100)
    ap.add_argument("--num-heads-q", type=int, default=128)
    ap.add_argument("--block-sizes", type=int, nargs="+",
                    default=[16, 32, 64])
    ap.add_argument("--splits", type=int, nargs="+",
                    default=[0, 4, 8, 16, 32])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--paths", nargs="+", default=["seqused", "hostlens"],
                    choices=["seqused", "hostlens"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50,
                    help="Raise for profiling runs, which need a long "
                         "steady-state loop to sample.")
    ap.add_argument("--bandwidth-gbs", type=float, default=1300.0,
                    help="Peak HBM BW used for the roofline column.")
    ap.add_argument("--ragged", type=float, nargs="+", default=[0.0],
                    help="Relative spread of per-seq KV lengths (0 = uniform, "
                         "0.9 = 0.1x..1.9x the mean). Batch total is held "
                         "fixed so each spread moves the same KV bytes.")
    ap.add_argument("--fixed-max", action="store_true",
                    help="treat --seq-len as the max instead of the mean; "
                         "total work then falls with --ragged")
    ap.add_argument("--show-plan", action="store_true",
                    help="Print the compact-grid split plan per config.")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    props = torch.xpu.get_device_properties(0)
    print(f"device  : {props.name}")
    print(f"max CU  : {getattr(props, 'max_compute_units', '?')}")
    print(f"shape   : heads_q={args.num_heads_q} heads_kv=1 "
          f"head_qk={HEAD_QK} head_vo={KV_LORA_RANK} "
          f"mean_seq={args.seq_len} dtype={args.dtype}")
    print("note    : the compact-grid split plan needs host_kv_lens AND an "
          "explicit num_splits_kv > 1; seqused/auto is what vLLM runs today.")
    print()

    header = (f"{'batch':>6} {'block':>6} {'ragged':>7} {'maxlen':>7} "
              f"{'path':>9} {'splits':>7} {'us':>10} {'GB/s':>9} "
              f"{'roof_us':>9} {'eff%':>6}")
    print(header)
    print("-" * len(header))

    best = {}
    for batch, block_size, ragged in itertools.product(
        args.batches, args.block_sizes, args.ragged
    ):
        seq_lens = make_seq_lens(batch, args.seq_len, ragged, args.fixed_max)
        max_len = max(seq_lens)
        try:
            inputs = build_inputs(
                batch, seq_lens, args.num_heads_q, block_size, dtype
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{batch:>6} {block_size:>6} {ragged:>7.2f} {max_len:>7} "
                  f"{'-':>9} {'-':>7} {'ALLOC FAIL':>10}  "
                  f"{str(exc).splitlines()[0][:40]}")
            continue

        if args.show_plan:
            kv_tile = _kv_tile_from_block_size(block_size)
            plan, work = build_decode_split_plan(
                seq_lens, kv_tile=kv_tile, num_kv_splits=32,
                num_xe_cores=_infer_num_xe_cores(torch.device("xpu:0")),
                num_heads_kv=1)
            print(f"       plan b={batch} ragged={ragged:.2f}: "
                  f"lens={seq_lens[:4]}{'...' if batch > 4 else ''} "
                  f"tiles={[max(1, (s + kv_tile - 1) // kv_tile) for s in seq_lens[:4]]}"
                  f"{'...' if batch > 4 else ''} "
                  f"splits={plan.tolist()[:8]}{'...' if batch > 8 else ''} "
                  f"total_wgs={work.size(0)}")

        # K and V alias one buffer, so the kernel reads sum(seq)*HEAD_QK once.
        kv_bytes = sum(seq_lens) * HEAD_QK * dtype.itemsize
        roof_us = kv_bytes / (args.bandwidth_gbs * 1e9) * 1e6

        for path, splits in itertools.product(
            args.paths, args.splits
        ):
            ns = None if splits == 0 else splits
            label = "auto" if ns is None else str(splits)
            try:
                def fn():
                    return run_once(*inputs, max_seqlen_k=max_len,
                                    num_splits_kv=ns, path=path)

                _, msg = call_capturing_stdout(fn)
                if "Invalid Problem Size" in msg:
                    print(f"{batch:>6} {block_size:>6} {ragged:>7.2f} "
                          f"{max_len:>7} {path:>9} {label:>7} "
                          f"{'REJECTED':>10}  unsupported config")
                    continue
                us = timeit(fn, warmup=args.warmup, iters=args.iters)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).splitlines()[0][:40]
                print(f"{batch:>6} {block_size:>6} {ragged:>7.2f} "
                      f"{max_len:>7} {path:>9} {label:>7} {'FAIL':>10}  {msg}")
                continue
            gbs = kv_bytes / (us * 1e-6) / 1e9
            eff = roof_us / us * 100.0
            print(f"{batch:>6} {block_size:>6} {ragged:>7.2f} {max_len:>7} "
                  f"{path:>9} {label:>7} {us:>10.1f} {gbs:>9.1f} "
                  f"{roof_us:>9.1f} {eff:>6.1f}")
            key = (batch, ragged)
            if key not in best or us < best[key][0]:
                best[key] = (us, block_size, f"{path}/{label}")

    print()
    for (batch, ragged), (us, block_size, label) in sorted(best.items()):
        print(f"best batch={batch} ragged={ragged:.2f}: {us:.1f} us "
              f"(block_size={block_size}, splits={label})")

    if len(best) > 1:
        print()
        print("Compare rows at equal batch and equal total KV bytes: any "
              "change across 'ragged' is pure split-granularity, and any "
              "seqused/hostlens gap is the compact grid.")


if __name__ == "__main__":
    main()
