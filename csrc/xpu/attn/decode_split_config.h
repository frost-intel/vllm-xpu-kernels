#pragma once

// Decode ShapeOut / V-split configuration, shared by the device policies in
// fmha_utils.hpp and the host split heuristic in flash_attn/flash_api.cpp.
// Both must agree on how wide a work-group's V extent is: the host derives
// grid.x from it, and disagreement silently mis-sizes the split heuristic.
// Keep this header free of CUTLASS and SYCL so host translation units can
// include it.

#ifndef VLLM_DECODE_ACC_BUDGET
#define VLLM_DECODE_ACC_BUDGET 2048
#endif

// Subgroups split the V (N) dimension of P*V instead of the kv (K) dimension.
// Each subgroup then owns a disjoint V slice and needs the whole P tile, which
// the mainloop exchanges through SLM every KV tile. In return one work-group
// owns the entire V extent, so K is read once per work-group instead of once
// per V split, and the epilogue's cross-subgroup reduction disappears because
// ReduceK collapses to 1.
#ifndef VLLM_DECODE_SPLIT_V
#define VLLM_DECODE_SPLIT_V 1
#endif

// Only the xe_2 decode mainloop implements the SLM P exchange split-V needs;
// the xe_3 collective is a separate copy without it, and both instantiate the
// shared decode_policy_qpacked_head. The xe_2 kernel library defines this to 1
// (see xe_2/CMakeLists.txt); every other consumer keeps the default policy.
#ifndef VLLM_DECODE_SPLIT_V_SUPPORTED
#define VLLM_DECODE_SPLIT_V_SUPPORTED 0
#endif

// Number of 128-wide V tiles a split-V work-group owns. 4 is the full-width
// target (one work-group covers all 512 V, so K is read once); 1 keeps the
// output tile equal to the P*V MMA tile and was used to bisect mainloop
// correctness from the epilogue's VTiles handling.
#ifndef VLLM_DECODE_SPLITV_VTILES
#define VLLM_DECODE_SPLITV_VTILES 4
#endif

namespace vllm_xpu {

inline constexpr int kDecodeAccBudget = VLLM_DECODE_ACC_BUDGET;

// Largest ShapeOut V extent the non-split-V decode policies will use. Only
// head_dim > this is split across grid.x.
inline constexpr int kDecodeMaxShapeOutV = 256;

// Subgroups per kv_tile=_64 decode work-group, and the V width of one P*V MMA
// tile -- the granularity a split-V work-group's V extent is floored to.
inline constexpr int kDecodeSplitVSubgroups = 4;
inline constexpr int kDecodeSplitVTileV = 128;

// Only the kv_tile=_64 policy splits V; callers must check the block size.
// This is the arch-agnostic answer, used by the host heuristic, which must
// additionally establish that the dispatch lands on the xe_2 kernels.
constexpr bool decode_splits_v(int head_dim) {
  return VLLM_DECODE_SPLIT_V != 0 && head_dim > kDecodeMaxShapeOutV;
}

// Whether the policies compiled into *this* translation unit split V.
constexpr bool decode_policy_splits_v(int head_dim) {
  return VLLM_DECODE_SPLIT_V_SUPPORTED != 0 && decode_splits_v(head_dim);
}

constexpr int decode_shapeout_v(int q_packed, int head_dim) {
  if (head_dim <= kDecodeMaxShapeOutV) return head_dim;
  const int budgeted = kDecodeAccBudget / q_packed;
  return budgeted < kDecodeMaxShapeOutV ? budgeted : kDecodeMaxShapeOutV;
}

// V extent owned by one work-group when subgroups split V. Bounded by the same
// per-subgroup accumulator budget as the default policy, times the subgroups
// now sharing it, and floored to a whole number of P*V V-tiles.
constexpr int decode_splitv_shapeout_v(int q_packed, int head_dim) {
  const int budgeted = kDecodeSplitVSubgroups * kDecodeAccBudget / q_packed;
  const int capped = budgeted < head_dim ? budgeted : head_dim;
  const int full = (capped / kDecodeSplitVTileV) * kDecodeSplitVTileV;
  const int capped_tiles = VLLM_DECODE_SPLITV_VTILES * kDecodeSplitVTileV;
  return full < capped_tiles ? full : capped_tiles;
}

}  // namespace vllm_xpu
