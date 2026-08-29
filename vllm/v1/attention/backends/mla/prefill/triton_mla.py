# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Carlo Pasquale
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Triton MLA prefill fallback for ROCm.

This backend is intentionally selected after the tuned AITER and
FlashAttention implementations.  It keeps prefill on the GPU on ROCm devices
where neither tuned backend supports the device or is installed.  Unlike the
generic Triton prefill kernel, it supports MLA's different Q/K and V head
dimensions and can return LSE values for chunked-context merging.
"""

from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability


@triton.jit
def _triton_mla_prefill_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    QStartLoc,
    KStartLoc,
    scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_out_t,
    stride_oh,
    stride_od,
    stride_lseh,
    stride_lset,
    KV_GROUP_NUM: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_V: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)
    kv_head_idx = q_head_idx // KV_GROUP_NUM

    q_start = tl.load(QStartLoc + seq_idx)
    q_len = tl.load(QStartLoc + seq_idx + 1) - q_start
    k_start = tl.load(KStartLoc + seq_idx)
    k_len = tl.load(KStartLoc + seq_idx + 1) - k_start

    offs_m = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_qk = tl.arange(0, BLOCK_QK)
    offs_v = tl.arange(0, BLOCK_V)

    q_ptrs = (
        Q
        + (q_start + offs_m[:, None]) * stride_qt
        + q_head_idx * stride_qh
        + offs_qk[None, :] * stride_qd
    )
    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < q_len) & (offs_qk[None, :] < QK_HEAD_DIM),
        other=0.0,
    )

    # Scores are kept in base-2 units for Triton's exp2 implementation.
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_V], dtype=tl.float32)

    end_n = k_len
    if IS_CAUSAL:
        # Bottom-right causal alignment also handles q_len != k_len.
        end_n = tl.minimum(k_len, (q_block_idx + 1) * BLOCK_M + k_len - q_len)
    end_n = tl.maximum(end_n, 0)

    for start_n in range(0, end_n, BLOCK_N):
        pos_q = offs_m[:, None] + k_len - q_len
        pos_k = start_n + offs_n[None, :]
        valid_k = pos_k < k_len
        mask = valid_k
        if IS_CAUSAL:
            mask &= pos_q >= pos_k

        k_ptrs = (
            K
            + (k_start + start_n + offs_n[None, :]) * stride_kt
            + kv_head_idx * stride_kh
            + offs_qk[:, None] * stride_kd
        )
        k = tl.load(
            k_ptrs,
            mask=valid_k & (offs_qk[:, None] < QK_HEAD_DIM),
            other=0.0,
        )

        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * scale * 1.4426950408889634, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc *= alpha[:, None]

        v_ptrs = (
            V
            + (k_start + start_n + offs_n[:, None]) * stride_vt
            + kv_head_idx * stride_vh
            + offs_v[None, :] * stride_vd
        )
        value = tl.load(
            v_ptrs,
            mask=(start_n + offs_n[:, None] < k_len) & (offs_v[None, :] < V_HEAD_DIM),
            other=0.0,
        )
        acc = tl.dot(p.to(value.dtype), value, acc)
        m_i = m_ij

    has_keys = l_i > 0.0
    acc /= tl.where(has_keys[:, None], l_i[:, None], 1.0)
    out_ptrs = (
        Out
        + (q_start + offs_m[:, None]) * stride_out_t
        + q_head_idx * stride_oh
        + offs_v[None, :] * stride_od
    )
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_m[:, None] < q_len) & (offs_v[None, :] < V_HEAD_DIM),
    )

    if RETURN_LSE:
        lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
        lse = tl.where(has_keys, lse, -float("inf"))
        lse_ptrs = Lse + q_head_idx * stride_lseh + (q_start + offs_m) * stride_lset
        tl.store(lse_ptrs, lse, mask=offs_m < q_len)


def triton_mla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_start_loc: torch.Tensor,
    k_start_loc: torch.Tensor,
    max_query_len: int,
    scale: float,
    causal: bool,
    return_lse: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run packed variable-length MLA attention without host payload copies."""
    assert HAS_TRITON
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[-1] == k.shape[-1]
    assert k.shape[1] == v.shape[1]
    assert q.shape[1] % k.shape[1] == 0
    assert q_start_loc.ndim == k_start_loc.ndim == 1
    assert q_start_loc.shape == k_start_loc.shape
    assert q.device == k.device == v.device
    assert q_start_loc.device == q.device and k_start_loc.device == q.device
    assert q.dtype == k.dtype == v.dtype

    if out is None:
        out = torch.empty(
            (q.shape[0], q.shape[1], v.shape[-1]),
            dtype=v.dtype,
            device=v.device,
        )
    else:
        assert out.shape == (q.shape[0], q.shape[1], v.shape[-1])
        assert out.device == q.device and out.dtype == v.dtype

    lse = torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
    batch_size = q_start_loc.shape[0] - 1
    block_m = 64
    block_n = 64
    grid = (batch_size, q.shape[1], triton.cdiv(max_query_len, block_m))
    _triton_mla_prefill_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        q_start_loc,
        k_start_loc,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        KV_GROUP_NUM=q.shape[1] // k.shape[1],
        QK_HEAD_DIM=q.shape[-1],
        V_HEAD_DIM=v.shape[-1],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_QK=triton.next_power_of_2(q.shape[-1]),
        BLOCK_V=triton.next_power_of_2(v.shape[-1]),
        IS_CAUSAL=causal,
        RETURN_LSE=return_lse,
        num_warps=4,
        num_stages=1,
    )
    return (out, lse) if return_lse else out


class TritonMLAPrefillBackend(MLAPrefillBackend):
    """Correctness-first GPU MLA prefill fallback for ROCm."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return current_platform.is_rocm()

    @classmethod
    def is_available(cls) -> bool:
        return current_platform.is_rocm() and HAS_TRITON

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert output_scale is None, (
            "TritonMLAPrefillBackend does not support fused quantized output."
        )
        return triton_mla_prefill(
            q=q,
            k=k,
            v=v,
            q_start_loc=self._prefill_metadata.query_start_loc,
            k_start_loc=self._prefill_metadata.query_start_loc,
            max_query_len=self._prefill_metadata.max_query_len,
            scale=self.scale,
            causal=True,
            return_lse=return_softmax_lse,
            out=out,
        )

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = triton_mla_prefill(
            q=q,
            k=k,
            v=v,
            q_start_loc=chunk.query_start_loc,
            k_start_loc=chunk.cu_seq_lens,
            max_query_len=chunk.max_query_len,
            scale=self.scale,
            causal=False,
            return_lse=True,
            out=out,
        )
        assert isinstance(result, tuple)
        return result
