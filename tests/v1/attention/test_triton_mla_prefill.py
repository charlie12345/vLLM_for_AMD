# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 Carlo Pasquale

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill import triton_mla as triton_mla_module
from vllm.v1.attention.backends.mla.prefill.triton_mla import triton_mla_prefill


def test_context_chunk_uses_current_metadata_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(triton_mla_module.TritonMLAPrefillBackend)
    backend.scale = 0.5
    q = torch.empty(2, 1, 4)
    k = torch.empty(3, 1, 4)
    v = torch.empty(3, 1, 2)
    out = torch.empty(2, 1, 2)
    chunk = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2]),
        cu_seq_lens=torch.tensor([0, 3]),
        max_query_len=2,
    )
    expected = (out, torch.empty(1, 2))
    observed: dict[str, object] = {}

    def fake_prefill(**kwargs):
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(triton_mla_module, "triton_mla_prefill", fake_prefill)

    actual = backend.run_prefill_context_chunk(chunk, q, k, v, out=out)

    assert actual is expected
    assert observed["q_start_loc"] is chunk.query_start_loc
    assert observed["k_start_loc"] is chunk.cu_seq_lens
    assert observed["max_query_len"] == chunk.max_query_len
    assert observed["out"] is out


def _reference_attention(q, k, v, q_starts, k_starts, scale, causal):
    out = torch.empty(
        (q.shape[0], q.shape[1], v.shape[-1]), dtype=v.dtype, device=q.device
    )
    lse = torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
    for seq_idx in range(q_starts.numel() - 1):
        qs, qe = q_starts[seq_idx : seq_idx + 2].tolist()
        ks, ke = k_starts[seq_idx : seq_idx + 2].tolist()
        q_seq = q[qs:qe].transpose(0, 1).float()
        k_seq = k[ks:ke].transpose(0, 1).float()
        v_seq = v[ks:ke].transpose(0, 1).float()
        scores = torch.matmul(q_seq, k_seq.transpose(-1, -2)) * scale
        if causal:
            q_pos = torch.arange(qe - qs, device=q.device) + (ke - ks) - (qe - qs)
            k_pos = torch.arange(ke - ks, device=q.device)
            scores.masked_fill_(q_pos[:, None] < k_pos[None, :], -math.inf)
        lse[:, qs:qe] = torch.logsumexp(scores, dim=-1)
        out[qs:qe] = (
            torch.matmul(torch.softmax(scores, dim=-1), v_seq)
            .transpose(0, 1)
            .to(v.dtype)
        )
    return out, lse


@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.accelerator.is_available(),
    reason="requires a ROCm GPU",
)
@pytest.mark.parametrize(
    ("q_lens", "k_lens", "causal"),
    [([5, 3], [5, 3], True), ([5, 3], [7, 2], False)],
)
def test_triton_mla_prefill_matches_torch(q_lens, k_lens, causal):
    device = torch.device(torch.accelerator.current_accelerator())
    q_starts = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device=device
    )
    k_starts = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device=device
    )
    torch.manual_seed(1)
    q = torch.randn(sum(q_lens), 4, 192, dtype=torch.bfloat16, device=device)
    k = torch.randn(sum(k_lens), 4, 192, dtype=torch.bfloat16, device=device)
    v = torch.randn(sum(k_lens), 4, 128, dtype=torch.bfloat16, device=device)
    scale = 192**-0.5

    actual_out, actual_lse = triton_mla_prefill(
        q,
        k,
        v,
        q_starts,
        k_starts,
        max(q_lens),
        scale,
        causal,
        True,
    )
    expected_out, expected_lse = _reference_attention(
        q, k, v, q_starts.cpu(), k_starts.cpu(), scale, causal
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-3, atol=2e-3)
