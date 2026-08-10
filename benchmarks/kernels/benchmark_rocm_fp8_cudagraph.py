# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce FP8 ``torch._scaled_mm`` HIP graph behavior on ROCm.

Run this script through ``vram_guard.ps1`` on Windows. Channelwise mode mirrors
vLLM's unfused per-token/per-channel fallback; rowwise mode exercises the fused
hipBLASLt path that produces bf16 directly; Triton mode exercises vLLM's
compressed-tensors scaled-MM kernel and checks it against channelwise output.
"""

import argparse
import time

import torch


def make_operands(m: int, n: int, k: int):
    fp8 = torch.float8_e4m3fn
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).to(fp8)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(fp8).t()
    scale_a = torch.full((m, 1), 0.01, device="cuda", dtype=torch.float32)
    scale_b = torch.full((1, n), 0.01, device="cuda", dtype=torch.float32)
    return a, b, scale_a, scale_b


def scaled_mm(mode: str, a, b, scale_a, scale_b):
    if mode == "triton":
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
            triton_scaled_mm,
        )

        return triton_scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_b=scale_b.t(),
            out_dtype=torch.bfloat16,
        )

    if mode == "rowwise":
        return torch._scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )

    one = torch.ones(1, device="cuda", dtype=torch.float32)
    output = torch._scaled_mm(
        a,
        b,
        scale_a=one,
        scale_b=one,
        out_dtype=torch.float32,
    )
    return (output * scale_a * scale_b).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("channelwise", "rowwise", "triton"), required=True
    )
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--replays", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm GPU")

    print(
        f"ROCm {torch.version.hip} | {torch.cuda.get_device_name()} | "
        f"{args.mode=} | shape=({args.m}, {args.k}) x ({args.k}, {args.n})",
        flush=True,
    )
    a, b, scale_a, scale_b = make_operands(args.m, args.n, args.k)

    eager_output = scaled_mm(args.mode, a, b, scale_a, scale_b)
    for _ in range(args.warmups):
        scaled_mm(args.mode, a, b, scale_a, scale_b)
    torch.cuda.synchronize()
    print("eager warmup complete; capturing", flush=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = scaled_mm(args.mode, a, b, scale_a, scale_b)
    print("capture complete; replaying", flush=True)

    start = time.perf_counter()
    for replay in range(args.replays):
        graph.replay()
        torch.cuda.synchronize()
        print(f"replay {replay + 1}/{args.replays} complete", flush=True)
    elapsed = time.perf_counter() - start

    torch.testing.assert_close(graph_output, eager_output, rtol=1e-2, atol=1e-2)
    if args.mode == "triton":
        reference = scaled_mm("channelwise", a, b, scale_a, scale_b)
        torch.testing.assert_close(eager_output, reference, rtol=2e-2, atol=1e-1)
    tflops = 2 * args.m * args.n * args.k * args.replays / elapsed / 1e12
    print(f"PASS: {elapsed / args.replays * 1e3:.3f} ms, {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
