# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify a native-Windows ROCm vLLM installation without loading a model."""

import argparse
import importlib
import json

import torch

import vllm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-arch",
        help=(
            "Fail unless GPU 0 reports this base GCN architecture "
            "(for example gfx1201)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.accelerator.is_available():
        raise SystemExit("Torch cannot see an AMD GPU through HIP.")
    if torch.version.hip is None:
        raise SystemExit("This is not a ROCm/HIP PyTorch build.")
    accelerator_type = torch.accelerator.current_accelerator()
    device_module = torch.get_device_module(accelerator_type)

    extensions = (
        "vllm._C",
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm._rocm_C",
    )
    for extension in extensions:
        importlib.import_module(extension)

    probe = torch.arange(16, dtype=torch.float32, device=accelerator_type)
    result = probe.mul(2).cpu().tolist()
    torch.accelerator.synchronize()
    expected = [float(value * 2) for value in range(16)]
    if result != expected:
        raise SystemExit(f"The HIP tensor probe returned wrong values: {result!r}")

    devices = []
    for index in range(torch.accelerator.device_count()):
        properties = device_module.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": device_module.get_device_name(index),
                "architecture": str(properties.gcnArchName),
                "total_memory": properties.total_memory,
            }
        )

    architecture = devices[0]["architecture"]
    architecture_base = architecture.split(":", 1)[0]
    if args.expected_arch and architecture_base != args.expected_arch:
        raise SystemExit(
            f"Expected {args.expected_arch}, but Torch reported {architecture}."
        )

    print(
        json.dumps(
            {
                "vllm": vllm.__version__,
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "devices": devices,
                "extensions": extensions,
                "tensor_probe": "passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
