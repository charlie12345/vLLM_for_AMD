"""Convert a HF checkpoint to FP8 W8A8 (compressed-tensors) for vLLM on RDNA4.

FP8 e4m3fn is the fastest format this card has a matrix path for: ~236 TFLOPS
measured on gfx1201 versus ~130 for bf16. The FP8_DYNAMIC recipe is per-channel
weight scales plus per-token dynamic activation scales, so it needs no
calibration data and runs entirely on CPU -- which matters here, because the
GPU only has 32 GB and the bigger targets do not fit twice.

Run with the isolated quantization venv, not the serving one:
    .venv-quant\\Scripts\\python.exe quantize_fp8.py <model> [output_dir]

llmcompressor pins transformers<=5.10.1 while vLLM wants a newer one, which is
why the two environments are kept apart.
"""

import argparse
import os
from pathlib import Path

if not hasattr(os, "sysconf"):
    # compressed_tensors sizes the CPU offload budget with os.sysconf(), which
    # only exists on POSIX. Report the same number Windows would: total
    # physical RAM, expressed as page size x page count.
    import ctypes

    def _sysconf(name: str) -> int:
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        page_size = 4096
        if name == "SC_PAGE_SIZE":
            return page_size
        if name == "SC_PHYS_PAGES":
            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys) // page_size
        raise ValueError(f"unsupported sysconf name: {name}")

    os.sysconf = _sysconf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HF repo id or local path")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="where to write the quantized checkpoint "
        "(default: C:\\AI\\models\\<name>-FP8-Dynamic)",
    )
    args = parser.parse_args()

    out = args.output_dir
    if out is None:
        name = args.model.rstrip("/\\").split("/")[-1].split("\\")[-1]
        out = str(Path(r"C:\AI\models") / f"{name}-FP8-Dynamic")
    os.makedirs(out, exist_ok=True)

    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} on CPU in bfloat16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        # lm_head stays in bf16: it is a single large GEMM whose output feeds
        # the sampler directly, so quantizing it costs accuracy for very little
        # memory, and vLLM does not run it through the FP8 path anyway.
        ignore=["lm_head"],
    )

    print(f"Quantizing to FP8 W8A8 -> {out}")
    oneshot(model=model, recipe=recipe, output_dir=out, tokenizer=tokenizer)

    total = sum(f.stat().st_size for f in Path(out).rglob("*") if f.is_file())
    print(f"Done. {out} is {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
