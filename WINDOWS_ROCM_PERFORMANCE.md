# Native Windows ROCm performance notes

These results are for a Radeon AI PRO R9700 (gfx1201, 32 GiB), ROCm 7.13.0,
PyTorch 2.11, and models at a maximum model length of 2,048. The current source
base is vLLM v0.27.1. The GPU also
drives the desktop. Every model run below used `vram_guard.ps1` with a 26 GiB
hard limit and a per-run stall log.

## v0.27.1 update validation

The 2026-08-23 update passed the full native extension build and guarded
Torch/HIP/import probes. Binary Windows `fs_io_C` store/load and `spinloop`
smokes passed. Qwen3-0.6B FP8 generated 8 tokens at 92.21 output tokens/s with
`--max-num-batched-tokens 2048` and graph replay disabled. Qwen3-4B AWQ selected
`RDNAHybridW4A16LinearKernel`, generated 4 tokens at 9.02 output tokens/s in
eager mode, and peaked at 18.43 GiB.

An otherwise identical FP8 cold start using upstream's 8,192-token compile
warmup briefly reached 31.48 GiB for two watchdog samples. The launch wrappers
therefore default to `--max-num-batched-tokens 2048`; an explicit CLI argument
or `WINDOWS_ROCM_MAX_NUM_BATCHED_TOKENS` overrides it. The larger historical
throughput rows below predate this conservative default.

## Recommended configurations

For the local Qwen3-8B FP8 compressed-tensors checkpoint, the fastest stable
configuration measured was:

```text
--gpu-memory-utilization 0.55
--kv-cache-dtype fp8
-cc.cudagraph_mode=NONE
```

It produced 614.88 output tokens/s for 64 random prompts with 1,024 input and
128 output tokens, used 104,320 KV-cache tokens, and peaked at 20.79 GiB.
Inductor compilation remains enabled; only HIP graph capture/replay is
disabled.

For BF16 Qwen3-8B, a throughput-oriented configuration is:

```text
--gpu-memory-utilization 0.67
--kv-cache-dtype fp8
```

It produced 459.42 and 464.02 output tokens/s in repeated warm runs, held all
64 requests in 74,816 KV-cache tokens, and peaked at 24.97 and 24.79 GiB. This
is a benchmark setting with about 1 GiB of watchdog headroom. Use 0.65 for a
more conservative desktop setting; it produced 358.01 output tokens/s and
peaked at 24.11 GiB.

FP8 KV cache can affect output quality when the model does not provide tuned
KV scales. These are throughput results, not an accuracy evaluation. Run the
model's task evaluation before making FP8 KV the production default.

### Larger model without a GGUF plugin

`openai/gpt-oss-20b` works from its official Safetensors checkpoint through
vLLM's built-in `GptOssForCausalLM` and `gpt_oss_mxfp4` paths. It does not use
GGUF, a GGUF plugin, or remote model code. The local checkpoint is in
`C:\AI\models\gpt-oss-20b`.

The stable baseline is:

```text
--gpu-memory-utilization 0.55
--max-model-len 2048
-cc.cudagraph_mode=NONE
```

vLLM loaded 12.82 GiB of checkpoint shards into 14.16 GiB of GPU model memory,
selected `OAITritonMxfp4ExpertsMonolithic`, and allocated 51,741 BF16 KV-cache
tokens. A single request with 256 input and 64 output tokens produced 32.80
output tokens/s. Eight concurrent requests of the same size produced 209.62
output tokens/s and 1,048.08 total tokens/s. The two guarded runs peaked at
21.25 and 21.20 GiB, respectively.

An offline chat request also completed with HTTP 200, a normal `stop` finish,
and the correct final response. The `run-batch` frontend needs
`WindowsSelectorEventLoopPolicy` for pyzmq on native Windows. After output is
written, its cleanup currently calls the Unix-only `signal.SIGKILL`, so that
frontend exits with status 1 despite successful inference. The throughput
runner exits cleanly; these are frontend compatibility issues, not a model or
ROCm failure.

### Native AWQ INT4

`Qwen/Qwen3-4B-AWQ` works from its official 2.48 GiB Safetensors checkpoint
without a GGUF plugin. On gfx1201, compatible native AWQ layers are converted
to the modular packed format and select `RDNAHybridW4A16LinearKernel`: HIP
skinny W4A16 for decode batches up to five tokens and a gfx1201-tuned Triton
W4A16 kernel for larger batches.

With one random prompt, 64 input tokens, 16 output tokens, maximum model length
512, `--gpu-memory-utilization 0.40`, and `cudagraph_mode=NONE`, the repeated
run produced 65.80 output tokens/s and peaked at 16.35 GiB. The earlier generic
AutoAWQ path produced 30.45 output tokens/s with the same workload. These
single-request measurements are sensitive to first-shape Triton JIT; use a
repeated run and a representative serving workload before drawing broader
throughput conclusions. Deterministic offline generation produced coherent
text through the hybrid path.

The hybrid path is enabled by default for 4-bit AWQ group sizes 32, 64, and
128 on gfx11/gfx12. Set `VLLM_ROCM_USE_RDNA_W4A16=0` to retain generic AutoAWQ
for compatibility diagnosis.

## Launcher settings

`bench_windows_rocm.cmd` and `serve_windows_rocm.cmd` keep the conservative
0.55 memory default. The following optional environment variables provide
defaults when the corresponding CLI argument is absent:

- `WINDOWS_ROCM_GPU_MEMORY_UTILIZATION`
- `WINDOWS_ROCM_MAX_NUM_BATCHED_TOKENS`
- `WINDOWS_ROCM_KV_CACHE_DTYPE`
- `WINDOWS_ROCM_CUDAGRAPH_MODE`

Explicit CLI arguments take precedence. For example, an FP8 serving session
can be launched from PowerShell with:

```powershell
$env:WINDOWS_ROCM_KV_CACHE_DTYPE = 'fp8'
$env:WINDOWS_ROCM_CUDAGRAPH_MODE = 'NONE'
.\serve_windows_rocm.cmd C:\AI\models\Qwen3-8B-FP8-Dynamic --max-model-len 2048
```

Keep the server under the watchdog used in the handoff when it shares a GPU
with the desktop.

## Measurements

All throughput rows use 64 prompts, 1,024 input tokens, 128 output tokens, and
a maximum model length of 2,048.

| Weights | KV dtype | Utilization | Scheduler/backend | Output tok/s | Peak GiB |
| --- | --- | ---: | --- | ---: | ---: |
| BF16 | BF16 | 0.55 | defaults/auto | 188.30 | 20.90 |
| BF16 | BF16 | 0.60 | defaults/auto | 279.48 | 22.39 |
| BF16 | BF16 | 0.65 | defaults/auto | 325.07 | 24.15 |
| BF16 | BF16 | 0.70 | defaults/auto | 383.77 | 25.60 |
| BF16 | FP8 | 0.55 | defaults/auto | 209.80 | 20.60 |
| BF16 | FP8 | 0.65 | defaults/auto | 358.01 | 24.11 |
| BF16 | FP8 | 0.67 | defaults/auto | 459.42, 464.02 | 24.97, 24.79 |
| BF16 | FP8 | 0.68 | defaults/auto | 446.23 | 25.27 |
| BF16 | FP8 | 0.67 | defaults/TRITON_ATTN, warm | 419.37 | 24.78 |
| FP8 | BF16 | 0.55 | no graphs/auto | 527.06 | 21.63 |
| FP8 | FP8 | 0.55 | no graphs/auto | 614.88 | 20.79 |
| FP8 + 0.6B draft | FP8 | 0.55 | no graphs, draft K=3 | 63.48 | 20.79 |

BF16 utilization values from 0.40 through 0.50 cannot allocate any KV cache
after loading the 15.27 GiB checkpoint. The 0.55 run was not a batch-64 decode:
it allocated 9,568 KV tokens and scheduled about eight requests at a time.
A graph-mode profile measured 31.4 ms per decode step for that eight-request
wave. The earlier estimate of roughly 240 ms per step incorrectly treated the
eight sequential waves as one batch.

An eager operator profile ranked GEMM above direct paged attention. Across 32
profiled decode steps, `aten::mm` accumulated about 1.07 seconds while direct
`_rocm_C::paged_attention` calls accumulated about 116 ms. Profiler overhead
distorts those absolute eager times, but not enough to support attention as a
10x bottleneck.

The default scheduler settings (`max_num_seqs=256`,
`max_num_batched_tokens=8192`) performed best. Setting `max_num_seqs=64`
reduced KV residency and produced 354.72-386.87 output tokens/s across token
budgets of 4,096-16,384. Explicit `TRITON_ATTN` was about 10% slower than the
auto-selected `ROCM_ATTN` wrapper after both paths were warm.

## Known blockers

- FP8-weight graph replay still wedges after successful size-64 capture, with
  either BF16 or FP8 KV cache. The watchdog reproduced and terminated the FP8
  KV case after 241 seconds of silence at 20.88 GiB peak. Keep
  `cudagraph_mode=NONE` for this checkpoint.
- Qwen3-8B and Qwen3-0.6B do not contain native MTP heads. Generic draft-model
  speculation disabled async scheduling and the V2 runner and was almost 10x
  slower on the measured batch. N-gram speculation currently requires the
  optional `numba` dependency, which is not installed in `.venv211`.
- The current vLLM tree has no Vulkan platform or execution backend. The
  Windows work in this branch accelerates the ROCm/HIP path. Vulkan performance
  must be handled in a Vulkan-capable runtime rather than routed through these
  vLLM launchers.

Do not start tuned Triton configurations or new GEMM kernels until a workload
that remains slow after correcting KV residency demonstrates that they are the
dominant cost.
