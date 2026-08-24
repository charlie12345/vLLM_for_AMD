# vLLM for AMD on Native Windows

> [!IMPORTANT]
> **Thank you, AMD**
>
> AMD's Threadripper platform and Radeon AI PRO R9700 hardware made this
> native-Windows ROCm project possible.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D4)](#requirements)
[![Accelerator](https://img.shields.io/badge/accelerator-AMD%20ROCm%2FHIP-ED1C24)](#how-it-works)
[![Status](https://img.shields.io/badge/status-experimental-orange)](#project-status)

Run the real vLLM engine directly on an AMD GPU in native Windows: no WSL,
Linux VM, Docker container, CUDA translation layer, or GGUF plugin required.

This is an experimental, community-maintained fork of
[vLLM](https://github.com/vllm-project/vllm), currently based on vLLM v0.27.1.
It is validated on Windows 11 with a Radeon AI PRO R9700 (RDNA4, `gfx1201`) and
AMD's Windows ROCm/PyTorch wheels. It provides an OpenAI-compatible server,
offline benchmarks, FP8/MXFP4 support where vLLM has a compatible kernel, and a
fail-closed VRAM watchdog for desktop GPUs.

> [!WARNING]
> This is not an official AMD or vLLM release. The tested stack is version
> pinned, single-GPU only, and still experimental. Read
> [VRAM safety](#vram-safety) before loading a model.

## Project status

| Item | Current status |
| --- | --- |
| Host OS | Native Windows 11 x64 |
| Tested GPU | AMD Radeon AI PRO R9700, RDNA4 `gfx1201`, 32 GiB |
| vLLM base | v0.27.1, base commit `6e448d0` |
| Python | 3.12.10 |
| PyTorch | 2.11.0 + ROCm 7.13.0 |
| ROCm SDK | 7.13.0 Python wheels, including development tools |
| Triton | `triton-windows` 3.6.0.post26 |
| Serving | OpenAI-compatible HTTP API |
| Parallelism | One process, one GPU; no tensor/pipeline multi-GPU |
| Model files | Hugging Face/Safetensors formats; no GGUF plugin included |
| Vulkan | Not used by this vLLM backend |

Other AMD architectures may work after selecting matching ROCm wheels and
changing the build target, but they are not validated by this repository yet.
The checked-in build environment currently targets `gfx1201` explicitly.

## Why this Windows port is different

Upstream vLLM supports AMD ROCm, but its official GPU requirements list Linux
and explicitly state that vLLM does not support Windows natively. This fork
keeps vLLM's ROCm/HIP execution path and native extensions instead of replacing
the engine with llama.cpp, ONNX Runtime, DirectML, or a Vulkan backend.

The comparison below was checked on 2026-08-09. These projects continue to
evolve, so follow the links for their latest status.

| Project | OS and accelerator | Architectural difference |
| --- | --- | --- |
| [Upstream vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) | Official GPU requirements list Linux; no native Windows support | This is the source project. This fork adds native-Windows build and runtime compatibility while retaining its ROCm platform. |
| [ThePie88/vLLM-ROCm-Windows](https://github.com/ThePie88/vLLM-ROCm-Windows) | Native Windows, AMD ROCm, primarily tested on RDNA3 `gfx1100` | An out-of-tree vLLM platform plugin and compatibility layer. It installs upstream vLLM without its normal kernel build, then separately loads selected native/custom kernels. |
| This repository | Native Windows, AMD ROCm, tested on RDNA4 `gfx1201` | An in-tree vLLM fork that builds vLLM's HIP extensions directly, patches Windows process/runtime assumptions, and ships safe AMD-tuned launchers plus a VRAM/stall watchdog. |

This project is therefore not the only native-Windows AMD experiment. Its
distinctive scope is an integrated, RDNA4-tested, in-tree vLLM ROCm build with
operational safety tooling.

## How it works

```text
OpenAI client / benchmark
          |
          v
vLLM V1 scheduler, PagedAttention, continuous batching, prefix cache
          |
          +--> Windows process, TCP/ZMQ and event-loop compatibility
          +--> single-rank c10d stand-in (fails if real peers are requested)
          +--> torch.compile / Inductor / Triton
          |
          v
PyTorch ROCm -> HIP / hipBLASLt -> AMD Radeon GPU

Windows GPU performance counters ---> vram_guard.ps1 ---> launched tree only
run-specific log timestamps --------> stall detector -----^
```

The important porting work includes:

- Building vLLM's C++/HIP extensions with `clang-cl`, MSVC, CMake, and the ROCm
  SDK delivered inside AMD's Python wheels.
- Normalizing Windows paths before CMake interprets them and providing MSVC
  equivalents for POSIX-only source APIs.
- Detecting ROCm devices through `torch.cuda` when `amdsmi` is unavailable on
  Windows.
- Installing a single-rank `torch.distributed` stand-in because the tested AMD
  Windows PyTorch wheel has no c10d extension. Operations requiring another
  rank fail instead of silently returning incorrect results.
- Replacing Unix IPC assumptions with loopback TCP/ZMQ and handling Windows
  multiprocessing handles correctly.
- Selecting `winloop` or standard `asyncio` when POSIX-only `uvloop` is absent.
- Preferring hipBLASLt on RDNA4. On the test card, the untuned rocBLAS path was
  dramatically slower for BF16/FP16 GEMMs.
- Keeping Inductor and Triton caches in short paths to avoid Windows `MAX_PATH`
  failures.
- Monitoring total dedicated VRAM and log progress independently of vLLM, then
  terminating only the process tree launched by the watchdog.

See [WINDOWS_ROCM_PERFORMANCE.md](WINDOWS_ROCM_PERFORMANCE.md) for measured
results, profiles, and current kernel limitations.

## Requirements

### Hardware and operating system

- Windows 11 x64.
- An AMD GPU supported by the Windows ROCm/PyTorch release you install. Check
  AMD's current [Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
  and [HIP SDK system requirements](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/shared/hipsdk/reference/system-requirements.html).
- This branch is tested only on `gfx1201`: Radeon AI PRO R9700 and the same
  RDNA4 target family.
- Enough system RAM and disk for the checkpoint plus build intermediates.
  Thirty-two GiB of system RAM is a practical floor for smaller models; 64 GiB
  or more is recommended for larger checkpoints and source builds.

### Software

| Component | Tested version | Why it is needed |
| --- | --- | --- |
| AMD Radeon Software driver | A version matched to the selected ROCm release | Provides the Windows display/compute driver and HIP device access. |
| Python | 3.12 x64 | Matches the tested AMD PyTorch wheels. |
| Git | Current Windows release | Clones and updates the source tree. |
| [uv](https://docs.astral.sh/uv/) | Current release | Creates the isolated environment and installs Python packages. |
| Visual Studio 2022 Build Tools | MSVC v143, Desktop development with C++, Windows SDK | Supplies the Windows compiler, linker, headers, and libraries. The current script expects the standard Build Tools installation path. |
| CMake | 4.4.2 | Required for the tested `clang-cl` HIP configuration. CMake 3.31 failed during configuration. |
| Ninja | 1.13 | Drives the native extension build. |
| ROCm SDK development wheels | 7.13.0 | Supply HIP, Clang, device libraries, headers, and math libraries inside the Python environment. |
| PyTorch ROCm wheel | 2.11.0 + ROCm 7.13.0 | Tensor runtime and AMD GPU integration. |
| `triton-windows` | 3.6.0.post26 | Compiles vLLM's Triton kernels on native Windows. |

AMD notes that the complete Linux ROCm stack is not present on Windows. This
project uses the Windows-supported HIP/ROCm components packaged with AMD's
PyTorch and ROCm SDK wheels. AMD's [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
document the current Windows packages and GPU-specific `device-*` targets.
The tested Torch 2.11/ROCm 7.13 packages are newer than AMD's current stable
Windows support matrix and should be treated as a pinned preview stack.

### Do I need Vulkan?

No. This repository uses ROCm/HIP, not Vulkan. Installing the Vulkan SDK will
not enable or accelerate vLLM here.

The AMD display driver normally supplies the Vulkan runtime for applications
that use it. Install the Vulkan SDK only if you are developing or compiling a
separate Vulkan application, such as a Vulkan-enabled llama.cpp build. Do not
install Vulkan as a substitute for the ROCm/PyTorch stack above.

## Installation

The commands below reproduce the stack tested in this repository. Package
indexes change over time; do not mix arbitrary Torch, ROCm, Triton, and vLLM
versions and expect ABI compatibility.

### 1. Install the system tools

Install:

- A compatible AMD Radeon Software driver.
- Python 3.12 x64.
- Git for Windows.
- Visual Studio 2022 Build Tools with **Desktop development with C++**, MSVC
  v143, and a Windows 10 or 11 SDK.
- `uv` from its [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Reboot after installing or changing the AMD driver.

### 2. Clone into a short path

PowerShell:

```powershell
New-Item -ItemType Directory -Path C:\AI -Force | Out-Null
git clone https://github.com/charlie12345/vLLM_for_AMD.git C:\AI\vllm
Set-Location C:\AI\vllm
```

A short path matters when Torch/Inductor generates deeply nested cache names.

### Automated setup (recommended)

After installing the system tools and cloning the repository, run the
repository-local setup from PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows_rocm.ps1 -PlanOnly
.\setup_windows_rocm.ps1
```

`-PlanOnly` checks Windows, the AMD adapter, Git, `uv`, Visual Studio, system
RAM, disk headroom, and repository files without installing packages, opening a
GPU context, or building anything. The full run then:

1. Creates or reuses `.venv211` with Python 3.12.
2. Installs the exact ROCm 7.13, PyTorch 2.11, and Triton packages below.
3. Verifies Torch/HIP and requires `gfx1201` through the fail-closed VRAM guard.
4. Builds and editable-installs the native Windows ROCm extensions.
5. Runs a second guarded vLLM import and device verification.

The script never downloads or launches a model. It does not silently install or
update the AMD driver, Visual Studio, Git, or `uv`; review and install those
system-wide prerequisites yourself. The manual commands below remain available
for troubleshooting and auditing the automated process.

### 3. Create the serving environment

```powershell
uv venv --python 3.12 .venv211

uv pip install --python .\.venv211\Scripts\python.exe `
  --extra-index-url https://repo.amd.com/rocm/whl/gfx120X-all/ `
  --index-strategy unsafe-best-match `
  "torch==2.11.0+rocm7.13.0" `
  "torchvision==0.26.0+rocm7.13.0" `
  "rocm[devel]==7.13.0"

uv pip install --python .\.venv211\Scripts\python.exe `
  "triton-windows==3.6.0.post26" winloop

uv pip install --python .\.venv211\Scripts\python.exe `
  -r .\requirements\common.txt

uv pip install --python .\.venv211\Scripts\python.exe `
  "cmake==4.4.2" ninja "packaging>=24.2" `
  "setuptools>=77,<80" "setuptools-scm>=8" `
  "setuptools-rust>=1.9" wheel "jinja2>=3.1.6"
```

Do **not** install `requirements/rocm.txt` wholesale on Windows. It contains
Linux-only or unsupported packages such as TileLang, RunAI model streaming,
FastSafetensors, AMD Quark, and other optional integrations. The commands above
install the tested core serving stack while preserving AMD's ROCm Torch wheel.

### 4. Verify ROCm before building vLLM

```powershell
.\.venv211\Scripts\python.exe -m rocm_sdk test

.\.venv211\Scripts\python.exe -c `
  "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).gcnArchName)"
```

For the tested card, the final line must report `gfx1201`. Stop here if Torch
cannot see the expected AMD GPU.

### 5. Build and install this fork

```powershell
.\build_windows_rocm.cmd
.\install_windows_rocm.cmd
```

These scripts load the Visual Studio environment, find ROCm through
`python -m rocm_sdk path --root`, select `clang-cl` for both C++ and HIP, build
for `gfx1201`, and install vLLM into `.venv211` in editable mode.

Verify the install without loading a model:

```powershell
.\.venv211\Scripts\python.exe -c "import vllm; print(vllm.__version__)"
```

## Choosing and downloading a model

Use a Hugging Face model architecture supported by
[upstream vLLM](https://docs.vllm.ai/en/latest/models/supported_models.html), in
Safetensors or another native vLLM format. A model being popular does not mean
its particular quantization has a fast AMD RDNA4 kernel.

### Models tested on this branch

| Model | Format | Purpose and notes |
| --- | --- | --- |
| [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | BF16 or locally converted FP8 | Small smoke-test model. |
| [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | BF16; locally tested with compressed-tensors FP8 | Good general baseline. Disable graph replay for the tested FP8-weight checkpoint. |
| [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | Official MXFP4 Safetensors | Recommended larger native model. It loaded through built-in `GptOssForCausalLM`/`gpt_oss_mxfp4`, used 14.16 GiB for model memory, and needed no GGUF plugin or remote code. |

Download a small public model:

```powershell
New-Item -ItemType Directory -Path C:\AI\models -Force | Out-Null
.\.venv211\Scripts\hf.exe download Qwen/Qwen3-0.6B `
  --local-dir C:\AI\models\Qwen3-0.6B
```

Download the tested larger model without its duplicate reference directories:

```powershell
.\.venv211\Scripts\hf.exe download openai/gpt-oss-20b `
  --local-dir C:\AI\models\gpt-oss-20b `
  --exclude "original/*" "metal/*"
```

For a gated model, authenticate interactively instead of placing a Hugging
Face token in a script or command line:

```powershell
.\.venv211\Scripts\hf.exe auth login
```

The token is stored in the user's Hugging Face cache, outside this repository.
See the official [Hugging Face authentication documentation](https://huggingface.co/docs/huggingface_hub/package_reference/authentication).

### Model-format guidance

| Format | RDNA4 status in this fork |
| --- | --- |
| BF16/FP16 Safetensors | Supported when the weights plus runtime/KV memory fit. BF16 is the safer default. |
| GPT-OSS MXFP4 | Supported through vLLM's built-in Triton MXFP4 MoE path. |
| FP8 compressed-tensors | Supported, but graph replay can hang on some larger FP8 shapes; start with `cudagraph_mode=NONE`. |
| FP8 KV cache | Supported and roughly halves KV memory. Validate output quality because untuned default scales can affect accuracy. |
| AWQ INT4 | Native 4-bit AWQ Safetensors with group size 32/64/128 use the hybrid RDNA W4A16 path on gfx11/gfx12: HIP skinny GEMM for decode and tuned Triton GEMM for larger batches. `Qwen/Qwen3-4B-AWQ` is validated on gfx1201. |
| GPTQ INT4 | Kernel-dependent. The native AWQ validation does not establish that every GPTQ packing or activation-order configuration works on gfx1201. |
| GGUF | Not included in core vLLM here. The official GGUF path is an out-of-tree plugin, which this project does not install. Prefer native Safetensors checkpoints. |
| `trust_remote_code` models | Potentially supported, but review the repository code first. Never enable the flag for an untrusted model. |

As a conservative rule on a display-connected GPU, keep several GiB free for
Windows and temporary activations. Checkpoint file size is not the same as peak
VRAM use. Model weights, KV cache, activations, compiler workspaces, and the
desktop all share the card.

## Quick start: guarded OpenAI-compatible server

The launchers default to `--gpu-memory-utilization 0.55`. The watchdog adds an
independent absolute VRAM ceiling.

Open PowerShell in the repository and run:

```powershell
$root = 'C:\AI\vllm'
$model = 'C:\AI\models\gpt-oss-20b'
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null

$serverLog = Join-Path $logs 'gpt-oss-server.log'
$guardLog = Join-Path $logs 'gpt-oss-server.guard.log'
$command = "`"$root\serve_windows_rocm.cmd`" `"$model`" " +
  "--served-model-name gpt-oss-20b --host 127.0.0.1 --port 8000 " +
  "--max-model-len 2048 --gpu-memory-utilization 0.55 " +
  "-cc.cudagraph_mode=NONE > `"$serverLog`" 2>&1"

& "$root\vram_guard.ps1" `
  -Command $command `
  -LimitGiB 26 `
  -WarnGiB 23 `
  -LogPath $guardLog
```

For a long-lived server, the example intentionally omits `-StallLogPath`: an
idle API server can legitimately produce no log output. The absolute VRAM
limit remains active. Use stall detection for finite benchmarks and conversion
jobs.

The guard prints the exact child PID. To stop the server, terminate that exact
tree from another terminal rather than killing every Python process on the
machine:

```powershell
$childPid = Read-Host "Enter the child PID printed by vram_guard.ps1"
taskkill /PID $childPid /T /F
```

Do not close or interrupt the guard and leave its child running. After a forced
stop, verify any remaining process by command line before terminating it.

### Send a chat request

Run this in a second PowerShell window:

```powershell
$body = @{
  model = 'gpt-oss-20b'
  messages = @(
    @{ role = 'user'; content = 'Explain PagedAttention in two sentences.' }
  )
  max_tokens = 128
  temperature = 0
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/v1/chat/completions `
  -Method Post `
  -ContentType 'application/json' `
  -Body $body
```

## Benchmarking

Every model benchmark should use a fresh log and the watchdog:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = "C:\AI\vllm\logs\qwen-smoke-$stamp.log"
$guardLog = "C:\AI\vllm\logs\qwen-smoke-$stamp.guard.log"
$command = "C:\AI\vllm\bench_windows_rocm.cmd throughput " +
  "--model C:\AI\models\Qwen3-0.6B --tokenizer C:\AI\models\Qwen3-0.6B " +
  "--dataset-name random --num-prompts 8 " +
  "--random-input-len 256 --random-output-len 64 " +
  "--max-model-len 2048 --gpu-memory-utilization 0.55 " +
  "> `"$log`" 2>&1"

.\vram_guard.ps1 `
  -Command $command `
  -LimitGiB 26 `
  -WarnGiB 23 `
  -StallLogPath $log `
  -StallSec 300 `
  -LogPath $guardLog
```

The first run for a new model/batch shape may include Triton and Inductor JIT
compilation. Repeat the identical workload before calling a result steady
state.

## VRAM safety

`--gpu-memory-utilization` and `vram_guard.ps1` solve different problems:

- The launcher setting tells vLLM how much GPU memory it may plan to use for
  weights, activations, graphs, and KV cache. The wrappers inject `0.55` unless
  the command already specifies a value.
- The watchdog reads Windows' dedicated-GPU-memory performance counter. It sees
  total adapter usage, including the desktop and other applications, and can
  terminate only the process tree it launched.

The watchdog fails closed:

- Exit `98`: Windows GPU memory counters were unavailable, so the command was
  not launched.
- Exit `99`: the process tree crossed the VRAM limit, stopped producing log
  output for the configured interval, or the counter repeatedly failed.
- Any other exit code is the launched command's exit code.

### Changing the limits for another GPU

You normally do not need to edit `vram_guard.ps1`. Pass limits per run:

```powershell
.\vram_guard.ps1 `
  -Command $command `
  -LimitGiB 40 `
  -WarnGiB 36 `
  -StallLogPath $log `
  -StallSec 420 `
  -LogPath $guardLog
```

Choose values from measurements, not only the advertised card capacity:

1. Measure the idle Windows desktop VRAM use.
2. Reserve at least 4-6 GiB on a display-connected card; reserve more if other
   GPU applications remain open.
3. Set `LimitGiB` to the highest **total dedicated usage** you will tolerate.
4. Set `WarnGiB` 2-4 GiB below that limit.
5. Start vLLM at `--gpu-memory-utilization 0.55`, confirm peak usage, then move
   upward in small steps. More KV cache is not automatically faster.

For example, `26/23 GiB` is measured and conservative on the 32 GiB R9700 that
also drives the desktop. A headless compute card or a card with more VRAM may
use a higher absolute limit, but still needs workspace and OS headroom.

Change the launcher's default without editing it:

```powershell
$env:WINDOWS_ROCM_GPU_MEMORY_UTILIZATION = '0.65'
$env:WINDOWS_ROCM_KV_CACHE_DTYPE = 'fp8'
$env:WINDOWS_ROCM_CUDAGRAPH_MODE = 'NONE'
```

Explicit command-line flags take precedence over these variables.

> [!CAUTION]
> `vram_guard.ps1` monitors dedicated GPU memory, not system RAM. Windows may
> spill GPU allocations into shared system memory before an ordinary CUDA/HIP
> free-memory query looks obviously wrong. Keep real dedicated-VRAM headroom.

## Useful flags

| Flag | Recommended starting point | Effect |
| --- | --- | --- |
| `--gpu-memory-utilization` | `0.55` | vLLM's planned fraction of GPU memory. Raise only after guarded measurements. |
| `--max-model-len` | `2048` for the first test | Caps context length and makes initial memory requirements predictable. |
| `--kv-cache-dtype` | `auto`/BF16 first; test `fp8` later | FP8 roughly halves KV memory, but quality should be evaluated. |
| `-cc.cudagraph_mode=NONE` | Use for known-problem FP8 checkpoints | Keeps Torch/Inductor compilation but disables HIP graph capture/replay. |
| `--attention-backend` | Leave on `auto` | Allows the ROCm platform to choose. `TRITON_ATTN` is useful for an explicit A/B test. |
| `--max-num-seqs` | Leave at the vLLM default initially | Limits concurrent sequences. The default beat manually reduced values in the measured Qwen workload. |
| `--max-num-batched-tokens` | Leave at the vLLM default initially | Controls scheduler token capacity; tune with a representative workload. |
| `--served-model-name` | A short stable API name | Avoids requiring clients to send an absolute Windows model path. |
| `--host` | `127.0.0.1` | Keeps the server local. Binding `0.0.0.0` exposes it to the network. |
| `--trust-remote-code` | Off | Executes Python from the model repository. Enable only after reviewing trusted code. |

On a machine with multiple GPUs, select one before starting vLLM:

```powershell
$env:HIP_VISIBLE_DEVICES = '0'
```

This does not enable multi-GPU vLLM; it selects the one visible GPU used by the
single-rank runtime.

## Measured results

Radeon AI PRO R9700, Windows 11, maximum model length 2,048:

| Model and workload | Configuration | Output throughput | Peak dedicated VRAM |
| --- | --- | ---: | ---: |
| Qwen3-8B FP8, 64 × 1,024 input / 128 output | FP8 KV, util 0.55, no graphs | 614.88 tok/s | 20.79 GiB |
| Qwen3-8B BF16, 64 × 1,024 input / 128 output | FP8 KV, util 0.67 | 459.42-464.02 tok/s | 24.79-24.97 GiB |
| GPT-OSS-20B, 1 × 256 input / 64 output | BF16 KV, util 0.55, no graphs | 32.80 tok/s | 21.25 GiB |
| GPT-OSS-20B, 8 × 256 input / 64 output | BF16 KV, util 0.55, no graphs | 209.62 tok/s | 21.20 GiB |

These are throughput checks, not model-quality evaluations. See
[WINDOWS_ROCM_PERFORMANCE.md](WINDOWS_ROCM_PERFORMANCE.md) for methodology and
known caveats.

## Security and credentials

Do not commit API keys, Hugging Face tokens, model-provider credentials, TLS
private keys, `.env` files, logs, profiles, or downloaded model weights.

This repository's `.gitignore` excludes common local secret files and generated
run artifacts, but ignore rules are not a security boundary. In particular,
`vram_guard.ps1` records the launched command, so never put a real token in
`--api-key` or another command-line flag.

If the API must require a key, read it interactively into an environment
variable before launching the guard:

```powershell
$secure = Read-Host 'Local vLLM API key' -AsSecureString
$env:VLLM_API_KEY = [System.Net.NetworkCredential]::new('', $secure).Password
Remove-Variable secure
```

The child process inherits the variable without the value appearing in the
guard's logged command. Keep `--host 127.0.0.1` unless you have intentionally
configured authentication, Windows Firewall, and network access controls.

Before every commit and push:

```powershell
git status --short
git diff --cached
git grep -n -I -E "(hf_|ghp_|github_pat_|AKIA|BEGIN .*PRIVATE KEY|sk-)" -- .
```

Review every match; some source identifiers can be false positives. Enable
[GitHub secret scanning and push protection](https://docs.github.com/en/code-security/concepts/secret-security/push-protection)
on the repository as a second layer. If a real credential is ever committed,
rotate or revoke it immediately; deleting the working-tree file does not remove
it from Git history.

## Building for a different AMD architecture

The current `env_windows_rocm.cmd` contains two explicit `gfx1201` settings:

- `PYTORCH_ROCM_ARCH=gfx1201`
- `-DCMAKE_HIP_ARCHITECTURES=gfx1201`

To experiment with another GPU:

1. Find its GFX target in AMD's support matrix or with a working ROCm Torch
   install.
2. Install the matching AMD ROCm/PyTorch device wheels. Do not use `gfx120X-all`
   for an unrelated architecture.
3. Change both build targets above to the same GFX value.
4. Delete only the repository's generated build directory, rebuild, and start
   with a very small model under conservative guard limits.

Architecture support in a ROCm wheel does not guarantee that every vLLM
attention or quantization kernel supports that architecture. Treat all targets
other than `gfx1201` as unverified until they pass model-load, coherent-output,
VRAM, and throughput tests.

## Known limitations and troubleshooting

- **Single GPU only:** Windows ROCm Torch lacks the tested c10d/RCCL stack. The
  compatibility layer supports rank 0/world size 1 and rejects real peer work.
- **Expected `amdsmi` warning:** Windows has no compatible `amdsmi` package in
  this stack. The port falls back to Torch for device detection and properties.
- **CMake must be new enough:** the tested HIP + `clang-cl` configuration needs
  CMake 4.4.2. Upstream's ROCm build requirements currently constrain CMake
  below 4, so do not install that requirements file unchanged.
- **Short cache paths matter:** the launchers default to `C:\AI\vc` and
  `C:\AI\vt`. Override them with `VLLM_CACHE_ROOT` and `TRITON_CACHE_DIR`, but
  keep them short unless Windows long-path support is enabled.
- **FP8-weight graph replay:** a tested Qwen3-8B FP8 checkpoint can wedge during
  HIP graph replay. Use `-cc.cudagraph_mode=NONE`; the watchdog catches the
  silent stall.
- **Cold-start compilation:** new models and batch shapes can trigger Triton or
  Inductor compilation. Measure the repeated run.
- **`run-batch` frontend:** native Windows may require
  `WindowsSelectorEventLoopPolicy` for pyzmq, and current cleanup contains a
  Unix-only `SIGKILL` call. Output can be valid even if shutdown returns status
  1. The throughput runner exits cleanly.
- **Vulkan is not a fallback:** if ROCm fails, installing Vulkan does not repair
  this vLLM backend.
- **AWQ compatibility is format-specific:** validated native AWQ uses 4-bit
  weights, group size 32/64/128, FP16/BF16 activations, and no activation-order
  index. Set `VLLM_ROCM_USE_RDNA_W4A16=0` to fall back to generic AutoAWQ while
  diagnosing an unsupported checkpoint. GPTQ and other INT4 layouts remain
  kernel-dependent.

## Repository scripts

| File | Purpose |
| --- | --- |
| `setup_windows_rocm.ps1` | Checks prerequisites, installs the pinned environment, runs guarded GPU probes, builds, and verifies the fork. |
| `env_windows_rocm.cmd` | Shared Visual Studio, ROCm, compiler, and architecture environment. |
| `build_windows_rocm.cmd` | Builds vLLM's C++/HIP extensions in place. |
| `install_windows_rocm.cmd` | Installs this source tree into `.venv211` without replacing ROCm Torch. |
| `serve_windows_rocm.cmd` | Runs `vllm serve` with Windows/AMD settings and safe configurable defaults. |
| `bench_windows_rocm.cmd` | Runs `vllm bench` in the same environment used for serving. |
| `vram_guard.ps1` | Fail-closed dedicated-VRAM and optional stall watchdog. |
| `quantize_fp8.py` | Optional CPU-side FP8 conversion helper; use a separate quantization environment. |
| `WINDOWS_ROCM_PERFORMANCE.md` | Reproducible measurements, profiles, and blockers. |
| `update_windows_rocm.ps1` | Checks for the latest stable upstream tag and safely rebases the Windows patch stack. |
| `UPSTREAM_VERSION` | Machine-readable upstream stable tag used by update automation. |

## Updating from upstream

This is a focused patch stack on top of vLLM, not an independent inference
engine. Upstream changes can alter Python APIs, kernels, dependencies, and build
requirements. Check for and apply the latest stable release with:

```powershell
.\update_windows_rocm.ps1 -CheckOnly
.\update_windows_rocm.ps1
```

The updater ignores prereleases, fetches the selected tag directly from the
official vLLM repository, creates a timestamped backup branch, and rebases the
current patch stack. It never pushes or merges. If upstream and Windows changes
overlap, it stops at the conflict for review.

The scheduled `Sync Windows ROCm with upstream vLLM` GitHub workflow performs
the same check daily. A clean rebase becomes a validation PR; a conflicted
rebase becomes an issue. Neither path auto-merges. Review every Windows patch,
rebuild the native extensions, and rerun guarded model/quality benchmarks before
publishing an update.

Contributors must also follow [AGENTS.md](AGENTS.md) and upstream's
[contribution guide](CONTRIBUTING.md). AI-assisted changes require human review,
appropriate tests, and disclosure under the repository's contribution policy.

## License and attribution

This repository is based on
[vllm-project/vllm](https://github.com/vllm-project/vllm) and retains its
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for the upstream base and
modification attribution.

ROCm, PyTorch, Triton, Hugging Face models, and other downloaded dependencies
are separate projects with their own licenses and notices. They are not
relicensed by this repository. Review a model's license before downloading,
redistributing, or serving it.

If you use vLLM in research, cite the original project:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

Upstream documentation: <https://docs.vllm.ai/>
