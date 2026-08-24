@echo off
REM Run `vllm serve` on native Windows + ROCm (gfx1201 / RDNA4).
REM Usage: serve_windows_rocm.cmd <model> [extra vllm serve args...]
setlocal

set "ROCM_VLLM_ROOT=%~dp0"
if not "%VLLM_ROOT%"=="" set "ROCM_VLLM_ROOT=%VLLM_ROOT%"
set "VLLM_ROOT="
set "VENV=%ROCM_VLLM_ROOT%.venv211"

REM rocBLAS has no tuned gfx1201 kernels; without this bf16/fp16 GEMMs run ~10x slow.
set TORCH_BLAS_PREFER_HIPBLASLT=1

REM Windows has no fork.
set VLLM_WORKER_MULTIPROC_METHOD=spawn

REM VLLM_ATTENTION_BACKEND was removed upstream. The ROCm platform chooses
REM ROCM_ATTN by default; use --attention-backend for an explicit override.
set "VLLM_ATTENTION_BACKEND="

REM Short cache roots: LongPathsEnabled is off, and the default Inductor cache
REM path overflows MAX_PATH (260) by a couple of characters.
if "%VLLM_CACHE_ROOT%"=="" set VLLM_CACHE_ROOT=C:\AI\vc
if "%TRITON_CACHE_DIR%"=="" set TRITON_CACHE_DIR=C:\AI\vt

REM Safety: vLLM claims gpu-memory-utilization of the card and sizes the KV
REM cache to fill whatever the weights leave over, so even a small model will
REM allocate ~27 GiB by default. This card also drives the desktop, and RDNA4
REM falls off a large-GEMM cliff under ~3 GiB free. Default to a conservative
REM fraction unless the caller states one explicitly. The wrapper settings
REM below are optional; explicit CLI arguments always win.
if "%WINDOWS_ROCM_GPU_MEMORY_UTILIZATION%"=="" set WINDOWS_ROCM_GPU_MEMORY_UTILIZATION=0.55
echo %* | findstr /C:"gpu-memory-utilization" >nul
if errorlevel 1 (
  set "ROCM_MEM_ARG=--gpu-memory-utilization %WINDOWS_ROCM_GPU_MEMORY_UTILIZATION%"
) else (
  set "ROCM_MEM_ARG="
)

REM v0.27's upstream 8192-token compile warmup can transiently consume nearly
REM all dedicated VRAM even with a conservative KV-cache fraction.
if "%WINDOWS_ROCM_MAX_NUM_BATCHED_TOKENS%"=="" set WINDOWS_ROCM_MAX_NUM_BATCHED_TOKENS=2048
echo %* | findstr /C:"max-num-batched-tokens" >nul
if errorlevel 1 (
  set "ROCM_BATCH_ARG=--max-num-batched-tokens %WINDOWS_ROCM_MAX_NUM_BATCHED_TOKENS%"
) else (
  set "ROCM_BATCH_ARG="
)

set "ROCM_KV_ARG="
if not "%WINDOWS_ROCM_KV_CACHE_DTYPE%"=="" (
  echo %* | findstr /C:"kv-cache-dtype" >nul
  if errorlevel 1 set "ROCM_KV_ARG=--kv-cache-dtype %WINDOWS_ROCM_KV_CACHE_DTYPE%"
)

set "ROCM_GRAPH_ARG="
if not "%WINDOWS_ROCM_CUDAGRAPH_MODE%"=="" (
  echo %* | findstr /C:"cudagraph_mode" >nul
  if errorlevel 1 set "ROCM_GRAPH_ARG=-cc.cudagraph_mode=%WINDOWS_ROCM_CUDAGRAPH_MODE%"
)

"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main serve %* %ROCM_MEM_ARG% %ROCM_BATCH_ARG% %ROCM_KV_ARG% %ROCM_GRAPH_ARG%
exit /b %errorlevel%
