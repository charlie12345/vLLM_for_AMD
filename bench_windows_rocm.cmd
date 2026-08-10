@echo off
REM Run `vllm bench ...` on native Windows + ROCm. Same environment as serving,
REM so the numbers reflect what the server actually does.
REM Usage: bench_windows_rocm.cmd throughput --model <model> ...
setlocal

set "ROCM_VLLM_ROOT=%~dp0"
if not "%VLLM_ROOT%"=="" set "ROCM_VLLM_ROOT=%VLLM_ROOT%"
set "VLLM_ROOT="
set "VENV=%ROCM_VLLM_ROOT%.venv211"

set TORCH_BLAS_PREFER_HIPBLASLT=1
set VLLM_WORKER_MULTIPROC_METHOD=spawn
REM VLLM_ATTENTION_BACKEND was removed upstream. The ROCm platform chooses
REM ROCM_ATTN by default; use --attention-backend for an explicit override.
set "VLLM_ATTENTION_BACKEND="
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

"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main bench %* %ROCM_MEM_ARG% %ROCM_KV_ARG% %ROCM_GRAPH_ARG%
exit /b %errorlevel%
