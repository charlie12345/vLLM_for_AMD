@echo off
REM Run `vllm bench ...` on native Windows + ROCm. Same environment as serving,
REM so the numbers reflect what the server actually does.
REM Usage: bench_windows_rocm.cmd throughput --model <model> ...
setlocal

if "%VLLM_ROOT%"=="" set VLLM_ROOT=%~dp0
set VENV=%VLLM_ROOT%.venv211

set TORCH_BLAS_PREFER_HIPBLASLT=1
set VLLM_WORKER_MULTIPROC_METHOD=spawn
set VLLM_ATTENTION_BACKEND=TRITON_ATTN
if "%VLLM_CACHE_ROOT%"=="" set VLLM_CACHE_ROOT=C:\AI\vc
if "%TRITON_CACHE_DIR%"=="" set TRITON_CACHE_DIR=C:\AI\vt

REM Safety: vLLM claims gpu-memory-utilization of the card and sizes the KV
REM cache to fill whatever the weights leave over, so even a small model will
REM allocate ~27 GiB by default. This card also drives the desktop, and RDNA4
REM falls off a large-GEMM cliff under ~3 GiB free. Default to a conservative
REM fraction unless the caller states one explicitly.
echo %* | findstr /C:"gpu-memory-utilization" >nul
if errorlevel 1 (
  set VLLM_MEM_ARG=--gpu-memory-utilization 0.55
) else (
  set VLLM_MEM_ARG=
)

"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main bench %* %VLLM_MEM_ARG%
exit /b %errorlevel%
