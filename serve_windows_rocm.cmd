@echo off
REM Run `vllm serve` on native Windows + ROCm (gfx1201 / RDNA4).
REM Usage: serve_windows_rocm.cmd <model> [extra vllm serve args...]
setlocal

if "%VLLM_ROOT%"=="" set VLLM_ROOT=%~dp0
set VENV=%VLLM_ROOT%.venv211

REM rocBLAS has no tuned gfx1201 kernels; without this bf16/fp16 GEMMs run ~10x slow.
set TORCH_BLAS_PREFER_HIPBLASLT=1

REM Windows has no fork.
set VLLM_WORKER_MULTIPROC_METHOD=spawn

REM gfx1201 compiles none of vLLM's ROCm attention kernels (they are ISA-gated
REM to gfx9/gfx11/gfx1100), so Triton is the only working backend.
set VLLM_ATTENTION_BACKEND=TRITON_ATTN

REM Short cache roots: LongPathsEnabled is off, and the default Inductor cache
REM path overflows MAX_PATH (260) by a couple of characters.
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

"%VENV%\Scripts\python.exe" -m vllm.entrypoints.cli.main serve %* %VLLM_MEM_ARG%
exit /b %errorlevel%
