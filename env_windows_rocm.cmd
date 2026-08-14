@echo off
REM =====================================================================
REM Shared build environment for native Windows + ROCm (gfx1200/gfx1201).
REM Dot-sourced by build_windows_rocm.cmd and install_windows_rocm.cmd.
REM Deliberately has no setlocal: the caller owns the scope.
REM
REM Requires:
REM   - Visual Studio 2022 Build Tools (for the Windows SDK + libs)
REM   - .venv211 with torch+rocm, rocm-sdk-devel, cmake>=4.4, ninja
REM
REM Notes learned the hard way:
REM   - CMake >= 4.4 is REQUIRED. 3.31 cannot configure HIP with clang-cl.
REM   - clang-cl must be used for BOTH CXX and HIP; CMake refuses to mix
REM     a GNU-style driver for HIP with MSVC cl for C++.
REM   - ROCM_PATH/HIP_PATH must use forward slashes.
REM =====================================================================

if "%VLLM_ROOT%"=="" set VLLM_ROOT=%~dp0
set VENV=%VLLM_ROOT%.venv211

REM --- MSVC environment (vcvars64 invokes vswhere by bare name) ---------
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo [FAIL] vcvars64 & exit /b 1)

REM --- ROCm SDK location, from the wheel itself ------------------------
for /f "delims=" %%i in ('"%VENV%\Scripts\python.exe" -m rocm_sdk path --root') do set ROCM_ROOT=%%i
set ROCM_ROOT_FWD=%ROCM_ROOT:\=/%

REM CMake interpolates these into generated cmake code unescaped, so
REM backslashes surface as "Invalid character escape". Forward slashes only.
set ROCM_PATH=%ROCM_ROOT_FWD%
set HIP_PATH=%ROCM_ROOT_FWD%
set PATH=%ROCM_ROOT%\bin;%ROCM_ROOT%\lib\llvm\bin;%VENV%\Scripts;%PATH%

REM --- vLLM build configuration ----------------------------------------
set VLLM_TARGET_DEVICE=rocm
if "%VLLM_ROCM_GPU_ARCH%"=="" set VLLM_ROCM_GPU_ARCH=gfx1201
set PYTORCH_ROCM_ARCH=%VLLM_ROCM_GPU_ARCH%
if "%MAX_JOBS%"=="" set MAX_JOBS=16

REM setup.py appends $CMAKE_ARGS verbatim, and later -D wins for cache vars,
REM so this also overrides the -DROCM_PATH that setup.py derives from torch.
set CMAKE_ARGS=-DCMAKE_CXX_COMPILER=%ROCM_ROOT_FWD%/lib/llvm/bin/clang-cl.exe ^
 -DCMAKE_HIP_COMPILER=%ROCM_ROOT_FWD%/lib/llvm/bin/clang-cl.exe ^
 -DCMAKE_HIP_PLATFORM=amd ^
 -DCMAKE_HIP_ARCHITECTURES=%VLLM_ROCM_GPU_ARCH% ^
 -DCMAKE_HIP_FLAGS=--rocm-device-lib-path=%ROCM_ROOT_FWD%/lib/llvm/amdgcn/bitcode ^
 -DROCM_PATH=%ROCM_ROOT_FWD%

echo === ROCM_ROOT   : %ROCM_ROOT%
echo === TARGET      : %VLLM_TARGET_DEVICE% / %PYTORCH_ROCM_ARCH%
echo === MAX_JOBS    : %MAX_JOBS%
echo.
exit /b 0
