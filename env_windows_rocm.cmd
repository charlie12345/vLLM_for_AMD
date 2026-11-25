@echo off
REM =====================================================================
REM Shared build environment for native Windows + ROCm (gfx1201 by default).
REM Dot-sourced by build_windows_rocm.cmd and install_windows_rocm.cmd.
REM Deliberately has no setlocal: the caller owns the scope.
REM
REM Requires:
REM   - Visual Studio 2022 Build Tools (for the Windows SDK + libs)
REM   - VLLM_VENV (or .venv-rocm10) with torch+rocm, rocm-sdk-devel,
REM     cmake>=4.4, and ninja
REM
REM Notes learned the hard way:
REM   - CMake >= 4.4 is REQUIRED. 3.31 cannot configure HIP with clang-cl.
REM   - clang-cl must be used for BOTH CXX and HIP; CMake refuses to mix
REM     a GNU-style driver for HIP with MSVC cl for C++.
REM   - ROCM_PATH/HIP_PATH must use forward slashes.
REM =====================================================================

if "%VLLM_ROOT%"=="" set "VLLM_ROOT=%~dp0"
set "VENV=%VLLM_VENV%"
if not defined VENV set "VENV=%VLLM_ROOT%.venv-rocm10"

REM --- MSVC environment (vcvars64 invokes vswhere by bare name) ---------
REM A stray quote in a machine/user PATH entry makes vcvars64's batch parser
REM fail with "was unexpected at this time". Sanitize only this child shell;
REM this does not modify the persistent Windows PATH.
set "PATH=%PATH:"=%"
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
rem vcvars64.bat exists in every Visual Studio 2022 edition, and the installer's
rem default root depends on the edition (Build Tools lands in Program Files (x86),
rem Community/Professional/Enterprise in Program Files). Ask vswhere for the
rem installation that actually has the C++ toolset instead of guessing a root;
rem the Installer directory was just added to PATH, so vswhere is on it.
rem An explicit VCVARS64 (Developer Prompt, CI) still wins.
if not defined VCVARS64 for /f "delims=" %%i in ('vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul') do if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=%%i\VC\Auxiliary\Build\vcvars64.bat"
rem Fallbacks for the case where vswhere is missing or reports nothing.
if not defined VCVARS64 if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS64 if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS64 if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS64 (
  echo [FAIL] vcvars64.bat not found for any Visual Studio 2022 edition.
  echo        Install the "Desktop development with C++" workload, or set VCVARS64
  echo        to the full path of your vcvars64.bat.
  exit /b 1
)
call "%VCVARS64%" >nul
if errorlevel 1 (echo [FAIL] vcvars64 & exit /b 1)

REM --- ROCm SDK location, from the wheel itself ------------------------
for /f "delims=" %%i in ('"%VENV%\Scripts\python.exe" -m rocm_sdk path --root') do set "ROCM_ROOT=%%i"
set "ROCM_ROOT_FWD=%ROCM_ROOT:\=/%"

REM CMake interpolates these into generated cmake code unescaped, so
REM backslashes surface as "Invalid character escape". Forward slashes only.
set "ROCM_PATH=%ROCM_ROOT_FWD%"
set "HIP_PATH=%ROCM_ROOT_FWD%"
set "PATH=%ROCM_ROOT%\bin;%ROCM_ROOT%\lib\llvm\bin;%VENV%\Scripts;%PATH%"

REM --- vLLM build configuration ----------------------------------------
set "VLLM_TARGET_DEVICE=rocm"
if not defined VLLM_ROCM_ARCH set "VLLM_ROCM_ARCH=gfx1201"
set "PYTORCH_ROCM_ARCH=%VLLM_ROCM_ARCH%"
if "%MAX_JOBS%"=="" set "MAX_JOBS=8"

REM setup.py appends $CMAKE_ARGS verbatim, and later -D wins for cache vars,
REM so this also overrides the -DROCM_PATH that setup.py derives from torch.
set CMAKE_ARGS=-DCMAKE_CXX_COMPILER=%ROCM_ROOT_FWD%/lib/llvm/bin/clang-cl.exe ^
 -DCMAKE_HIP_COMPILER=%ROCM_ROOT_FWD%/lib/llvm/bin/clang-cl.exe ^
 -DCMAKE_HIP_PLATFORM=amd ^
 -DCMAKE_HIP_ARCHITECTURES=%VLLM_ROCM_ARCH% ^
 -DCMAKE_HIP_FLAGS=--rocm-device-lib-path=%ROCM_ROOT_FWD%/lib/llvm/amdgcn/bitcode ^
 -DROCM_PATH=%ROCM_ROOT_FWD%

echo === ROCM_ROOT   : %ROCM_ROOT%
echo === TARGET      : %VLLM_TARGET_DEVICE% / %PYTORCH_ROCM_ARCH%
echo === MAX_JOBS    : %MAX_JOBS%
echo.
exit /b 0
