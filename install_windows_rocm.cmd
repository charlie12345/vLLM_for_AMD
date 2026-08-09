@echo off
REM Editable-install vLLM into .venv211 so importlib.metadata can see it.
REM --no-deps: requirements/common.txt is installed separately; the ROCm and
REM CUDA requirement files pull Linux-only packages.
REM --no-build-isolation: the build needs this venv's torch and cmake>=4.4.
setlocal

if "%VLLM_ROOT%"=="" set VLLM_ROOT=%~dp0
call "%VLLM_ROOT%env_windows_rocm.cmd"
if errorlevel 1 exit /b 1

cd /d "%VLLM_ROOT%"
uv pip install --python "%VENV%\Scripts\python.exe" -e . --no-deps --no-build-isolation %*
exit /b %errorlevel%
