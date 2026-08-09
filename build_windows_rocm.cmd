@echo off
REM Build the vLLM C/HIP extensions in-place. See env_windows_rocm.cmd.
setlocal

if "%VLLM_ROOT%"=="" set VLLM_ROOT=%~dp0
call "%VLLM_ROOT%env_windows_rocm.cmd"
if errorlevel 1 exit /b 1

cd /d "%VLLM_ROOT%"
"%VENV%\Scripts\python.exe" setup.py build_ext --inplace %*
exit /b %errorlevel%
