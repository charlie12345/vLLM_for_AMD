<#
.SYNOPSIS
    Prepare, build, and verify the tested native-Windows ROCm vLLM stack.

.DESCRIPTION
    Automates the repository-local parts of the Windows ROCm installation:

    - validates the required Windows, AMD, uv, Git, and Visual Studio setup;
    - creates or reuses an architecture-isolated Python 3.12 environment;
    - installs the pinned AMD ROCm/PyTorch, Triton, and build dependencies;
    - probes the AMD GPU through vram_guard.ps1 and requires the selected arch;
    - builds and editable-installs this vLLM fork; and
    - performs a second guarded import/device verification.

    It deliberately does not install or update the AMD display driver, Visual
    Studio, Git, uv, or model weights. System-wide installers require explicit
    user review, and models have separate licenses and storage requirements.

.PARAMETER MaxJobs
    Maximum parallel native build jobs. Defaults to 8 to preserve desktop
    responsiveness during large HIP translation units.

.PARAMETER GpuArch
    ROCm GCN architecture to install and compile. gfx1201 is the validated
    default. Other accepted RDNA targets are build-supported but unvalidated.

.PARAMETER AllowUnhealthyAdapter
    Continue when Windows reports an AMD display adapter in a non-OK state.
    This is unsafe for model execution and exists only for expert diagnostics.

.PARAMETER GuardLimitGiB
    Dedicated-VRAM kill threshold used for the guarded GPU probes.

.PARAMETER GuardWarnGiB
    Dedicated-VRAM warning threshold used for the guarded GPU probes.

.PARAMETER PlanOnly
    Validate system prerequisites and print the planned work without changing
    the environment, downloading packages, probing the GPU, or building vLLM.

.EXAMPLE
    .\setup_windows_rocm.ps1 -PlanOnly

.EXAMPLE
    .\setup_windows_rocm.ps1
#>
#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet(
        'gfx1030',
        'gfx1100', 'gfx1101', 'gfx1102', 'gfx1103',
        'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153',
        'gfx1200', 'gfx1201'
    )]
    [string]$GpuArch = 'gfx1201',

    [ValidateRange(1, 256)]
    [int]$MaxJobs = 8,

    [ValidateRange(1.0, 256.0)]
    [double]$GuardLimitGiB = 26.0,

    [ValidateRange(0.0, 256.0)]
    [double]$GuardWarnGiB = 23.0,

    [switch]$PlanOnly,

    [switch]$AllowUnhealthyAdapter
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ExpectedPython = '3.12'
$ExpectedGpuArch = $GpuArch
$TorchVersion = '2.13.0+rocm10.0.0'
$TorchvisionVersion = '0.28.0+rocm10.0.0'
$RocmVersion = '10.0.0'
$TritonVersion = '3.7.1.post27'
$AmdPytorchWheelIndex = 'https://stable.repo.amd.com/rocm/pytorch/whl-next/'
$AmdRocmWheelIndex = 'https://stable.repo.amd.com/rocm/core/whl-next/'

$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path.TrimEnd('\')
$Venv = if ([string]::IsNullOrWhiteSpace($env:VLLM_VENV)) {
    $venvName = if ($ExpectedGpuArch -eq 'gfx1201') {
        '.venv-rocm10'
    }
    else {
        ".venv-rocm10-$ExpectedGpuArch"
    }
    Join-Path $Root $venvName
}
else {
    [IO.Path]::GetFullPath($env:VLLM_VENV)
}
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
$Requirements = Join-Path $Root 'requirements\common.txt'
$BuildScript = Join-Path $Root 'build_windows_rocm.cmd'
$InstallScript = Join-Path $Root 'install_windows_rocm.cmd'
$GuardScript = Join-Path $Root 'vram_guard.ps1'
$RuntimeVerifier = Join-Path $Root 'tools\verify_windows_rocm_runtime.py'
$Vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'

function Write-Step([string]$Message) {
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Write-Detail([string]$Name, [string]$Value) {
    Write-Host ('{0,-18} {1}' -f ($Name + ':'), $Value)
}

function Invoke-Native(
    [string]$Description,
    [string]$FilePath,
    [string[]]$ArgumentList
) {
    Write-Step $Description
    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
}

function Get-RequiredApplication([string]$Name, [string]$InstallHint) {
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $command) {
        throw "$Name was not found on PATH. $InstallHint"
    }
    return $command.Source
}

function Invoke-GuardedProbe([string]$Name, [string]$PythonCode) {
    $logs = Join-Path $Root 'logs'
    [void](New-Item -ItemType Directory -Path $logs -Force)

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $safeName = $Name.ToLowerInvariant().Replace(' ', '-')
    $probeLog = Join-Path $logs "setup-$safeName-$stamp.log"
    $guardLog = Join-Path $logs "setup-$safeName-$stamp.guard.log"
    $probeFile = Join-Path ([IO.Path]::GetTempPath()) (
        'vllm-windows-rocm-probe-{0}.py' -f [guid]::NewGuid().ToString('N')
    )

    [IO.File]::WriteAllText(
        $probeFile,
        $PythonCode,
        [Text.UTF8Encoding]::new($false)
    )

    try {
        $command = 'call "{0}" "{1}" > "{2}" 2>&1' -f (
            $VenvPython,
            $probeFile,
            $probeLog
        )

        Write-Step "$Name (guarded)"
        & $GuardScript `
            -Command $command `
            -LimitGiB $GuardLimitGiB `
            -WarnGiB $GuardWarnGiB `
            -StallLogPath $probeLog `
            -StallSec 180 `
            -LogPath $guardLog
        $exitCode = $LASTEXITCODE

        if (Test-Path -LiteralPath $probeLog) {
            Get-Content -LiteralPath $probeLog
        }
        if ($exitCode -ne 0) {
            throw "$Name failed under the VRAM guard with exit code $exitCode. See $guardLog"
        }
    }
    finally {
        Remove-Item -LiteralPath $probeFile -Force -ErrorAction SilentlyContinue
    }
}

Write-Step 'Preflight checks'

if ($env:OS -ne 'Windows_NT') {
    throw 'This setup script supports native 64-bit Windows only.'
}
if (-not [Environment]::Is64BitOperatingSystem -or
    -not [Environment]::Is64BitProcess) {
    throw 'Run this script from a 64-bit PowerShell process on 64-bit Windows.'
}
if ($GuardWarnGiB -ge $GuardLimitGiB) {
    throw 'GuardWarnGiB must be lower than GuardLimitGiB.'
}

$requiredFiles = @(
    $Requirements,
    $BuildScript,
    $InstallScript,
    $GuardScript,
    $RuntimeVerifier,
    (Join-Path $Root 'env_windows_rocm.cmd'),
    (Join-Path $Root 'pyproject.toml')
)
foreach ($path in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "The clone is incomplete; required file is missing: $path"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $Root '.git'))) {
    throw @"
Git metadata is missing. Clone the repository with Git instead of downloading
a source ZIP so versioning and editable installation have the expected history.
"@
}

$uv = Get-RequiredApplication 'uv' `
    'Install it from https://docs.astral.sh/uv/getting-started/installation/'
$git = Get-RequiredApplication 'git' `
    'Install Git for Windows, reopen PowerShell, and try again.'

if (-not (Test-Path -LiteralPath $Vcvars -PathType Leaf)) {
    throw @"
Visual Studio 2022 Build Tools was not found at the path expected by
env_windows_rocm.cmd:
  $Vcvars
Install the Desktop development with C++ workload, MSVC v143, and a Windows
10 or 11 SDK. Reopen PowerShell after installation.
"@
}

$amdAdapters = @()
try {
    $amdAdapters = @(Get-PnpDevice -Class Display -PresentOnly -ErrorAction Stop |
        Where-Object { $_.FriendlyName -match 'AMD|Radeon' } |
        ForEach-Object {
            [pscustomobject]@{
                Name = $_.FriendlyName
                Status = [string]$_.Status
                Problem = [string]$_.Problem
            }
        })
}
catch {
    Write-Warning "PnP display query failed; trying bounded CIM: $($_.Exception.Message)"
    try {
        $amdAdapters = @(Get-CimInstance Win32_VideoController `
            -OperationTimeoutSec 5 -ErrorAction Stop |
            Where-Object { $_.Name -match 'AMD|Radeon' })
    }
    catch {
        Write-Warning "Could not query display adapters: $($_.Exception.Message)"
    }
}
if ($amdAdapters.Count -eq 0) {
    throw 'No AMD/Radeon display adapter was detected. Install the matching AMD driver first.'
}
$unhealthyAdapters = @($amdAdapters | Where-Object {
    $_.Status -and $_.Status -ne 'OK'
})
if ($unhealthyAdapters.Count -gt 0) {
    $details = ($unhealthyAdapters | ForEach-Object {
        '{0}: status={1}, problem={2}' -f $_.Name, $_.Status, $_.Problem
    }) -join '; '
    $message = @"
Windows reports an unhealthy AMD display adapter: $details
Restart Windows (cold-power-cycle if necessary) and verify every intended GPU
shows Device Manager status OK before building or running vLLM.
"@
    if ($AllowUnhealthyAdapter) {
        Write-Warning "$message Continuing only because -AllowUnhealthyAdapter was specified."
    }
    else {
        throw "$message Expert diagnostics can bypass this check with -AllowUnhealthyAdapter."
    }
}

if ($Root.Length -gt 80) {
    Write-Warning "The clone path is long ($($Root.Length) characters). C:\AI\vllm is recommended."
}

try {
    $system = Get-CimInstance Win32_ComputerSystem `
        -OperationTimeoutSec 5 -ErrorAction Stop
    $ramGiB = [math]::Round($system.TotalPhysicalMemory / 1GB, 1)
    if ($ramGiB -lt 32) {
        Write-Warning "Only $ramGiB GiB of system RAM was detected; 32 GiB is the practical minimum."
    }
}
catch {
    $ramGiB = 'unknown'
    Write-Warning "Could not query system RAM: $($_.Exception.Message)"
}

try {
    $rootItem = Get-Item -LiteralPath $Root
    $drive = Get-PSDrive -Name $rootItem.PSDrive.Name
    $freeGiB = [math]::Round($drive.Free / 1GB, 1)
    if ($freeGiB -lt 25) {
        Write-Warning "Only $freeGiB GiB is free on $($drive.Name):; builds and wheels need headroom."
    }
}
catch {
    $freeGiB = 'unknown'
    Write-Warning "Could not query free disk space: $($_.Exception.Message)"
}

Write-Detail 'Repository' $Root
Write-Detail 'Expected GPU' $ExpectedGpuArch
Write-Detail 'AMD adapter' (($amdAdapters | ForEach-Object {
    if ($_.Status) { "$($_.Name) [$($_.Status)]" } else { $_.Name }
}) -join '; ')
Write-Detail 'System RAM GiB' ([string]$ramGiB)
Write-Detail 'Free disk GiB' ([string]$freeGiB)
Write-Detail 'uv' $uv
Write-Detail 'Git' $git
Write-Detail 'Virtual env' $Venv
Write-Detail 'Build jobs' ([string]$MaxJobs)
Write-Detail 'VRAM guard' "$GuardWarnGiB GiB warn / $GuardLimitGiB GiB kill"

if ($PlanOnly) {
    Write-Step 'Plan'
    Write-Host "1. Create or reuse $Venv with Python 3.12."
    Write-Host '2. Install the pinned ROCm 10.0, PyTorch 2.13, and Triton stack.'
    Write-Host "3. Run a fail-closed guarded Torch/HIP/$ExpectedGpuArch probe."
    Write-Host '4. Build the native C++/HIP extensions with clang-cl.'
    Write-Host '5. Editable-install vLLM and run a guarded import/device probe.'
    Write-Host "`nPlanOnly completed; no files, packages, GPU contexts, or builds were changed."
    return
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    if (Test-Path -LiteralPath $Venv) {
        throw @"
$Venv exists but Scripts\python.exe is missing. Refusing to overwrite an
ambiguous environment. Rename or remove only that directory after reviewing
its contents, then rerun this script.
"@
    }
    Invoke-Native `
        'Create Python 3.12 virtual environment' `
        $uv `
        @('venv', '--python', $ExpectedPython, $Venv)
}
else {
    Write-Step 'Reuse existing virtual environment'
    Write-Host $Venv
}

$pythonVersion = & $VenvPython -c `
    'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))'
if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne $ExpectedPython) {
    throw "$Venv must contain Python $ExpectedPython; found '$pythonVersion'."
}

$uvPip = @('pip', 'install', '--python', $VenvPython)

Invoke-Native `
    'Install pinned AMD ROCm and PyTorch wheels' `
    $uv `
    ($uvPip + @(
        '--extra-index-url', $AmdPytorchWheelIndex,
        '--extra-index-url', $AmdRocmWheelIndex,
        '--index-strategy', 'unsafe-best-match',
        "torch[device-$ExpectedGpuArch]==$TorchVersion",
        "torchvision[device-$ExpectedGpuArch]==$TorchvisionVersion",
        "rocm[devel,device-$ExpectedGpuArch]==$RocmVersion"
    ))

Invoke-Native `
    'Install pinned Triton runtime' `
    $uv `
    ($uvPip + @(
        "triton-windows==$TritonVersion",
        'winloop'
    ))

Invoke-Native `
    'Install core vLLM Python requirements' `
    $uv `
    ($uvPip + @('-r', $Requirements))

Invoke-Native `
    'Install native build requirements' `
    $uv `
    ($uvPip + @(
        'cmake==4.4.2',
        'ninja',
        'packaging>=24.2',
        'setuptools>=77,<80',
        'setuptools-scm>=8',
        'setuptools-rust>=1.9',
        'wheel',
        'jinja2>=3.1.6'
    ))

$torchProbe = @"
import json
import torch

if not torch.accelerator.is_available():
    raise SystemExit('Torch cannot see an AMD GPU through HIP.')
if torch.version.hip is None:
    raise SystemExit('This is not a ROCm/HIP PyTorch build.')

device_module = torch.get_device_module(torch.accelerator.current_accelerator())
properties = device_module.get_device_properties(0)
architecture = str(properties.gcnArchName)
architecture_base = architecture.split(':', 1)[0]
result = {
    'torch': torch.__version__,
    'hip': torch.version.hip,
    'device': device_module.get_device_name(0),
    'architecture': architecture,
}
print(json.dumps(result, sort_keys=True))

if architecture_base != '$ExpectedGpuArch':
    raise SystemExit(
        f'Expected $ExpectedGpuArch, but Torch reported {architecture}. '
        'Use -GpuArch only for the architecture physically installed.'
    )
"@
Invoke-GuardedProbe -Name 'ROCm device verification' -PythonCode $torchProbe

$previousMaxJobs = [Environment]::GetEnvironmentVariable('MAX_JOBS', 'Process')
$previousVllmRoot = [Environment]::GetEnvironmentVariable('VLLM_ROOT', 'Process')
$previousVllmRocmArch = [Environment]::GetEnvironmentVariable(
    'VLLM_ROCM_ARCH',
    'Process'
)
$previousVllmVenv = [Environment]::GetEnvironmentVariable('VLLM_VENV', 'Process')
try {
    $env:MAX_JOBS = [string]$MaxJobs
    $env:VLLM_ROOT = $Root + '\'
    $env:VLLM_ROCM_ARCH = $ExpectedGpuArch
    $env:VLLM_VENV = $Venv

    Invoke-Native `
        'Build vLLM native Windows ROCm extensions' `
        $BuildScript `
        @()

    Invoke-Native `
        'Editable-install this vLLM fork' `
        $InstallScript `
        @()
}
finally {
    [Environment]::SetEnvironmentVariable(
        'MAX_JOBS',
        $previousMaxJobs,
        'Process'
    )
    [Environment]::SetEnvironmentVariable(
        'VLLM_ROOT',
        $previousVllmRoot,
        'Process'
    )
    [Environment]::SetEnvironmentVariable(
        'VLLM_ROCM_ARCH',
        $previousVllmRocmArch,
        'Process'
    )
    [Environment]::SetEnvironmentVariable(
        'VLLM_VENV',
        $previousVllmVenv,
        'Process'
    )
}

$installProbe = @"
import runpy
import sys

sys.argv = [
    r'$RuntimeVerifier',
    '--expected-arch',
    '$ExpectedGpuArch',
]
runpy.run_path(r'$RuntimeVerifier', run_name='__main__')
"@
Invoke-GuardedProbe -Name 'Installed vLLM verification' -PythonCode $installProbe

Write-Step 'Setup complete'
Write-Host 'The tested native-Windows ROCm vLLM environment is installed.'
Write-Host 'No model was downloaded or launched.'
Write-Host 'Next: follow README.md -> Choosing and downloading a model, then use'
Write-Host 'vram_guard.ps1 around every serve or benchmark command.'
