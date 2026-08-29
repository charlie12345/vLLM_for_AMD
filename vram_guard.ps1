<#
.SYNOPSIS
    Run a vLLM command under a watchdog: kills it on VRAM exhaustion or a stall.

.DESCRIPTION
    Two failure modes on this box can take the whole machine down. This script
    reduces the user-mode risk from both but cannot recover a kernel driver or
    firmware hang.

    1. VRAM exhaustion. vLLM claims `gpu-memory-utilization` of the card and
       sizes its KV cache to fill whatever the weights leave over, so even a
       small model will allocate ~27 GiB on a 32 GiB card. This GPU also drives
       the desktop, and RDNA4 falls off a large-GEMM cliff under ~3 GiB free.

    2. Stalls. HIP graph replay can wedge (observed with FP8 Qwen3-8B on
       gfx1201): the process pins a core, holds all its VRAM, and never
       returns. Nothing times this out on its own.

    The watchdog samples the adapter's dedicated VRAM and, if given
    -StallLogPath, the mtime of the run's own log. Either breach kills the
    whole process tree.

.PARAMETER Command
    Command line to run, executed via cmd.exe /c. Redirect the command's own
    output inside this string; the watchdog does not capture it.

.PARAMETER LimitGiB
    Kill threshold for total dedicated VRAM on the adapter.

.PARAMETER WarnGiB
    Log a warning at this level without killing.

.PARAMETER StallLogPath
    If set, kill the run when this file has not been written for -StallSec.
    Point it at the log your -Command redirects into.

.PARAMETER StallSec
    Seconds of no log growth before declaring a stall. Must comfortably exceed
    the longest legitimately quiet phase; torch.compile plus graph capture can
    run ~120 s silently on an 8B.

.EXAMPLE
    $log = "C:\AI\vllm\run.log"
    .\vram_guard.ps1 -StallLogPath $log -Command "C:\AI\vllm\bench_windows_rocm.cmd throughput --model C:\AI\models\Qwen3-8B-FP8-Dynamic > `"$log`" 2>&1"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Command,
    [double]$LimitGiB = 26.0,
    [double]$WarnGiB = 23.0,
    [int]$IntervalSec = 3,
    [int]$Consecutive = 3,
    [int]$CounterTimeoutSec = 5,
    [int]$CounterFailureLimit = 3,
    [string]$StallLogPath = '',
    [int]$StallSec = 300,
    [string]$LogPath = 'C:\AI\vllm\vram_guard.log'
)

$ErrorActionPreference = 'Continue'

function Write-Guard([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message
    Write-Host $line
    try { Add-Content -Path $LogPath -Value $line } catch { }
}

function Get-VramGiB {
    # Query in a disposable helper process. A GPU reset can wedge Get-Counter
    # itself; keeping it in the watchdog process would prevent stall cleanup.
    $counterScript = @'
$samples = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
if (-not $samples) { exit 2 }
$bytes = ($samples | Measure-Object -Property CookedValue -Maximum).Maximum
[Console]::Out.Write([string]::Format(
    [Globalization.CultureInfo]::InvariantCulture, '{0:R}', [double]$bytes))
'@
    $encoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($counterScript)
    )
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = 'powershell.exe'
    $startInfo.Arguments = "-NoProfile -EncodedCommand $encoded"
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $helper = New-Object System.Diagnostics.Process
    $helper.StartInfo = $startInfo

    try {
        if (-not $helper.Start()) { return $null }
        if (-not $helper.WaitForExit($CounterTimeoutSec * 1000)) {
            try { $helper.Kill() } catch { }
            return $null
        }
        $raw = $helper.StandardOutput.ReadToEnd().Trim()
        if ($helper.ExitCode -ne 0 -or -not $raw) { return $null }

        $bytes = 0.0
        $parsed = [double]::TryParse(
            $raw,
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture,
            [ref]$bytes
        )
        if (-not $parsed) { return $null }
        return $bytes / 1GB
    }
    catch { return $null }
    finally { $helper.Dispose() }
}

function Update-TrackedProcessTree(
    [System.Collections.Generic.HashSet[int]]$TrackedPids
) {
    try {
        $processes = @(Get-CimInstance Win32_Process `
            -OperationTimeoutSec $CounterTimeoutSec -ErrorAction Stop |
            Select-Object ProcessId, ParentProcessId)
        do {
            $added = $false
            foreach ($candidate in $processes) {
                $candidatePid = [int]$candidate.ProcessId
                $parentPid = [int]$candidate.ParentProcessId
                if (
                    $TrackedPids.Contains($parentPid) -and
                    $TrackedPids.Add($candidatePid)
                ) {
                    $added = $true
                }
            }
        } while ($added)
    }
    catch {
        Write-Guard "WARNING: could not refresh child process list: $($_.Exception.Message)"
    }
}

function Stop-Tree(
    [int]$TreePid,
    [string]$Why,
    [System.Collections.Generic.HashSet[int]]$TrackedPids
) {
    Write-Guard "KILLING pid ${TreePid}: $Why"

    # Preserve every descendant PID before terminating the root. EngineCore is
    # spawned through intermediate processes, and Windows keeps its original
    # ParentProcessId even if an intermediate parent has already exited.
    Update-TrackedProcessTree -TrackedPids $TrackedPids
    & taskkill.exe /PID $TreePid /T /F 2>&1 | Out-Null

    # A spawned grandchild can outlive taskkill /T after its intermediate
    # parent exits. Kill only PIDs observed in this run, never unrelated Python
    # processes on the machine. A second pass catches children racing shutdown.
    foreach ($pass in 1..2) {
        Start-Sleep -Seconds 1
        Update-TrackedProcessTree -TrackedPids $TrackedPids
        foreach ($trackedPid in (@($TrackedPids) | Sort-Object -Descending)) {
            if (Get-Process -Id $trackedPid -ErrorAction SilentlyContinue) {
                & taskkill.exe /PID $trackedPid /T /F 2>&1 | Out-Null
            }
        }
    }
}

Write-Guard "=== vram_guard start ==="
$baseline = $null
foreach ($attempt in 1..$CounterFailureLimit) {
    $baseline = Get-VramGiB
    if ($null -ne $baseline) { break }
    Write-Guard "WARNING: VRAM counter attempt ${attempt}/${CounterFailureLimit} failed."
}
if ($null -eq $baseline) {
    Write-Guard 'ERROR: GPU memory counters unavailable; refusing to launch unguarded.'
    exit 98
}
else {
    Write-Guard ('baseline {0:N2} GiB | limit {1:N2} | warn {2:N2}' -f $baseline, $LimitGiB, $WarnGiB)
}
if ($StallLogPath) { Write-Guard "stall guard: $StallLogPath idle > ${StallSec}s" }
Write-Guard "command: $Command"

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = if ($env:ComSpec) { $env:ComSpec } else { 'cmd.exe' }
# cmd.exe requires an extra outer quote when /C receives a command whose
# executable path is itself quoted. Start-Process flattens ArgumentList and can
# silently lose that boundary, causing a quoted Python path to exit before its
# redirection is created. Assign the exact command line through ProcessStartInfo
# so paths with spaces and shell redirection remain intact on PowerShell 5.1+.
$startInfo.Arguments = '/D /S /C "' + $Command + '"'
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $startInfo
if (-not $proc.Start()) {
    Write-Guard 'ERROR: failed to start guarded child process.'
    exit 97
}
Write-Guard "child pid $($proc.Id)"
$trackedPids = [System.Collections.Generic.HashSet[int]]::new()
[void]$trackedPids.Add($proc.Id)

$breaches = 0
$counterFailures = 0
$peak = 0.0
$killed = ''
$startedAt = Get-Date

while (-not $proc.HasExited) {
    Start-Sleep -Seconds $IntervalSec
    Update-TrackedProcessTree -TrackedPids $trackedPids

    $used = Get-VramGiB
    if ($null -eq $used) {
        $counterFailures++
        Write-Guard "VRAM counter unavailable (${counterFailures}/${CounterFailureLimit})"
        if ($counterFailures -ge $CounterFailureLimit) {
            $killed = "VRAM counter unavailable for $counterFailures consecutive samples"
            Stop-Tree -TreePid $proc.Id -Why $killed -TrackedPids $trackedPids
            break
        }
    }
    else {
        $counterFailures = 0
        if ($used -gt $peak) { $peak = $used }
        if ($used -ge $LimitGiB) {
            $breaches++
            Write-Guard ('VRAM {0:N2} GiB >= {1:N2} ({2}/{3})' -f $used, $LimitGiB, $breaches, $Consecutive)
            if ($breaches -ge $Consecutive) {
                $killed = 'VRAM {0:N2} GiB exceeded limit {1:N2} GiB' -f $used, $LimitGiB
                Stop-Tree -TreePid $proc.Id -Why $killed -TrackedPids $trackedPids
                break
            }
        }
        else {
            if ($breaches -gt 0) { Write-Guard ('VRAM back to {0:N2} GiB' -f $used) }
            $breaches = 0
            if ($used -ge $WarnGiB) { Write-Guard ('warn: VRAM {0:N2} GiB' -f $used) }
        }
    }

    if ($StallLogPath -and (Test-Path $StallLogPath)) {
        $idle = ((Get-Date) - (Get-Item $StallLogPath).LastWriteTime).TotalSeconds
        if ($idle -ge $StallSec) {
            $killed = 'stalled: no log output for {0:N0}s' -f $idle
            Stop-Tree -TreePid $proc.Id -Why $killed -TrackedPids $trackedPids
            break
        }
    }
    elseif ($StallLogPath) {
        # Log not created yet; treat time since launch as the idle period.
        $idle = ((Get-Date) - $startedAt).TotalSeconds
        if ($idle -ge $StallSec) {
            $killed = 'stalled: log never created after {0:N0}s' -f $idle
            Stop-Tree -TreePid $proc.Id -Why $killed -TrackedPids $trackedPids
            break
        }
    }
}

if ($killed) {
    Write-Guard ('killed by watchdog ({0}); peak VRAM {1:N2} GiB' -f $killed, $peak)
    exit 99
}

$proc.WaitForExit()
Write-Guard ('exited {0}; peak VRAM {1:N2} GiB' -f $proc.ExitCode, $peak)
exit $proc.ExitCode
