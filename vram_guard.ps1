<#
.SYNOPSIS
    Run a vLLM command under a watchdog: kills it on VRAM exhaustion or a stall.

.DESCRIPTION
    Two failure modes on this box take the whole machine down, and this guards
    against both.

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
    # Total dedicated VRAM on the adapter -- the desktop's share counts too.
    try {
        $s = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
        if (-not $s) { return $null }
        return (($s | Measure-Object -Property CookedValue -Maximum).Maximum / 1GB)
    }
    catch { return $null }
}

function Stop-Tree([int]$TreePid, [string]$Why) {
    Write-Guard "KILLING pid ${TreePid}: $Why"
    & taskkill.exe /PID $TreePid /T /F 2>&1 | Out-Null
    Start-Sleep -Seconds 2
    # The engine core is a spawned grandchild and outlives a tree kill once the
    # intermediate shell is gone, so sweep any surviving interpreters.
    foreach ($p in @(Get-Process python -ErrorAction SilentlyContinue)) {
        & taskkill.exe /PID $p.Id /T /F 2>&1 | Out-Null
    }
}

Write-Guard "=== vram_guard start ==="
$baseline = Get-VramGiB
if ($null -eq $baseline) {
    Write-Guard 'WARNING: GPU memory counters unavailable - VRAM guard INACTIVE.'
}
else {
    Write-Guard ('baseline {0:N2} GiB | limit {1:N2} | warn {2:N2}' -f $baseline, $LimitGiB, $WarnGiB)
}
if ($StallLogPath) { Write-Guard "stall guard: $StallLogPath idle > ${StallSec}s" }
Write-Guard "command: $Command"

$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $Command -PassThru -WindowStyle Hidden
Write-Guard "child pid $($proc.Id)"

$breaches = 0
$peak = 0.0
$killed = ''
$startedAt = Get-Date

while (-not $proc.HasExited) {
    Start-Sleep -Seconds $IntervalSec

    $used = Get-VramGiB
    if ($null -ne $used) {
        if ($used -gt $peak) { $peak = $used }
        if ($used -ge $LimitGiB) {
            $breaches++
            Write-Guard ('VRAM {0:N2} GiB >= {1:N2} ({2}/{3})' -f $used, $LimitGiB, $breaches, $Consecutive)
            if ($breaches -ge $Consecutive) {
                $killed = 'VRAM {0:N2} GiB exceeded limit {1:N2} GiB' -f $used, $LimitGiB
                Stop-Tree -TreePid $proc.Id -Why $killed
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
            Stop-Tree -TreePid $proc.Id -Why $killed
            break
        }
    }
    elseif ($StallLogPath) {
        # Log not created yet; treat time since launch as the idle period.
        $idle = ((Get-Date) - $startedAt).TotalSeconds
        if ($idle -ge $StallSec) {
            $killed = 'stalled: log never created after {0:N0}s' -f $idle
            Stop-Tree -TreePid $proc.Id -Why $killed
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
