<#
.SYNOPSIS
    Rebase the native-Windows ROCm patch stack onto the latest stable vLLM tag.

.DESCRIPTION
    Finds stable vLLM tags directly from the official upstream repository,
    fetches the selected tag, detects the stable tag at the base of the current
    patch stack, creates a timestamped backup branch, and rebases the current
    branch onto the new release.

    The script never selects release candidates or development tags
    automatically and never pushes or merges. A rebase conflict is left in
    place so it can be reviewed and resolved, or aborted with `git rebase
    --abort`.

.PARAMETER CheckOnly
    Report the current base, latest stable tag, and whether an update is
    available without rebasing.

.PARAMETER TargetTag
    Use a specific stable vLLM tag instead of the latest one.

.PARAMETER UpstreamUrl
    Official upstream Git URL used to list and fetch tags.

.EXAMPLE
    .\update_windows_rocm.ps1 -CheckOnly

.EXAMPLE
    .\update_windows_rocm.ps1

.EXAMPLE
    .\update_windows_rocm.ps1 -TargetTag v0.27.1
#>
#Requires -Version 5.1

[CmdletBinding()]
param(
    [switch]$CheckOnly,

    [ValidatePattern('^v\d+\.\d+\.\d+$')]
    [string]$TargetTag,

    [ValidateNotNullOrEmpty()]
    [string]$UpstreamUrl = 'https://github.com/vllm-project/vllm.git'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path.TrimEnd('\')
$Git = (Get-Command git -CommandType Application -ErrorAction Stop |
        Select-Object -First 1).Source

function Invoke-Git {
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $output = @(& $Git -C $Root @Arguments)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $exitCode."
    }
    return $output
}

function ConvertTo-StableVersion([string]$Tag) {
    if ($Tag -notmatch '^v(?<Version>\d+\.\d+\.\d+)$') {
        return $null
    }
    return [version]$Matches.Version
}

function Get-StableTagNamesFromUpstream {
    $lines = @(Invoke-Git @(
        'ls-remote', '--tags', '--refs', $UpstreamUrl, 'v*'
    ))
    $tags = foreach ($line in $lines) {
        if ($line -match 'refs/tags/(?<Tag>v\d+\.\d+\.\d+)$') {
            $Matches.Tag
        }
    }
    return @($tags | Sort-Object -Unique)
}

function Get-NewestStableTag([string[]]$Tags) {
    return @(
        $Tags |
            Where-Object { $null -ne (ConvertTo-StableVersion $_) } |
            Sort-Object { ConvertTo-StableVersion $_ } -Descending
    ) | Select-Object -First 1
}

function Write-UpdateState(
    [string]$CurrentBase,
    [string]$LatestStable,
    [bool]$UpdateAvailable
) {
    Write-Output "CURRENT_BASE=$CurrentBase"
    Write-Output "LATEST_STABLE=$LatestStable"
    Write-Output "UPDATE_AVAILABLE=$($UpdateAvailable.ToString().ToLowerInvariant())"
}

function Update-BaseMetadata([string]$Tag) {
    $commit = (Invoke-Git @('rev-parse', $Tag))[0]
    $shortCommit = $commit.Substring(0, 7)
    $utf8 = [Text.UTF8Encoding]::new($false)

    $versionFile = Join-Path $Root 'UPSTREAM_VERSION'
    [IO.File]::WriteAllText($versionFile, "$Tag`n", $utf8)

    $readmePath = Join-Path $Root 'README.md'
    $readme = [IO.File]::ReadAllText($readmePath)
    $readme = $readme -replace (
        'currently based on vLLM v\d+\.\d+\.\d+\.'
    ), "currently based on vLLM $Tag."
    $readme = $readme -replace (
        '\| vLLM base \| v\d+\.\d+\.\d+, base commit `[^`]+` \|'
    ), "| vLLM base | $Tag, base commit ``$shortCommit`` |"
    [IO.File]::WriteAllText($readmePath, $readme, $utf8)

    $noticePath = Join-Path $Root 'NOTICE'
    $notice = [IO.File]::ReadAllText($noticePath)
    $notice = $notice -replace (
        'Upstream base: vLLM v\d+\.\d+\.\d+, commit [0-9a-f]+'
    ), "Upstream base: vLLM $Tag, commit $shortCommit"
    [IO.File]::WriteAllText($noticePath, $notice, $utf8)

    [void](Invoke-Git @('add', 'UPSTREAM_VERSION', 'README.md', 'NOTICE'))
    & $Git -C $Root diff --cached --quiet
    $diffExitCode = $LASTEXITCODE
    if ($diffExitCode -eq 1) {
        [void](Invoke-Git @('commit', '-m', "Record upstream base $Tag"))
    } elseif ($diffExitCode -ne 0) {
        throw "Could not inspect the base metadata changes (exit $diffExitCode)."
    }
}

[void](Invoke-Git @('rev-parse', '--show-toplevel'))

$upstreamTags = @(Get-StableTagNamesFromUpstream)
if ($upstreamTags.Count -eq 0) {
    throw "No stable vLLM tags were found at $UpstreamUrl."
}

$latestStable = Get-NewestStableTag $upstreamTags
$selectedTag = if ($TargetTag) { $TargetTag } else { $latestStable }
if ($selectedTag -notin $upstreamTags) {
    throw "$selectedTag is not a stable tag in the official upstream repository."
}

Write-Host "Fetching upstream $selectedTag..." -ForegroundColor Cyan
[void](Invoke-Git @(
    'fetch', '--force', $UpstreamUrl,
    "refs/tags/${selectedTag}:refs/tags/${selectedTag}"
))

$mergedTags = @(Invoke-Git @('tag', '--merged', 'HEAD', '--list', 'v*'))
$currentBase = Get-NewestStableTag $mergedTags
if (-not $currentBase) {
    throw 'Could not identify a stable upstream tag in the current history.'
}

$currentVersion = ConvertTo-StableVersion $currentBase
$selectedVersion = ConvertTo-StableVersion $selectedTag
$updateAvailable = $selectedVersion -gt $currentVersion
Write-UpdateState $currentBase $selectedTag $updateAvailable

if ($CheckOnly -or -not $updateAvailable) {
    if (-not $updateAvailable) {
        Write-Host "Already based on $currentBase; no stable update is needed." -ForegroundColor Green
    }
    exit 0
}

$branch = (Invoke-Git @('symbolic-ref', '--quiet', '--short', 'HEAD'))[0]
if (-not $branch) {
    throw 'Check out a local branch before applying an update.'
}

$trackedChanges = @(Invoke-Git @(
    'status', '--porcelain', '--untracked-files=no'
))
if ($trackedChanges.Count -ne 0) {
    throw 'Tracked files are modified. Commit or stash them before updating.'
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backupBranch = "backup/$branch-$currentBase-$stamp"
[void](Invoke-Git @('branch', $backupBranch, 'HEAD'))
Write-Host "Backup branch: $backupBranch" -ForegroundColor Yellow
Write-Host "Rebasing $branch from $currentBase onto $selectedTag..." -ForegroundColor Cyan

& $Git -C $Root rebase --onto $selectedTag $currentBase $branch
$rebaseExitCode = $LASTEXITCODE
if ($rebaseExitCode -ne 0) {
    Write-Host ''
    Write-Host 'The rebase stopped for review.' -ForegroundColor Yellow
    Write-Host 'Resolve each conflict, run git add for the resolved files, then:'
    Write-Host '  git rebase --continue'
    Write-Host 'Or restore the pre-update state with:'
    Write-Host '  git rebase --abort'
    Write-Host "The original commit is also retained at $backupBranch."
    throw "Rebase onto $selectedTag stopped with exit code $rebaseExitCode."
}

Update-BaseMetadata $selectedTag
Write-Host "Rebased $branch onto $selectedTag." -ForegroundColor Green
Write-Host 'Run setup_windows_rocm.ps1 and the guarded model benchmarks before merging or publishing.'
