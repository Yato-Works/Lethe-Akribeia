# AM Apex phase 3: when the phase-2 driver has finished, produce the final report artefacts.
#
#  1. official LoCoMo metric cross-check, run with the PINNED upstream scorer
#     (third_party/benchmarks/locomo/task_eval/evaluation.py, metric=f1);
#  2. the consolidated AM Apex scorecard (accuracy / retrieval recall / abstention /
#     temporal / knowledge-update / multi-session / token efficiency / latency).
#
# Usage:
#   pwsh -NoProfile -File scripts/benchmarks/run_apex_suite_phase3.ps1 -WaitForPid <driverPid>

[CmdletBinding()]
param(
    [int]$WaitForPid = 0,
    [int]$PersonaPid = 0
)

$ErrorActionPreference = "Continue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$logDir = Join-Path $root "benchmark_results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "phase3.log"

function Write-Phase3 {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

Write-Phase3 "waiting for driver pid $WaitForPid"
while ($WaitForPid -gt 0 -and (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 15
}
Write-Phase3 "driver finished"

# 1. Pinned official LoCoMo scorer (F1 metric) over the fresh LoCoMo-10 run.
$beamPy = Join-Path $root ".venv-benchmarks\beam\Scripts\python.exe"
& $beamPy (Join-Path $PSScriptRoot "score_locomo_official.py") `
    --run benchmark_results/locomo10 `
    --out benchmark_results/official_locomo_score.json `
    --tag locomo10_coder7b_apex *> (Join-Path $logDir "official_locomo_score.log")
Write-Phase3 "official LoCoMo scorer exit=$LASTEXITCODE"

# 2. Consolidated scorecard (also refreshes AM_APEX_SCORECARD.md).
& uv run python (Join-Path $PSScriptRoot "apex_scorecard.py") *> (Join-Path $logDir "scorecard.log")
Write-Phase3 "scorecard exit=$LASTEXITCODE"

# 3. Wait for the queued PersonaMem subset run, then refresh the scorecard once more.
if ($PersonaPid -gt 0) {
    Write-Phase3 "waiting for personamem queue pid $PersonaPid"
    while (Get-Process -Id $PersonaPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 15 }
    & uv run python (Join-Path $PSScriptRoot "apex_scorecard.py") *> (Join-Path $logDir "scorecard_personamem.log")
    Write-Phase3 "scorecard refresh exit=$LASTEXITCODE"
}

Write-Phase3 "phase 3 done"
