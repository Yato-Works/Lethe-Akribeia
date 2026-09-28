# AM Apex phase 2: reorder the remaining stages and keep the $0 pipeline running.
#
# Stage 01 (LoCoMo 1,540) is the only stage without an incremental checkpoint, so it
# must run alone.  Once it finishes this script:
#   1. stops the phase-1 driver (tree) so the redundant LoCoMo-1986 stage does not
#      occupy the device for 80 minutes before the cheaper, high-coverage suites;
#   2. archives the stale (2026-09-20) LoCoMo-10 conversation artefacts so the fresh
#      run cannot be mixed with them;
#   3. relaunches the driver in coverage order: LongMemEval 500 -> BEAM 100K/500K/1M/10M
#      -> LoCoMo-10 1,986 -> (PersonaMem 32K is queued behind it, also $0/local).
#
# Usage:
#   pwsh -NoProfile -File scripts/benchmarks/run_apex_suite_phase2.ps1 -Phase1Pid 30464 -PersonaQueuePid 25292

[CmdletBinding()]
param(
    [int]$Phase1Pid = 0,
    [int]$PersonaQueuePid = 0,
    [int]$PersonaQuestions = 50
)

$ErrorActionPreference = "Continue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$logDir = Join-Path $root "benchmark_results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "phase2.log"

function Write-Phase2 {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

Write-Phase2 "waiting for phase-1 driver pid $Phase1Pid (stage 01 LoCoMo 1,540)"
while ($Phase1Pid -gt 0 -and (Get-Process -Id $Phase1Pid -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 15
}
Write-Phase2 "phase-1 driver exit detected"

# 1. Make sure nothing from phase 1 (or the queued PersonaMem run) keeps the device.
& taskkill /PID $Phase1Pid /T /F 2>&1 | Out-Null
if ($PersonaQueuePid -gt 0) { & taskkill /PID $PersonaQueuePid /T /F 2>&1 | Out-Null }
Start-Sleep -Seconds 5
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*artificial_memory*" } |
    ForEach-Object { Write-Phase2 ("stopping leftover python pid " + $_.Id); Stop-Process -Id $_.Id -Force }
Start-Sleep -Seconds 3

# 2. Archive stale LoCoMo-10 artefacts (2026-09-20 baseline) so fresh results are unambiguous.
$archive = Join-Path $root "benchmark_results\locomo10\_archive_20260920"
New-Item -ItemType Directory -Force -Path $archive | Out-Null
Get-ChildItem (Join-Path $root "benchmark_results\locomo10") -File |
    Where-Object { $_.Name -match "^(conv_\d+_results|full_locomo10_report)\.json$" } |
    ForEach-Object {
        Write-Phase2 ("archiving stale artefact " + $_.Name)
        Move-Item -Path $_.FullName -Destination (Join-Path $archive $_.Name) -Force
    }

# 3. Relaunch in coverage order (LongMemEval + BEAM first, LoCoMo-1986 last).
$driver = Join-Path $PSScriptRoot "run_apex_official_suite.ps1"
$only = "03_longmemeval,04_beam100K,05_beam500K,06_beam1M,07_beam10M,02_locomo1986"
$newDriver = Start-Process -FilePath "pwsh" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $driver, "-Only", $only -PassThru -WindowStyle Hidden
Write-Phase2 "phase-2 driver started pid $($newDriver.Id) (stage order: $only)"

# 4. Re-arm the PersonaMem 32K subset run behind the new driver.
$queue = Start-Process -FilePath "pwsh" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "run_after_suite_personamem.ps1"),
    "-WaitForPid", "$($newDriver.Id)", "-NumQuestions", "$PersonaQuestions" -PassThru -WindowStyle Hidden
Write-Phase2 "personamem queue re-armed pid $($queue.Id) (waits for $($newDriver.Id))"

# 5. Chain the finisher so the final scorer + scorecard are produced automatically.
$phase3 = Start-Process -FilePath "pwsh" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "run_apex_suite_phase3.ps1"),
    "-WaitForPid", "$($newDriver.Id)", "-PersonaPid", "$($queue.Id)" -PassThru -WindowStyle Hidden
Write-Phase2 "phase-3 finisher started pid $($phase3.Id)"
Write-Phase2 "phase-2 setup done"
