# Queue a PersonaMem 32K run so it starts only after the main official suite
# driver (which owns the single Ollama device) has finished.
#
# Usage:
#   pwsh -NoProfile -File scripts/benchmarks/run_after_suite_personamem.ps1 -WaitForPid 30464 -NumQuestions 50

[CmdletBinding()]
param(
    [int]$WaitForPid = 0,
    [int]$NumQuestions = 50,
    [string]$Scale = "32K"
)

$ErrorActionPreference = "Continue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$logDir = Join-Path $root "benchmark_results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "personamem_queue.log"

function Write-QueueLog {
    param([string]$Message)
    Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message) -Encoding utf8
}

Write-QueueLog "queued: waiting for pid $WaitForPid to release the Ollama device"
while ($WaitForPid -gt 0 -and (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 20
}
Write-QueueLog "suite finished; starting PersonaMem $Scale ($NumQuestions questions)"

$lab = Join-Path $root "benchmark_results\official\personamem_lab"
New-Item -ItemType Directory -Force -Path $lab | Out-Null
$subset = Join-Path $lab "questions_$(($Scale).ToLower())_first$NumQuestions.csv"
$result = Join-Path $lab "personamem_$(($Scale).ToLower())_first$NumQuestions.csv"

$q = Join-Path $root "datasets\official\personamem\questions_$Scale.csv"
& uv run python -c "import csv,sys; rows=list(csv.reader(open(r'$q',encoding='utf-8',newline=''))); csv.writer(open(r'$subset','w',encoding='utf-8',newline='')).writerows(rows[:$($NumQuestions + 1)]); print('subset rows', len(rows[:$($NumQuestions + 1)]) - 1)"

& pwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_personamem_local.ps1") `
    -Scale custom -QuestionsCsv $subset -ResultCsv $result *> (Join-Path $logDir "personamem_run.log")

Write-QueueLog "PersonaMem run finished (exit=$LASTEXITCODE); scoring $result"
& uv run python -c "import csv,sys; rows=list(csv.DictReader(open(r'$result',encoding='utf-8'))); n=len(rows); c=sum(1 for r in rows if str(r.get('score','')).strip().lower() in ('true','1','1.0')); print(f'PersonaMem $Scale subset: {c}/{n} = {100*c/max(1,n):.1f}%')" |
    Tee-Object -FilePath (Join-Path $logDir "personamem_score.txt")
Write-QueueLog "done"
