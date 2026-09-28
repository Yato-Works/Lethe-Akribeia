# AM Apex official benchmark suite driver (local $0: Ollama qwen2.5-coder:7b)
#
# This driver does NOT contain benchmark logic. It only launches the existing,
# already-tuned official runner scripts in priority order, one stage at a time
# (Ollama is a single shared device), and tees every stage into its own log:
#
#   benchmark_results/logs/01_locomo1540.log ... 07_beam10M.log
#
# Usage:
#   pwsh -NoProfile -File scripts/benchmarks/run_apex_official_suite.ps1
#   pwsh -NoProfile -File scripts/benchmarks/run_apex_official_suite.ps1 -Only 03_longmemeval
#
# Every stage is resumable: LoCoMo writes per-conversation checkpoints, LongMemEval
# writes a checkpoint, BEAM writes one JSON per scale.

[CmdletBinding()]
param(
    [string[]]$Only = @()
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root

$logDir = Join-Path $root "benchmark_results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$driverLog = Join-Path $logDir "driver.log"

function Write-Driver {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $driverLog -Value $line -Encoding utf8
}

function Invoke-Stage {
    param(
        [string]$Name,
        [string[]]$Argv
    )
    $log = Join-Path $logDir "$Name.log"
    Write-Driver ("START {0}: uv {1}" -f $Name, ($Argv -join " "))
    $stageStart = Get-Date
    & uv @Argv *> $log
    $exit = $LASTEXITCODE
    $secs = [math]::Round(((Get-Date) - $stageStart).TotalSeconds, 1)
    Write-Driver ("END   {0}: exit={1} elapsed={2}s" -f $Name, $exit, $secs)
}

$stages = [ordered]@{
    "01_locomo1540"  = @("run", "python", "scripts/run_locomo_1540.py")
    "02_locomo1986"  = @("run", "python", "scripts/run_locomo_full_suite.py", "--start-conv", "0", "--end-conv", "9")
    "03_longmemeval" = @("run", "python", "scripts/run_longmemeval_full_suite.py", "--tag", "coder7b_apex", "--num-ctx", "8192")
    "04_beam100K"    = @("run", "python", "scripts/run_coder7b_beam.py", "--scale", "100K", "--limit-questions", "20")
    "05_beam500K"    = @("run", "python", "scripts/run_coder7b_beam.py", "--scale", "500K", "--limit-questions", "20")
    "06_beam1M"      = @("run", "python", "scripts/run_coder7b_beam.py", "--scale", "1M", "--limit-questions", "5")
    "07_beam10M"     = @("run", "python", "scripts/run_coder7b_beam.py", "--scale", "10M", "--limit-questions", "3")
}

Write-Driver "=== AM Apex official suite driver start (model: qwen2.5-coder:7b) ==="

foreach ($name in $stages.Keys) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $name)) {
        Write-Driver "SKIP  $name (filtered)"
        continue
    }
    Invoke-Stage -Name $name -Argv $stages[$name]
}

Write-Driver "=== AM Apex official suite driver done ==="
