<#
.SYNOPSIS
    Step 2 finalize: pair the sensitivity arms that have landed, then render the scorecard.

.DESCRIPTION
    The arms are slow (minutes per question at 7B); this script is what turns them
    into §7.  For each pairing whose *both* runs exist it calls
    `model_sensitivity.py` with the frozen claims file and the cohort that keeps
    the pooled row honest, then re-renders `AM_APEX_SCORECARD.md`.  Safe to re-run:
    every pairing overwrites only its own artefact, and a pairing whose arm has not
    been rebuilt yet is skipped with a line in the log rather than run against a
    stale reference.

    With -Wait it blocks until the arms log's last line is `=== ALL ARMS DONE ===`
    (after this script started), so one detached invocation covers wait + pair + render.

.EXAMPLE
    pwsh -NoProfile -File scripts/benchmarks/finalize_model_sensitivity.ps1 -Wait
#>
param(
    [switch]$Wait,
    [int]$WaitSeconds = 5400,
    [switch]$SkipScorecard
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repo
$log = Join-Path $repo "benchmark_results/logs/model_sensitivity_finalize.log"
$armsLog = Join-Path $repo "benchmark_results/logs/model_sensitivity_arms.log"

function Say([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line
}

if ($Wait) {
    $started = Get-Date
    $deadline = $started.AddSeconds($WaitSeconds)
    Say "waiting for the arms (up to ${WaitSeconds}s), pairing each one as it lands..."
} else {
    $started = Get-Date
    $deadline = Get-Date
}

$claims = "benchmark_results/committer_metrics"
$pairings = @(
    @{
        name = "lme-postfix"
        a = "benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json"
        b = "benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json"
        claims = "$claims/lme_all_types_claims.json"
        cohort = "longmemeval_all500"
        label = "LongMemEval 500 - deployed configuration"
        out = "benchmark_results/model_sensitivity/longmemeval_all500_system.json"
    },
    @{
        name = "locomo-system-postfix"
        a = "benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json"
        b = "benchmark_results/locomo1540/temporal321_rules_commit_15b.json"
        claims = "$claims/locomo_cat2_temporal_claims.json"
        cohort = "locomo_cat2_temporal"
        label = "LoCoMo 1,540 - category 2 (temporal) - deployed system"
        out = "benchmark_results/model_sensitivity/locomo_cat2_temporal_system.json"
    }
)
$paired = @{}

function Invoke-Pairing($p) {
    Say "PAIR $($p.name) :: $($p.a) vs $($p.b)"
    & uv run python scripts/benchmarks/model_sensitivity.py `
        --a $p.a --b $p.b --claims $p.claims --cohort $p.cohort `
        --label $p.label --out $p.out 2>&1 |
        Tee-Object -FilePath (Join-Path (Split-Path $script:log) "finalize_$($p.name).log")
    if ($LASTEXITCODE -ne 0) {
        Say "PAIR FAILED $($p.name) (exit $LASTEXITCODE)"
        throw "pairing '$($p.name)' failed"
    }
}

do {
    foreach ($p in $pairings) {
        if (-not $paired[$p.name] -and (Test-Path $p.a) -and (Test-Path $p.b)) {
            Invoke-Pairing $p
            $paired[$p.name] = $true
        }
    }
    $last = Get-Content $armsLog -Tail 1 -ErrorAction SilentlyContinue
    $armsDone = ($last -match "ALL ARMS DONE") -and
                ($last -match "^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]") -and
                ([datetime]::ParseExact($Matches[1], "yyyy-MM-dd HH:mm:ss", $null)) -ge $started
    if (-not $armsDone) { Start-Sleep -Seconds 20 }
} while (-not $armsDone -and (Get-Date) -lt $deadline)

foreach ($p in $pairings) {
    if (-not $paired[$p.name]) {
        if ((Test-Path $p.a) -and (Test-Path $p.b)) {
            Invoke-Pairing $p
            $paired[$p.name] = $true
        } else {
            Say "SKIP $($p.name): $((@($p.a, $p.b) | Where-Object { -not (Test-Path $_) }) -join ', ') not on disk"
        }
    }
}

if (-not $SkipScorecard) {
    Say "rendering the scorecard"
    & uv run python scripts/benchmarks/apex_scorecard.py
    if ($LASTEXITCODE -ne 0) { throw "scorecard render failed (exit $LASTEXITCODE)" }
}

Say "=== FINALIZE DONE ==="