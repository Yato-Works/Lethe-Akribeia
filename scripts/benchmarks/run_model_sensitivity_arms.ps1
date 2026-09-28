<#
.SYNOPSIS
    Step 2 - model-sensitivity arms: replay the frozen caches with a smaller reader.

.DESCRIPTION
    Answers one question: how much of Lethe's accuracy is the reader model's
    ability, and how much is the runtime around it?  Every arm below re-uses an
    already-frozen context cache, an already-frozen prompt, the same question
    order and the same matcher as its 7B counterpart, so the ONLY difference
    between a pair is `--model`.

    Arms produced (never overwrites a 7B artefact):

      LoCoMo temporal 321, reader only   temporal321_rules_15b.json
        pairs with temporal321_rules.json            (7B, 47.98%)
      LoCoMo temporal 321, + committer   temporal321_rules_commit_15b.json
        pairs with temporal321_rules_commit.json     (7B, 49.22%)
      LongMemEval 500                    grand_longmemeval_report_15b_apex_ctx8192.json
        pairs with grand_longmemeval_report_coder7b_apex_ctx8192.json (7B, 75.80%)
      -Recheck7bLmeArm                   grand_longmemeval_report_7b_apex_ctx8192_postfix.json
      -Recheck7bSystemArm                temporal321_rules_commit_7b_postfix.json

    Pairing discipline: the two LoCoMo arms differ by `--commit-temporal` on
    purpose.  The reader-only pair isolates reader sensitivity; the commit pair
    measures the deployed system, where the deterministic zone is identical in
    both arms and can be subtracted out.  A pairing whose controls fire (retrieval
    drift, a changed committer decision) is reported as such by
    `model_sensitivity.py` - and when the cause is a stored 7B arm built by older
    code, the 7B recheck switches above rebuild it rather than papering over it.

.EXAMPLE
    pwsh -NoProfile -File scripts/benchmarks/run_model_sensitivity_arms.ps1
    pwsh -NoProfile -File scripts/benchmarks/run_model_sensitivity_arms.ps1 -SkipLme
#>
param(
    [string]$Model = "qwen2.5:1.5b",
    [string]$ArmSuffix = "15b",
    [string]$Cache = "benchmark_results/locomo_context_cache_rules.jsonl",
    [switch]$SkipLme,
    # The stored 7B system arm (temporal321_rules_commit.json) was produced by the
    # committer *before* its tie-breaks were made deterministic, so it claims a
    # slightly different question set (111 vs 109).  Re-running it with the
    # current code keeps the system pair free of that confound - it costs about an
    # hour at 7B, which is why it is opt-in.
    [switch]$Recheck7bSystemArm,
    # The stored 7B LongMemEval arm (grand_longmemeval_report_coder7b_apex_ctx8192)
    # was produced before the MSC compiler was tuned: recompiling its questions
    # with today's code returns different token costs on 340/500 questions and a
    # different oracle on 5, so it cannot be paired against an arm built today.
    # Re-running it (~45 min at 7B) makes the LongMemEval pair differ by model only.
    [switch]$Recheck7bLmeArm,
    # Skip the three 1.5B arms: only the 7B rechecks are left to run.
    [switch]$Skip15bArms,
    [string]$SevenBModel = "qwen2.5:7b-instruct"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repo
$logDir = Join-Path $repo "benchmark_results/logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "model_sensitivity_arms.log"

function Say([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line
}

function Invoke-Arm([string]$name, [string[]]$argv) {
    $armLog = Join-Path $logDir "$name.log"
    Say "ARM START $name :: uv run python $($argv -join ' ')"
    $t0 = Get-Date
    # The arm's own progress lines go to their own log as well as the console:
    # they are the only place a mid-run failure or a stall is visible.
    & uv run python @argv 2>&1 | Tee-Object -FilePath $armLog
    $code = $LASTEXITCODE
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if ($code -ne 0) {
        Say "ARM FAILED $name (exit $code) after ${secs}s - see $armLog"
        throw "arm '$name' failed with exit code $code"
    }
    Say "ARM DONE $name in ${secs}s"
}

Say "=== model sensitivity arms :: model=$Model suffix=$ArmSuffix ==="

# 1. Reader-only arm: the reader answers everything, exactly like the 7B
#    `temporal321_rules.json` artefact it pairs against.
if (-not $Skip15bArms) {
    Invoke-Arm "locomo-temporal-reader-only" @(
        "scripts/run_locomo_1540.py",
        "--model", $Model,
        "--num-ctx", "8192",
        "--cache-file", $Cache,
        "--tag", "temporal321_rules_$ArmSuffix"
    )

    # 2. System arm: same cache, but the deterministic committer may answer
    #    temporal questions without an LLM call, exactly like the 7B
    #    `temporal321_rules_commit.json` artefact it pairs against.
    Invoke-Arm "locomo-temporal-with-committer" @(
        "scripts/run_locomo_1540.py",
        "--model", $Model,
        "--num-ctx", "8192",
        "--cache-file", $Cache,
        "--commit-temporal",
        "--tag", "temporal321_rules_commit_$ArmSuffix"
    )

    # 3. LongMemEval, all 500 questions.  This runner has no committer, so the arm
    #    is a pure reader comparison; `--resume` lets a killed run continue.
    if (-not $SkipLme) {
        Invoke-Arm "longmemeval-all-500" @(
            "scripts/run_longmemeval_full_suite.py",
            "--num-questions", "0",
            "--resume",
            "--model", $Model,
            "--num-ctx", "8192",
            "--tag", "${ArmSuffix}_apex_ctx8192"
        )
    }
}

# 4. Optional: rebuild the 7B LongMemEval arm with the current MSC compiler, so
#    the flagship pair differs by the reader and not by two pipeline builds.
if ($Recheck7bLmeArm) {
    Invoke-Arm "longmemeval-7b-postfix" @(
        "scripts/run_longmemeval_full_suite.py",
        "--num-questions", "0",
        "--resume",
        "--model", $SevenBModel,
        "--num-ctx", "8192",
        "--tag", "7b_apex_ctx8192_postfix"
    )
}

# 5. Optional: rebuild the 7B system arm with the current committer so the system
#    pair differs by the reader and not by two different committer builds.
if ($Recheck7bSystemArm) {
    Invoke-Arm "locomo-temporal-commit-7b-postfix" @(
        "scripts/run_locomo_1540.py",
        "--model", $SevenBModel,
        "--num-ctx", "8192",
        "--cache-file", $Cache,
        "--commit-temporal",
        "--tag", "temporal321_rules_commit_7b_postfix"
    )
}

Say "=== ALL ARMS DONE ==="
