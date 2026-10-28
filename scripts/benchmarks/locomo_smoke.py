"""One-command LoCoMo smoke check.

The repo carries a large `scripts/benchmarks/` surface, so "which command do I
run after touching the pipeline?" was not answerable without reading the code.
This is that command.  It takes a small PINNED set of real LoCoMo questions
(`benchmark_config/smoke_manifest.json`), runs them end to end through the same
frozen pipeline the full benchmark uses (retrieval -> compiled context -> reader),
and leaves one readable report next to the raw artifact:

    benchmark_results/smoke/<stamp>/
        manifest.json       the exact cases that ran (pinned, not positionally cut)
        results.json        verbatim artifact from eval_locomo_full.py
        reader_inputs/*.txt the exact text each reader call saw (never gold)
        report.md           per-case table, category rollup, reader inputs
        run.log             full stdout/stderr of the eval run

Usage:
    python scripts/benchmarks/locomo_smoke.py             # 18 pinned cases
    python scripts/benchmarks/locomo_smoke.py -n 6        # first 6 pinned cases
    python scripts/benchmarks/locomo_smoke.py --list      # show the pinned set
    python scripts/benchmarks/locomo_smoke.py --no-llm    # contexts only, 0 LLM calls
    python scripts/benchmarks/locomo_smoke.py --show-prompts 4
    python scripts/benchmarks/locomo_smoke.py --rerank cross-encoder

Exit codes: 0 = every case answered, 1 = a case crashed or came back empty,
2 = preflight failed (dataset missing / Ollama unreachable).  Accuracy is
deliberately NOT a gate: 18 questions cannot decide a release.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "scripts" / "benchmarks" / "eval_locomo_full.py"
DEFAULT_MANIFEST = REPO / "benchmark_config" / "smoke_manifest.json"
DEFAULT_DATA = REPO / "datasets" / "external" / "locomo10.json"
DEFAULT_OUT_ROOT = REPO / "benchmark_results" / "smoke"

CAT_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# Must agree with artificial_memory.research.benchmarks.llm.FROZEN_MODEL; the
# preflight treats a mismatch as "reader not installed" rather than silently
# running a different model than the published numbers used.
FROZEN_MODEL = "qwen2.5:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"

#: Model used by each re-ranking backend when --rerank-model is not given.
RERANK_DEFAULT_MODEL = {
    "cross-encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ollama": "qwen2.5:7b-instruct",
}


# --------------------------------------------------------------------- inputs
def load_manifest(path: Path) -> dict:
    """Load a smoke manifest; a bare list is accepted and wrapped."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {"name": path.stem, "cases": raw}
    if not isinstance(raw, dict) or not raw.get("cases"):
        raise SystemExit(f"smoke manifest has no cases: {path}")
    return raw


def preflight_ollama(model: str = FROZEN_MODEL, base_url: str = OLLAMA_BASE_URL,
                     timeout: float = 5.0) -> str | None:
    """Return a human-readable error, or None when the reader is available."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout) as r:
            tags = json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return (
            f"Ollama is not reachable at {base_url} ({exc}).\n"
            "  start it with:  ollama serve\n"
            f"  then:           ollama pull {model}\n"
            "  or run with --no-llm for a 0-call plumbing check"
        )
    names = {m.get("name") for m in tags.get("models", []) if isinstance(m, dict)}
    if model not in names:
        found = ", ".join(sorted(n for n in names if n)) or "none"
        return (
            f"Ollama is running at {base_url} but {model!r} is not installed "
            f"(found: {found}).\n  pull it with:  ollama pull {model}"
        )
    return None


def git_revision() -> str:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(REPO), capture_output=True,
                text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return ""

    rev = _run("rev-parse", "--short", "HEAD") or "unknown"
    if _run("status", "--porcelain"):
        rev += " (dirty)"
    return rev


def _run_dir(root: Path, stamp: str) -> Path:
    run_dir = root / stamp
    n = 2
    while run_dir.exists():
        run_dir = root / f"{stamp}_{n}"
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _display(path: Path) -> str:
    """Repo-relative POSIX path for reports (Windows separators read badly)."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


# ------------------------------------------------------------------ execution
def run_eval(*, run_dir: Path, manifest: dict, data_path: Path,
             no_llm: bool, rerank: str = "off",
             rerank_model: str = "",
             speaker_first: bool = False,
             refusal_gate: bool = False) -> tuple[int, str]:
    """Invoke the canonical harness and return (exit code, combined log)."""
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    cmd = [
        sys.executable, str(EVAL),
        "--data", str(data_path),
        "--manifest", str(manifest_path),
    ]
    if rerank != "off":
        cmd += ["--rerank", rerank]
        if rerank_model:
            cmd += ["--rerank-model", rerank_model]
    if speaker_first:
        cmd += ["--speaker-first"]
    if refusal_gate:
        cmd += ["--refusal-gate"]
    if no_llm:
        # 0 LLM calls: compile the contexts and stop.  Proves retrieval and the
        # compiler still run, and shows how much evidence each case would get.
        cmd += ["--freeze-contexts", str(run_dir / "frozen_ctx")]
    else:
        cmd += [
            "--out", str(run_dir / "results.json"),
            "--dump-prompts", str(run_dir / "reader_inputs"),
        ]

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        cmd, cwd=str(REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    (run_dir / "run.log").write_text(log, encoding="utf-8")
    return proc.returncode, log


# --------------------------------------------------------------------- report
def _clip(text: object, limit: int) -> str:
    s = "" if text is None else str(text)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def normalize_results(results: list[dict]) -> list[dict]:
    """Artifact rows -> report rows (identical shape for both modes)."""
    rows = []
    for r in results:
        pred = str(r.get("predicted_answer") or "")
        error = pred if pred.startswith("<ERROR:") else ""
        rows.append({
            "id": r.get("question_id", "?"),
            "conv": r.get("sample_id") or r.get("conversation", ""),
            "cat": int(r.get("category") or 0),
            "question": r.get("question", ""),
            "gold": r.get("ground_truth", ""),
            "predicted": pred,
            "ok": bool(r.get("is_correct")),
            "ctx_tokens": int(r.get("context_tokens") or 0),
            "latency": float(r.get("latency_seconds") or 0.0),
            "error": error,
        })
    return rows


def normalize_frozen(rows: list[dict]) -> list[dict]:
    """frozen contexts.jsonl rows -> report rows with no prediction."""
    out = []
    for r in rows:
        out.append({
            "id": r.get("question_id", "?"),
            "conv": "",
            "cat": int(r.get("category") or 0),
            "question": r.get("question", ""),
            "gold": r.get("gold", ""),
            "predicted": None,
            "ok": None,
            "ctx_tokens": int(r.get("context_tokens") or 0),
            "latency": 0.0,
            "error": "",
        })
    return out


def build_report(rows: list[dict], meta: dict, prompts: dict[str, str] | None = None,
                 max_prompts: int = 2) -> str:
    """Render report.md.  Pure: every input arrives as an argument."""
    prompts = prompts or {}
    llm = meta["mode"] == "llm"
    total = len(rows)
    errored = [r for r in rows if r["error"]]
    answered = [r for r in rows if r["predicted"]]
    correct = [r for r in rows if r["ok"]]
    refused = [r for r in rows if r["cat"] == 5]

    lines: list[str] = []
    lines.append(f"# LoCoMo smoke - {meta['stamp']}")
    lines.append("")
    if llm:
        lines.append(
            "> **This is a plumbing check, not a benchmark score.** "
            f"{total} pinned cases from `{meta['name']}`: one per "
            "(conversation, category) cell on three conversations, plus the "
            "three questions that failed the 2026-10-26 reader autopsy. "
            "Publishable numbers live in `benchmark_results/RESULTS_REGISTRY.md`.")
    else:
        lines.append(
            "> **Context-only check, 0 LLM calls, not a benchmark score.** "
            f"{total} pinned cases from `{meta['name']}`. This proves retrieval "
            "and the context compiler still run and shows how much evidence each "
            "case would receive. No answers were generated, so there is no score.")
    lines.append("")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    lines.append(f"| reader | `{meta['model']}` (frozen: temp 0, seed 42) |" if llm
                 else "| reader | not called (`--no-llm`) |")
    lines.append(f"| prompt style | `{meta['prompt_style']}` |" if llm
                 else "| prompt style | n/a |")
    lines.append(f"| code | `{meta['revision']}` |")
    lines.append(f"| re-rank | {meta.get('rerank', 'off')} |")
    lines.append(f"| manifest | `{meta['name']}` (`{meta['manifest']}`) |")
    lines.append(f"| dataset | `{meta['data']}` |")
    lines.append(f"| wall clock | {meta['duration']:.1f}s |")
    if llm:
        mean_ctx = sum(r["ctx_tokens"] for r in rows) / max(1, total)
        lat = [r["latency"] for r in rows if r["latency"]]
        lines.append(
            f"| outcome | {len(correct)}/{total} correct "
            f"({len(correct) / max(1, total) * 100:.1f}%), {len(errored)} errors |")
        lines.append(f"| mean context | {mean_ctx:.0f} tokens |")
        if lat:
            lines.append(f"| mean latency | {sum(lat) / len(lat):.1f}s/case |")
    else:
        mean_ctx = sum(r["ctx_tokens"] for r in rows) / max(1, total)
        empty = [r for r in rows if not r["ctx_tokens"]]
        lines.append(f"| mean context | {mean_ctx:.0f} tokens |")
        lines.append(f"| empty contexts | {len(empty)} |")
    lines.append("")

    lines.append("## Per-case results")
    lines.append("")
    if llm:
        lines.append("| # | case | cat | verdict | ctx tok | lat | question | gold | predicted |")
        lines.append("|--:|------|-----|---------|--------:|----:|----------|------|-----------|")
    else:
        lines.append("| # | case | cat | ctx tok | question | gold |")
        lines.append("|--:|------|-----|--------:|----------|------|")
    for i, r in enumerate(rows, 1):
        verdict = "OK" if r["ok"] else ("ERROR" if r["error"] else "FAIL")
        gold = r["gold"] or ("(must refuse)" if r["cat"] == 5 else "(none)")
        if not llm:
            lines.append(
                f"| {i} | {r['id']} | {r['cat']} {CAT_NAMES.get(r['cat'], '?')} | "
                f"{r['ctx_tokens']} | {_clip(r['question'], 70)} | "
                f"{_clip(gold, 40)} |")
            continue
        lines.append(
            f"| {i} | {r['id']} | {r['cat']} {CAT_NAMES.get(r['cat'], '?')} | "
            f"{verdict} | {r['ctx_tokens']} | {r['latency']:.1f}s | "
            f"{_clip(r['question'], 70)} | {_clip(gold, 40)} | "
            f"{_clip(r['predicted'], 40)} |")
    lines.append("")

    by_cat: dict[int, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["cat"], []).append(r)
    lines.append("## Category rollup")
    lines.append("")
    if llm:
        lines.append("| cat | n | correct | share |")
        lines.append("|-----|--:|--------:|------:|")
    else:
        lines.append("| cat | n | mean ctx tok |")
        lines.append("|-----|--:|-------------:|")
    for cat in sorted(by_cat):
        group = by_cat[cat]
        if llm:
            good = sum(1 for r in group if r["ok"])
            lines.append(
                f"| {cat} {CAT_NAMES.get(cat, '?')} | {len(group)} | {good} | "
                f"{good / len(group) * 100:.0f}% |")
        else:
            mean = sum(r["ctx_tokens"] for r in group) / len(group)
            lines.append(f"| {cat} {CAT_NAMES.get(cat, '?')} | {len(group)} | {mean:.0f} |")
    lines.append("")

    if llm:
        if refused:
            credited = sum(1 for r in refused if r["ok"])
            lines.append(
                f"Category 5 (adversarial) refusal contract: {credited}/{len(refused)} "
                "credited.  Gold is empty by construction; any non-refusal is wrong.")
            lines.append("")
        failures = [r for r in rows if not r["ok"]]
        if failures:
            lines.append("## Failures in detail")
            lines.append("")
            for r in failures:
                lines.append(f"### {r['id']} - cat{r['cat']} {CAT_NAMES.get(r['cat'], '?')}")
                lines.append("")
                lines.append(f"- question: {r['question']}")
                lines.append(f"- gold: {r['gold'] or '(must refuse)'}")
                lines.append(f"- predicted: {r['predicted'] or '(empty)'}")
                lines.append(f"- context: {r['ctx_tokens']} tokens, {r['latency']:.1f}s")
                lines.append("")

    if prompts:
        shown = 0
        order = [r for r in rows if llm and not r["ok"]] + [r for r in rows if llm and r["ok"]]
        if not llm:
            order = rows
        for r in order:
            if shown >= max_prompts:
                break
            body = prompts.get(r["id"])
            if not body:
                continue
            shown += 1
            if llm:
                lines.append(f"## Reader input - {r['id']}")
                lines.append("")
                lines.append(f"Question: `{r['question']}`")
                lines.append("")
                lines.append("The exact text the model saw (question + compiled "
                             "evidence + instructions), truncated to 3000 characters.")
            else:
                lines.append(f"## Compiled context - {r['id']}")
                lines.append("")
                lines.append(f"Question: `{r['question']}`")
                lines.append("")
                lines.append("The evidence the compiler would hand to the reader. "
                             "No reader call was made in this mode. Truncated to "
                             "3000 characters.")
            lines.append("")
            lines.append("```text")
            lines.append(body[:3000] + ("\n[... truncated]" if len(body) > 3000 else ""))
            lines.append("```")
            lines.append("")
    return "\n".join(lines) + "\n"


def load_run(run_dir: Path) -> dict:
    """Read a finished smoke run directory into report rows."""
    results_file = run_dir / "results.json"
    if results_file.exists():
        artifact = json.loads(results_file.read_text(encoding="utf-8"))
        return {
            "mode": "llm",
            "rows": normalize_results(artifact.get("results", [])),
            "config": artifact.get("config") or {},
        }
    ctx_file = run_dir / "frozen_ctx" / "contexts.jsonl"
    if ctx_file.exists():
        frozen = [
            json.loads(line)
            for line in ctx_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        return {"mode": "contexts", "rows": normalize_frozen(frozen), "config": {}}
    raise SystemExit(f"no results.json or frozen_ctx/contexts.jsonl in {run_dir}")


def build_comparison(before: dict, after: dict, labels: tuple[str, str]) -> str:
    """Diff two smoke runs case by case.

    This is the "did my change help?" surface: fixed, broken, and unchanged
    cases are listed explicitly, so a net-zero result cannot be mistaken for a
    win (or a loss) on the strength of a single total.
    """
    label_b, label_a = labels
    a = {r["id"]: r for r in before["rows"]}
    b = {r["id"]: r for r in after["rows"]}
    shared = [qid for qid in a if qid in b]
    llm = before["mode"] == "llm" and after["mode"] == "llm"

    lines = ["# Smoke comparison", ""]
    lines.append(f"- before: `{label_b}`")
    lines.append(f"- after: `{label_a}`")
    lines.append(f"- shared cases: {len(shared)}")
    lines.append("")
    if not shared:
        lines.append("No case ids in common: the manifests differ.")
        return "\n".join(lines) + "\n"

    if llm:
        fixed = [q for q in shared if not a[q]["ok"] and b[q]["ok"]]
        broken = [q for q in shared if a[q]["ok"] and not b[q]["ok"]]
        before_ok = sum(1 for q in shared if a[q]["ok"])
        after_ok = sum(1 for q in shared if b[q]["ok"])
        lines.append(f"- correct: {before_ok} -> {after_ok} ({after_ok - before_ok:+d})")
        lines.append(f"- fixed: {len(fixed)} | broken: {len(broken)} | "
                     f"unchanged: {len(shared) - len(fixed) - len(broken)}")
        lines.append("")
        lines.append("| case | cat | before | after | before pred | after pred |")
        lines.append("|------|-----|--------|-------|-------------|------------|")
        for qid in shared:
            ra, rb = a[qid], b[qid]
            if ra["ok"] == rb["ok"]:
                continue
            lines.append(
                f"| {qid} | {ra['cat']} {CAT_NAMES.get(ra['cat'], '?')} | "
                f"{'OK' if ra['ok'] else 'FAIL'} | {'OK' if rb['ok'] else 'FAIL'} | "
                f"{_clip(ra['predicted'], 40)} | {_clip(rb['predicted'], 40)} |")
        lines.append("")
    else:
        lines.append("| case | cat | before tok | after tok | delta |")
        lines.append("|------|-----|-----------:|----------:|------:|")
        for qid in shared:
            ra, rb = a[qid], b[qid]
            lines.append(f"| {qid} | {ra['cat']} {CAT_NAMES.get(ra['cat'], '?')} | "
                         f"{ra['ctx_tokens']} | {rb['ctx_tokens']} | "
                         f"{rb['ctx_tokens'] - ra['ctx_tokens']:+d} |")
        lines.append("")

    mean_a = sum(a[q]["ctx_tokens"] for q in shared) / len(shared)
    mean_b = sum(b[q]["ctx_tokens"] for q in shared) / len(shared)
    lines.append(f"Mean context tokens: {mean_a:.0f} -> {mean_b:.0f} "
                 f"({mean_b - mean_a:+.0f}).")
    lines.append("")
    lines.append("> 18 pinned dev cases. A delta here is a lead to investigate, "
                 "not a published result; confirm on the full run before claiming it.")
    return "\n".join(lines) + "\n"


def console_summary(rows: list[dict], meta: dict) -> str:
    llm = meta["mode"] == "llm"
    total = len(rows)
    correct = sum(1 for r in rows if r["ok"])
    errored = sum(1 for r in rows if r["error"])
    width = 74
    out = ["", "=" * width]
    if llm:
        out.append(f"  smoke {meta['name']}: {correct}/{total} correct, "
                   f"{errored} errors, {meta['duration']:.1f}s")
    else:
        out.append(f"  smoke {meta['name']}: {total} contexts built, "
                   f"0 LLM calls, {meta['duration']:.1f}s")
    out.append("=" * width)
    header = f"  {'case':<18}{'cat':<15}{'result':<8}{'tok':>6}"
    out.append(header)
    out.append("-" * width)
    for r in rows:
        if llm:
            result = "OK" if r["ok"] else ("ERROR" if r["error"] else "FAIL")
        else:
            result = "ok" if r["ctx_tokens"] else "EMPTY"
        cat = f"{r['cat']} {CAT_NAMES.get(r['cat'], '?')}"
        out.append(f"  {r['id']:<18}{cat[:14]:<14} {result:<7}{r['ctx_tokens']:>6}")
    out.append("=" * width)
    return "\n".join(out)


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    try:  # Windows consoles default to cp932; the report is UTF-8 either way.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="Run the pinned LoCoMo smoke set and write a readable report.")
    ap.add_argument("-n", "--questions", type=int, default=0,
                    help="run only the first N pinned cases (0 = all)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="smoke manifest (default: benchmark_config/smoke_manifest.json)")
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="LoCoMo json (default: datasets/external/locomo10.json)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT),
                    help="parent directory for the timestamped run folder")
    ap.add_argument("--no-llm", action="store_true",
                    help="compile contexts only; makes no reader calls")
    ap.add_argument("--rerank", default="off", choices=["off", "cross-encoder", "ollama"],
                    help="re-order retrieved evidence with a learned relevance model "
                         "before the budgeted compiler selects it "
                         "('cross-encoder' needs the [vector] extra; 'ollama' uses "
                         "the local server)")
    ap.add_argument("--rerank-model", default="",
                    help="re-ranker model (default: ms-marco-MiniLM-L-6-v2 for "
                         "cross-encoder, qwen2.5:7b-instruct for ollama)")
    ap.add_argument("--speaker-first", action="store_true",
                    help="reorder reader context so the question's target speaker's "
                         "evidence appears first")
    ap.add_argument("--refusal-gate", action="store_true",
                    help="apply the deterministic answer-provenance gate: an answer "
                         "whose wording only comes from another speaker's turn "
                         "becomes a refusal")
    ap.add_argument("--show-prompts", type=int, default=2,
                    help="how many full reader inputs to inline in report.md (default 2)")
    ap.add_argument("--list", action="store_true",
                    help="print the pinned cases and exit")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), default=None,
                    help="diff two finished run directories and exit")
    args = ap.parse_args(argv)

    if args.compare:
        before_dir = Path(args.compare[0]).expanduser().resolve()
        after_dir = Path(args.compare[1]).expanduser().resolve()
        report = build_comparison(load_run(before_dir), load_run(after_dir),
                                  (before_dir.name, after_dir.name))
        out = after_dir / "comparison.md"
        out.write_text(report, encoding="utf-8")
        print(report)
        print(f"written: {out}")
        return 0

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}")
        return 2
    manifest = load_manifest(manifest_path)
    cases = list(manifest["cases"])
    if args.questions > 0:
        cases = cases[: args.questions]
    selected = {**manifest, "cases": cases}

    if args.list:
        print(f"{manifest.get('name', manifest_path.stem)}: {len(cases)} cases")
        for c in cases:
            print(f"  conv_idx={c['conv_idx']:<2} q_idx={c['q_idx']:<4} "
                  f"cat{c.get('category', '?')}  "
                  f"{c.get('conversation', '')}  {_clip(c.get('question', ''), 80)}")
        return 0

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        print(f"error: dataset not found: {data_path}")
        print("  fetch it with:  python scripts/benchmarks/bootstrap_official_benchmarks.ps1")
        return 2
    if not EVAL.exists():
        print(f"error: canonical harness missing: {EVAL}")
        return 2

    model = manifest.get("reader", FROZEN_MODEL) if not args.no_llm else FROZEN_MODEL
    if not args.no_llm:
        problem = preflight_ollama(model)
        if problem:
            print(f"error: {problem}")
            return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _run_dir(Path(args.out_dir).expanduser().resolve(), stamp)
    mode = "contexts" if args.no_llm else "llm"
    print(f"smoke run: {manifest.get('name', manifest_path.stem)} "
          f"({len(cases)} cases, mode={mode})")
    print(f"output   : {run_dir}")
    if not args.no_llm:
        print(f"reader   : {model} via {OLLAMA_BASE_URL} "
              f"(roughly {len(cases) * 3}s)")
    if args.rerank != "off":
        print(f"rerank   : {args.rerank} "
              f"{args.rerank_model or RERANK_DEFAULT_MODEL[args.rerank]}")

    t0 = time.perf_counter()
    code, log = run_eval(run_dir=run_dir, manifest=selected,
                         data_path=data_path, no_llm=args.no_llm,
                         rerank=args.rerank, rerank_model=args.rerank_model,
                         speaker_first=args.speaker_first,
                         refusal_gate=args.refusal_gate)
    duration = time.perf_counter() - t0

    if args.rerank == "off":
        rerank_label = "off"
    else:
        rerank_label = f"{args.rerank} `{args.rerank_model or RERANK_DEFAULT_MODEL[args.rerank]}`"

    meta = {
        "stamp": run_dir.name,
        "name": manifest.get("name", manifest_path.stem),
        "manifest": _display(manifest_path),
        "data": _display(data_path),
        "mode": mode,
        "model": model,
        "prompt_style": "concise",
        "rerank": rerank_label,
        "speaker_first": args.speaker_first,
        "refusal_gate": args.refusal_gate,
        "revision": git_revision(),
        "duration": duration,
    }

    if args.no_llm:
        ctx_file = run_dir / "frozen_ctx" / "contexts.jsonl"
        if not ctx_file.exists():
            print(log[-4000:])
            print(f"error: harness produced no contexts (exit {code}); see {run_dir / 'run.log'}")
            return 2
        frozen = [
            json.loads(line) for line in ctx_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = normalize_frozen(frozen)
        prompts = {r["question_id"]: r.get("context", "") for r in frozen}
    else:
        results_file = run_dir / "results.json"
        if not results_file.exists():
            print(log[-4000:])
            print(f"error: harness produced no results (exit {code}); "
                  f"see {run_dir / 'run.log'}")
            return 2
        artifact = json.loads(results_file.read_text(encoding="utf-8"))
        rows = normalize_results(artifact.get("results", []))
        config = artifact.get("config") or {}
        meta["prompt_style"] = config.get("prompt_style", meta["prompt_style"])
        prompts = {}
        prompt_dir = run_dir / "reader_inputs"
        if prompt_dir.exists():
            for f in sorted(prompt_dir.glob("*.txt")):
                prompts[f.stem] = f.read_text(encoding="utf-8")

    if not rows:
        print(log[-2000:])
        print(f"error: no cases ran; see {run_dir / 'run.log'}")
        return 2 if code != 0 else 1

    report = build_report(rows, meta, prompts=prompts, max_prompts=args.show_prompts)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(console_summary(rows, meta))
    print(f"\nreport   : {run_dir / 'report.md'}")
    print(f"artifact : {run_dir / ('frozen_ctx/contexts.jsonl' if args.no_llm else 'results.json')}")

    errored = [r for r in rows if r["error"]]
    empty = [r for r in rows if not r["ctx_tokens"]]
    if errored:
        for r in errored:
            print(f"error in {r['id']}: {r['error']}")
        return 1
    if args.no_llm and empty:
        print(f"error: {len(empty)} case(s) produced an empty context")
        return 1
    if not args.no_llm and any(not r["predicted"] for r in rows):
        print("error: at least one case came back with an empty answer")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
