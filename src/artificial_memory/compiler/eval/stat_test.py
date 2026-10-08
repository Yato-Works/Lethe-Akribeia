"""
Statistical Testing for Lethe-Akribeia V4 Benchmark Comparisons.

Provides McNemar's test, Bootstrap confidence intervals, and paired comparison
for rigorous significance testing between model versions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

try:
    from scipy.stats import binomtest
    HAS_BINOMTEST = True
except ImportError:
    HAS_BINOMTEST = False


@dataclass(frozen=True)
class McNemarResult:
    """Result of McNemar's test for paired comparison."""
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    b: int  # Model A correct, Model B wrong
    c: int  # Model A wrong, Model B correct
    n_discordant: int


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval."""
    mean: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    metric_name: str


@dataclass(frozen=True)
class ComparisonResult:
    """Full comparison between two model runs."""
    model_a_name: str
    model_b_name: str
    accuracy_a: float
    accuracy_b: float
    accuracy_diff: float
    mcnemar: McNemarResult
    bootstrap_ci_a: BootstrapCI
    bootstrap_ci_b: BootstrapCI
    bootstrap_ci_diff: BootstrapCI
    per_type: dict[str, dict[str, Any]]


class StatisticalTester:
    """
    Rigorous statistical testing for benchmark comparisons.
    
    Uses:
    - McNemar's test: For paired binary outcomes (same questions, different models)
    - Bootstrap CI: For confidence intervals on accuracy/token metrics
    - Paired t-test: For continuous metrics (tokens, latency)
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.rng = np.random.default_rng(seed)
    
    def mcnemar_test(
        self,
        results_a: list[bool],
        results_b: list[bool],
    ) -> McNemarResult:
        """
        McNemar's test for paired comparison.
        
        Args:
            results_a: List of correctness booleans for model A
            results_b: List of correctness booleans for model B
            
        Returns:
            McNemarResult with statistic, p-value, significance
        """
        if len(results_a) != len(results_b):
            raise ValueError("Results must have same length")
        
        results_a = np.array(results_a, dtype=bool)
        results_b = np.array(results_b, dtype=bool)
        
        # Discordant pairs
        b = np.sum(results_a & ~results_b)  # A correct, B wrong
        c = np.sum(~results_a & results_b)  # A wrong, B correct
        n_discordant = b + c
        
        if n_discordant == 0:
            return McNemarResult(
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                b=b,
                c=c,
                n_discordant=0,
            )
        
        # McNemar's chi-square statistic (with continuity correction)
        statistic = (abs(b - c) - 1) ** 2 / n_discordant
        
        # p-value from chi-square distribution with 1 df
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        # Exact binomial test (more accurate for small samples)
        if HAS_BINOMTEST:
            exact_p = binomtest(min(b, c), n_discordant, 0.5, alternative='two-sided').pvalue
            p_value = min(p_value, exact_p)
        
        return McNemarResult(
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            b=int(b),
            c=int(c),
            n_discordant=int(n_discordant),
        )
    
    def bootstrap_ci(
        self,
        values: list[float],
        metric_name: str = "metric",
    ) -> BootstrapCI:
        """Bootstrap confidence interval for a metric."""
        values = np.array(values)
        n = len(values)
        
        if n == 0:
            return BootstrapCI(0, 0, 0, self.confidence_level, 0, metric_name)
        
        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            sample = self.rng.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        mean_val = float(np.mean(values))
        
        # Percentile CI
        alpha = 1 - self.confidence_level
        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
        
        return BootstrapCI(
            mean=mean_val,
            ci_lower=lower,
            ci_upper=upper,
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            metric_name=metric_name,
        )
    
    def bootstrap_ci_diff(
        self,
        values_a: list[float],
        values_b: list[float],
        metric_name: str = "diff",
    ) -> BootstrapCI:
        """Bootstrap CI for paired difference."""
        if len(values_a) != len(values_b):
            raise ValueError("Paired values must have same length")
        
        values_a = np.array(values_a)
        values_b = np.array(values_b)
        n = len(values_a)
        
        if n == 0:
            return BootstrapCI(0, 0, 0, self.confidence_level, 0, metric_name)
        
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.choice(n, size=n, replace=True)
            diff = np.mean(values_a[idx] - values_b[idx])
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        mean_diff = float(np.mean(values_a - values_b))
        
        alpha = 1 - self.confidence_level
        lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))
        
        return BootstrapCI(
            mean=mean_diff,
            ci_lower=lower,
            ci_upper=upper,
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            metric_name=metric_name,
        )
    
    def compare_runs(
        self,
        results_a: list[dict],
        results_b: list[dict],
        name_a: str = "Model A",
        name_b: str = "Model B",
    ) -> ComparisonResult:
        """
        Full statistical comparison between two benchmark runs.
        
        Args:
            results_a: List of result dicts with 'is_correct', 'question_type', 'prompt_tokens'
            results_b: Same for model B (must be same questions in same order)
            name_a, name_b: Names for reporting
            
        Returns:
            ComparisonResult with all statistical tests
        """
        if len(results_a) != len(results_b):
            raise ValueError("Runs must have same number of questions")
        
        # Extract paired data
        correct_a = [r['is_correct'] for r in results_a]
        correct_b = [r['is_correct'] for r in results_b]
        tokens_a = [r.get('prompt_tokens', 0) for r in results_a]
        tokens_b = [r.get('prompt_tokens', 0) for r in results_b]
        
        # Overall accuracy
        acc_a = np.mean(correct_a)
        acc_b = np.mean(correct_b)
        acc_diff = acc_b - acc_a
        
        # McNemar test
        mcnemar = self.mcnemar_test(correct_a, correct_b)
        
        # Bootstrap CIs
        ci_a = self.bootstrap_ci([float(c) for c in correct_a], f"{name_a}_accuracy")
        ci_b = self.bootstrap_ci([float(c) for c in correct_b], f"{name_b}_accuracy")
        ci_diff = self.bootstrap_ci_diff(
            [float(c) for c in correct_a],
            [float(c) for c in correct_b],
            "accuracy_diff",
        )
        
        # Per-type analysis
        per_type = {}
        all_types = set(r.get('question_type', 'unknown') for r in results_a)
        
        for qtype in all_types:
            idx = [i for i, r in enumerate(results_a) if r.get('question_type') == qtype]
            if not idx:
                continue
            
            type_a = [correct_a[i] for i in idx]
            type_b = [correct_b[i] for i in idx]
            
            type_mcnemar = self.mcnemar_test(type_a, type_b)
            type_ci_a = self.bootstrap_ci([float(c) for c in type_a], f"{name_a}_{qtype}")
            type_ci_b = self.bootstrap_ci([float(c) for c in type_b], f"{name_b}_{qtype}")
            
            per_type[qtype] = {
                'accuracy_a': np.mean(type_a),
                'accuracy_b': np.mean(type_b),
                'diff': np.mean(type_b) - np.mean(type_a),
                'mcnemar_p': type_mcnemar.p_value,
                'mcnemar_significant': type_mcnemar.significant,
                'mcnemar_b': type_mcnemar.b,
                'mcnemar_c': type_mcnemar.c,
                'ci_a': [type_ci_a.ci_lower, type_ci_a.ci_upper],
                'ci_b': [type_ci_b.ci_lower, type_ci_b.ci_upper],
                'n': len(idx),
            }
        
        # Token comparison (paired t-test)
        token_diff = np.array(tokens_b) - np.array(tokens_a)
        t_stat, t_p = stats.ttest_rel(tokens_b, tokens_a) if len(tokens_a) > 1 else (0, 1)
        
        return ComparisonResult(
            model_a_name=name_a,
            model_b_name=name_b,
            accuracy_a=float(acc_a),
            accuracy_b=float(acc_b),
            accuracy_diff=float(acc_diff),
            mcnemar=mcnemar,
            bootstrap_ci_a=ci_a,
            bootstrap_ci_b=ci_b,
            bootstrap_ci_diff=ci_diff,
            per_type=per_type,
        )
    
    def format_comparison(self, result: ComparisonResult) -> str:
        """Format comparison result as readable string."""
        lines = [
            f"\n{'='*70}",
            f"STATISTICAL COMPARISON: {result.model_a_name} vs {result.model_b_name}",
            f"{'='*70}",
            f"\nOverall Accuracy:",
            f"  {result.model_a_name}: {result.accuracy_a:.1%}",
            f"  {result.model_b_name}: {result.accuracy_b:.1%}",
            f"  Difference: {result.accuracy_diff:+.1%}",
            f"\nMcNemar's Test (paired):",
            f"  Discordant pairs: b={result.mcnemar.b} (A only), c={result.mcnemar.c} (B only)",
            f"  χ² = {result.mcnemar.statistic:.3f}, p = {result.mcnemar.p_value:.4f}",
            f"  Significant at α={result.mcnemar.alpha}: {'YES' if result.mcnemar.significant else 'NO'}",
            f"\nBootstrap 95% CI (Accuracy):",
            f"  {result.model_a_name}: [{result.bootstrap_ci_a.ci_lower:.1%}, {result.bootstrap_ci_a.ci_upper:.1%}]",
            f"  {result.model_b_name}: [{result.bootstrap_ci_b.ci_lower:.1%}, {result.bootstrap_ci_b.ci_upper:.1%}]",
            f"  Difference: [{result.bootstrap_ci_diff.ci_lower:.1%}, {result.bootstrap_ci_diff.ci_upper:.1%}]",
            f"\nPer-Type Breakdown:",
        ]
        
        for qtype, stats in sorted(result.per_type.items()):
            sig = " *" if stats['mcnemar_significant'] else ""
            lines.append(
                f"  {qtype:30s} A: {stats['accuracy_a']:.1%}  B: {stats['accuracy_b']:.1%}  "
                f"Δ: {stats['diff']:+.1%}  p={stats['mcnemar_p']:.4f}{sig}  (n={stats['n']})"
            )
        
        lines.append(f"\n{'='*70}")
        return "\n".join(lines)


def load_results(filepath: str | Path) -> list[dict]:
    """Load benchmark results from JSON/JSONL."""
    path = Path(filepath)
    
    with open(path) as f:
        if path.suffix == '.jsonl':
            return [json.loads(line) for line in f]
        else:
            data = json.load(f)
            if 'results' in data:
                return data['results']
            return data


def compare_result_files(
    file_a: str | Path,
    file_b: str | Path,
    name_a: str = "Model A",
    name_b: str = "Model B",
    alpha: float = 0.05,
) -> ComparisonResult:
    """Convenience function to compare two result files."""
    results_a = load_results(file_a)
    results_b = load_results(file_b)
    
    tester = StatisticalTester(alpha=alpha)
    return tester.compare_runs(results_a, results_b, name_a, name_b)


if __name__ == "__main__":
    # Quick test
    tester = StatisticalTester()
    
    # Simulated results
    np.random.seed(42)
    n = 250
    results_a = np.random.binomial(1, 0.74, n).astype(bool).tolist()
    results_b = np.random.binomial(1, 0.78, n).astype(bool).tolist()
    
    mcnemar = tester.mcnemar_test(results_a, results_b)
    print("McNemar Test:")
    print(f"  b={mcnemar.b}, c={mcnemar.c}")
    print(f"  χ²={mcnemar.statistic:.3f}, p={mcnemar.p_value:.4f}, sig={mcnemar.significant}")
    
    ci_a = tester.bootstrap_ci([float(x) for x in results_a])
    ci_b = tester.bootstrap_ci([float(x) for x in results_b])
    ci_diff = tester.bootstrap_ci_diff([float(x) for x in results_a], [float(x) for x in results_b])
    
    print(f"\nBootstrap CI A: [{ci_a.ci_lower:.1%}, {ci_a.ci_upper:.1%}]")
    print(f"Bootstrap CI B: [{ci_b.ci_lower:.1%}, {ci_b.ci_upper:.1%}]")
    print(f"Bootstrap CI Diff: [{ci_diff.ci_lower:.1%}, {ci_diff.ci_upper:.1%}]")