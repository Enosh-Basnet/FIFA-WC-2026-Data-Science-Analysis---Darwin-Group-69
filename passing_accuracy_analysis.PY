"""
FIFA World Cup 2026 - Passing Accuracy Analysis
=================================================
Do teams that advanced past the group stage pass the ball more
accurately than teams that were eliminated?

    H0: mu_advanced == mu_eliminated
    H1: mu_advanced  >  mu_eliminated

Design
------
Population : the 48 national teams that played the WC2026 group stage.
Sample     : stratified random sample (75% per stratum), stratified on
             `Outcome` so the sample keeps the population's
             Advanced:Eliminated ratio.
Assumptions checked before trusting the t-test:
    - Shapiro-Wilk normality per group (t-test is fairly robust to mild
      violations at these sample sizes, but worth knowing about).
    - Levene's test for equal variances (decides whether Welch's or
      Student's t-test is the right call; we default to Welch's
      regardless since it's strictly safer).
Effect size: Cohen's d, because a p-value tells you "is there a
             difference", not "does the difference matter".

Source: FIFA official team statistics, "Passing Accuracy Stats by Team",
Canada/Mexico/USA 2026.

Usage
-----
    python wc2026_passing_analysis.py [path/to/team_passing_data.csv]

Defaults to ./team_passing_data.csv if no path is given.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RANDOM_SEED = 42
SAMPLING_FRACTION = 0.75
ALPHA = 0.05

VALUE_COL = "Pass_Accuracy_Pct"
GROUP_COL = "Outcome"
ADVANCED = "Advanced"
ELIMINATED = "Eliminated"


class DataValidationError(Exception):
    """Raised when the input CSV doesn't have what we need to run the analysis."""


# --------------------------------------------------------------------
# 1. Data loading & cleaning
# --------------------------------------------------------------------
def load_clean_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"Can't find '{path}'. Check the file path/name.")

    df = pd.read_csv(path)

    required = {"Team", GROUP_COL, VALUE_COL}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"CSV is missing expected column(s): {sorted(missing)}")

    df = df.drop_duplicates(subset="Team")
    df = df.dropna(subset=[VALUE_COL, GROUP_COL])
    df[GROUP_COL] = df[GROUP_COL].str.strip()

    known_outcomes = {ADVANCED, ELIMINATED}
    unexpected = set(df[GROUP_COL].unique()) - known_outcomes
    if unexpected:
        raise DataValidationError(f"Unexpected values in '{GROUP_COL}': {unexpected}")

    return df


# --------------------------------------------------------------------
# 2. Stratified sampling
# --------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, strata_col: str, frac: float, seed: int) -> pd.DataFrame:
    """Sample a fixed fraction from each stratum so the group ratio in the
    sample matches the population, instead of leaving that to chance."""
    rng = np.random.default_rng(seed)
    parts = [
        group.sample(frac=frac, random_state=rng.integers(0, 2**32 - 1))
        for _, group in df.groupby(strata_col, sort=False)
    ]
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------
# 3. Descriptive statistics
# --------------------------------------------------------------------
def describe_group(series: pd.Series, label: str) -> dict:
    return {
        "Group": label,
        "n": len(series),
        "Mean": round(series.mean(), 2),
        "Median": round(series.median(), 2),
        "Std Dev": round(series.std(ddof=1), 2),
        "Min": round(series.min(), 2),
        "Max": round(series.max(), 2),
    }


# --------------------------------------------------------------------
# 4. Assumption checks
# --------------------------------------------------------------------
@dataclass(frozen=True)
class NormalityResult:
    group: str
    w_stat: float
    p_value: float

    @property
    def looks_normal(self) -> bool:
        return self.p_value > ALPHA

    def __str__(self) -> str:
        verdict = "no strong evidence against normality" if self.looks_normal else "deviates from normal"
        return f"{self.group}: W={self.w_stat:.3f}, p={self.p_value:.4f} ({verdict})"


def check_normality(series: pd.Series, label: str) -> NormalityResult:
    w_stat, p_value = stats.shapiro(series)
    return NormalityResult(label, w_stat, p_value)


@dataclass(frozen=True)
class LeveneResult:
    stat: float
    p_value: float

    @property
    def equal_variances(self) -> bool:
        return self.p_value > ALPHA

    def __str__(self) -> str:
        verdict = "variances look roughly equal" if self.equal_variances else "variances look unequal"
        return f"Levene's stat={self.stat:.3f}, p={self.p_value:.4f} ({verdict})"


def check_equal_variance(a: pd.Series, b: pd.Series) -> LeveneResult:
    stat, p_value = stats.levene(a, b)
    return LeveneResult(stat, p_value)


# --------------------------------------------------------------------
# 5. Inferential statistics
# --------------------------------------------------------------------
@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    confidence: float

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return f"mean = {self.mean:.2f}%  |  {pct}% CI = ({self.lower:.2f}%, {self.upper:.2f}%)"


def confidence_interval(series: pd.Series, confidence: float = 0.95) -> ConfidenceInterval:
    n = len(series)
    mean = series.mean()
    sem = series.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * sem
    return ConfidenceInterval(mean, mean - margin, mean + margin, confidence)


@dataclass(frozen=True)
class WelchTTestResult:
    t_stat: float
    df: float
    p_one_sided: float
    alpha: float

    @property
    def reject_null(self) -> bool:
        return self.p_one_sided < self.alpha


def welch_one_sided_ttest(greater: pd.Series, lesser: pd.Series, alpha: float = ALPHA) -> WelchTTestResult:
    """H1: mean(greater) > mean(lesser). Welch's over Student's because we
    shouldn't assume the two groups share the same variance going in."""
    t_stat, p_two_sided = stats.ttest_ind(greater, lesser, equal_var=False)
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

    s1, s2 = greater.var(ddof=1), lesser.var(ddof=1)
    n1, n2 = len(greater), len(lesser)
    df_welch = (s1 / n1 + s2 / n2) ** 2 / (
        (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    )
    return WelchTTestResult(t_stat, df_welch, p_one_sided, alpha)


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Pooled-SD effect size. Rule of thumb: ~0.2 small, ~0.5 medium, ~0.8 large."""
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_sd


def effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# --------------------------------------------------------------------
# 6. Report
# --------------------------------------------------------------------
def build_report(csv_path: Path) -> str:
    df = load_clean_data(csv_path)
    sample = stratified_sample(df, GROUP_COL, SAMPLING_FRACTION, RANDOM_SEED)
    adv = sample.loc[sample[GROUP_COL] == ADVANCED, VALUE_COL]
    eli = sample.loc[sample[GROUP_COL] == ELIMINATED, VALUE_COL]

    desc_table = pd.DataFrame([describe_group(adv, ADVANCED), describe_group(eli, ELIMINATED)])
    adv_normality = check_normality(adv, ADVANCED)
    eli_normality = check_normality(eli, ELIMINATED)
    levene = check_equal_variance(adv, eli)
    adv_ci = confidence_interval(adv)
    eli_ci = confidence_interval(eli)
    ttest = welch_one_sided_ttest(adv, eli)
    d = cohens_d(adv, eli)

    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)

    section("1. DATA WRANGLING")
    lines.append(f"Total teams in cleaned dataset: {len(df)}")
    lines.append(df[GROUP_COL].value_counts().to_string())
    lines.append("")

    section("2. DATA PREPARATION AND SAMPLING")
    lines.append(f"Population size (N): {len(df)} teams")
    lines.append(f"Sample size (n): {len(sample)} teams "
                 f"(stratified, {int(SAMPLING_FRACTION * 100)}% per stratum)")
    lines.append(sample[GROUP_COL].value_counts().to_string())
    lines.append("")

    section("3. DESCRIPTIVE STATISTICS (Sample)")
    lines.append(desc_table.to_string(index=False))
    lines.append("")

    section("4. ASSUMPTION CHECKS")
    lines.append("Normality (Shapiro-Wilk, H0: sample drawn from a normal distribution):")
    lines.append(f"  {adv_normality}")
    lines.append(f"  {eli_normality}")
    lines.append("Equal variances (Levene's test):")
    lines.append(f"  {levene}")
    lines.append("Note: proceeding with Welch's t-test regardless, since it doesn't")
    lines.append("require equal variances and loses little power when they are equal.")
    lines.append("")

    section("5. 95% CONFIDENCE INTERVALS (Population mean pass accuracy)")
    lines.append(f"Advanced teams  : {adv_ci}")
    lines.append(f"Eliminated teams: {eli_ci}")
    lines.append("")

    section("6. TWO-SAMPLE T-TEST (Welch's t-test, unequal variances)")
    lines.append(f"H0: mean pass accuracy ({ADVANCED}) = mean pass accuracy ({ELIMINATED})")
    lines.append(f"H1: mean pass accuracy ({ADVANCED}) > mean pass accuracy ({ELIMINATED})")
    lines.append("")
    lines.append(f"t-statistic        : {ttest.t_stat:.3f}")
    lines.append(f"degrees of freedom  : {ttest.df:.2f}")
    lines.append(f"p-value (one-tailed): {ttest.p_one_sided:.4f}")
    lines.append(f"significance level  : alpha = {ttest.alpha}")
    lines.append("")
    if ttest.reject_null:
        lines.append("Decision: Reject H0.")
        lines.append("Conclusion: statistically significant evidence that Advanced teams")
        lines.append("have a higher mean passing accuracy than Eliminated teams.")
    else:
        lines.append("Decision: Fail to reject H0.")
        lines.append("Conclusion: not enough evidence that Advanced teams have a higher")
        lines.append("mean passing accuracy than Eliminated teams.")
    lines.append("")

    section("7. EFFECT SIZE")
    lines.append(f"Cohen's d = {d:.2f} ({effect_size_label(d)} effect)")
    lines.append("Statistical significance says the gap probably isn't noise;")
    lines.append("effect size says how big that gap actually is in practice.")
    lines.append("")

    section("SUMMARY")
    mean_diff = adv_ci.mean - eli_ci.mean
    lines.append(f"Sample mean difference (Advanced - Eliminated): {mean_diff:.2f} percentage points")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WC2026 passing accuracy: Advanced vs Eliminated")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="team_passing_data.csv",
        help="Path to the team passing data CSV (default: ./team_passing_data.csv)",
    )
    args = parser.parse_args(argv)

    np.random.seed(RANDOM_SEED)

    try:
        report = build_report(Path(args.csv_path))
    except DataValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())