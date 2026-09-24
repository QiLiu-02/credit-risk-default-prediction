"""Cleaning and feature engineering for the Give Me Some Credit data.

`CreditCleaner` applies the Phase 1 cleaning rules the correct way: every number it needs
(medians, caps) is learned in `fit` from the TRAINING data only, then reused unchanged in
`transform` for any data (training, test, or future borrowers). `add_features` builds the
Phase 3 features and needs no learning, so it is a plain function.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "SeriousDlqin2yrs"
P30 = "NumberOfTime30-59DaysPastDueNotWorse"
P60 = "NumberOfTime60-89DaysPastDueNotWorse"
P90 = "NumberOfTimes90DaysLate"
PAST_DUE_COLS = [P30, P60, P90]
UTIL = "RevolvingUtilizationOfUnsecuredLines"

AGE_BINS = [17, 29, 39, 49, 59, 69, 120]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]


class CreditCleaner(BaseEstimator, TransformerMixin):
    """Phase 1 cleaning rules, with all learned values taken from the training data only.

    Rules (see notebooks/01_data_cleaning.ipynb for the reasoning):
      * 96/98 in the past-due columns are special codes: flag them, then set to the largest genuine count.
      * Utilization above `util_cap` is a recording error: cap it.
      * DebtRatio is unreliable if income is missing or the ratio exceeds `dr_max_plausible`:
        flag it and replace it with the median ratio of the reliable rows.
      * Income is capped at a high quantile, and missing income is filled with the median for the borrower's age band.
      * Missing dependents are filled with the median (0). Every fill gets a "was missing" flag.
    """

    def __init__(self, util_cap=2.0, dr_max_plausible=5.0, income_cap_quantile=0.999, code_threshold=96):
        self.util_cap = util_cap
        self.dr_max_plausible = dr_max_plausible
        self.income_cap_quantile = income_cap_quantile
        self.code_threshold = code_threshold

    def fit(self, X, y=None):
        # Largest genuine (non-code) count in each past-due column
        self.past_due_max_ = {c: X.loc[X[c] < self.code_threshold, c].max() for c in PAST_DUE_COLS}

        # Median DebtRatio among reliable rows (income present and ratio not absurd)
        reliable = X.MonthlyIncome.notna() & (X.DebtRatio <= self.dr_max_plausible)
        self.debt_ratio_fill_ = X.loc[reliable, "DebtRatio"].median()

        # Income cap, and median income within each age band (computed after capping)
        self.income_cap_ = X.MonthlyIncome.quantile(self.income_cap_quantile)
        income = X.MonthlyIncome.clip(upper=self.income_cap_)
        band = pd.cut(X.age, AGE_BINS, labels=AGE_LABELS)
        self.income_by_age_ = income.groupby(band, observed=True).median()
        self.income_overall_median_ = income.median()

        self.dependents_fill_ = X.NumberOfDependents.median()
        return self

    def transform(self, X):
        df = X.copy()

        code_mask = (df[PAST_DUE_COLS] >= self.code_threshold).any(axis=1)
        df["past_due_code_flag"] = code_mask.astype(int)
        for c in PAST_DUE_COLS:
            df.loc[df[c] >= self.code_threshold, c] = self.past_due_max_[c]

        df[UTIL] = df[UTIL].clip(upper=self.util_cap)

        unreliable = df.MonthlyIncome.isna() | (df.DebtRatio > self.dr_max_plausible)
        df["debt_ratio_unreliable_flag"] = unreliable.astype(int)
        df.loc[unreliable, "DebtRatio"] = self.debt_ratio_fill_

        df["income_missing_flag"] = df.MonthlyIncome.isna().astype(int)
        df["MonthlyIncome"] = df.MonthlyIncome.clip(upper=self.income_cap_)
        band = pd.cut(df.age, AGE_BINS, labels=AGE_LABELS)
        fill = band.map(self.income_by_age_).astype(float).fillna(self.income_overall_median_)
        df["MonthlyIncome"] = df.MonthlyIncome.fillna(fill)

        df["dependents_missing_flag"] = df.NumberOfDependents.isna().astype(int)
        df["NumberOfDependents"] = df.NumberOfDependents.fillna(self.dependents_fill_)
        return df


UTIL_BAND_EDGES = [-0.001, 0.25, 0.5, 0.75, 1.0, float("inf")]
UTIL_BAND_LABELS = ["util_0_25", "util_25_50", "util_50_75", "util_75_100", "util_over_limit"]


def add_features(df):
    """Add the three Phase 3 features. Uses fixed rules only, so nothing is learned from the data."""
    out = df.copy()
    out["total_late_payments"] = out[PAST_DUE_COLS].sum(axis=1)
    out["ever_late"] = (out.total_late_payments > 0).astype(int)

    band = pd.cut(out[UTIL], UTIL_BAND_EDGES, labels=UTIL_BAND_LABELS)
    dummies = pd.get_dummies(band).astype(int)
    # util_0_25 is the reference group (all zeros), so it is dropped to avoid redundant columns
    out = pd.concat([out, dummies.drop(columns="util_0_25")], axis=1)
    return out
