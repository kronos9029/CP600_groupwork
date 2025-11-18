from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns derived from Dataset (sensitive vs non-sensitive)
SENSITIVE_COLUMNS = {
    "education",
    "education_num",
    "marital_status",
    "relationship",
    "race",
    "sex",
    "native_country",
}

NON_SENSITIVE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "occupation",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

TARGET_COLUMN = "income"

# will set from user input when program start
DEFAULT_SAMPLE_SIZE = 0

AGE_BIN_EDGES = [16, 25, 35, 50, 65, 80, 100]
HOURS_BIN_EDGES = [0, 30, 40, 50, 60, 100]
GAIN_BIN_EDGES = [-1, 0, 5000, 15000, 50000, 100000]
LOSS_BIN_EDGES = [-1, 0, 2000, 4000, 10000]

DIV_FINITE_FEATURES = [
    "age",
    "fnlwgt",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

COMPLEXITY_INFO = {
    "end_to_end": "O(n*d)",  # n rows, d non-sensitive features
    "discretization": "O(n*d)",
    "grouping": "O(n)",
    "sampling": "O(m*d)",
    "space": "O(n + m)",
}


# i log pipeline stage here so friend can follow
def log_stage(stage: str, **stats: object) -> None:
    """Print high-level stage markers with optional key/value stats."""
    
    if stats:
        parts = []
        for key, value in stats.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        detail = ", ".join(parts)
        print(f"[Stage] {stage}: {detail}")
    else:
        print(f"[Stage] {stage}")


@dataclass
class SamplingReport:
    sample_size: int
    total_records: int
    strata_kept: int
    allocation_summary: pd.DataFrame
    distribution_drift: pd.DataFrame
    metrics_full: dict[str, float]
    metrics_sample: dict[str, float]
    classification_summary: str
    used_full_training_set: bool = False
    membership_mi: float = float("nan")


@dataclass
class EvaluationArtifacts:
    y_test: pd.Series
    proba_full: np.ndarray
    proba_sample: np.ndarray
    pred_full: np.ndarray
    pred_sample: np.ndarray
    classes: np.ndarray


# load adult csv and clean columns
def load_dataset(path: Path) -> pd.DataFrame:
    """Load Adult CSV and normalize column names/values."""
    df = pd.read_csv(path, skipinitialspace=True, na_values="?", encoding="utf-8")
    df.columns = [
        col.strip()
        .lower()
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
        for col in df.columns
    ]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.replace("?", pd.NA)
    return df


# ask user sample size and clamp to limit
def prompt_sample_size(max_entries: int) -> int:
    """Read sample size from stdin and clamp to [0, max_entries]."""
    try:
        raw = input(
            f"Enter sample size: (Note: sample size should be between 0 - {max_entries}) "
        )
        value = int(raw)
    except Exception:
        return max_entries
    if value < 0 or value > max_entries:
        return max_entries
    return value


# fill missing value simple
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values with simple, privacy-preserving defaults."""
    result = df.copy()
    for col in result.columns:
        if col == TARGET_COLUMN:
            continue
        if result[col].dtype == "O":
            result[col] = result[col].fillna("Unknown")
        else:
            median = result[col].median()
            result[col] = result[col].fillna(median)
    return result


# make bins for nums so next stratify easy
def add_discretized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add binned views of numeric features used for stratification."""
    result = df.copy()
    result["age_bin"] = pd.cut(
        result["age"],
        bins=AGE_BIN_EDGES,
        labels=[
            "17-25",
            "26-35",
            "36-50",
            "51-65",
            "66-80",
            "81+",
        ],
        include_lowest=True,
        right=True,
    )
    result["hours_bin"] = pd.cut(
        result["hours_per_week"],
        bins=HOURS_BIN_EDGES,
        labels=["0-30", "31-40", "41-50", "51-60", "61+"],
        include_lowest=True,
        right=True,
    )
    result["gain_bin"] = pd.cut(
        result["capital_gain"],
        bins=GAIN_BIN_EDGES,
        labels=["0", "1-5k", "5k-15k", "15k-50k", "50k+"],
        include_lowest=True,
        right=True,
    )
    result["loss_bin"] = pd.cut(
        result["capital_loss"],
        bins=LOSS_BIN_EDGES,
        labels=["0", "1-2k", "2k-4k", "4k+"],
        include_lowest=True,
        right=True,
    )
    result["workclass"] = result["workclass"].fillna("Unknown")
    result["occupation"] = result["occupation"].fillna("Unknown")
    return result


# make dataframe to markdown string even if library missing
def dataframe_to_markdown(df: pd.DataFrame, **kwargs) -> str:
    """Render DataFrame to markdown, falling back to plain text if needed."""
    try:
        return df.to_markdown(**kwargs)
    except ImportError:
        return df.to_string(index=kwargs.get("index", True))


# normalize matrix columns to compare distance fair
def _normalize_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standardize columns and replace NaNs for diversity computation."""
    col_means = np.nanmean(matrix, axis=0)
    inds = np.where(np.isnan(matrix))
    matrix[inds] = np.take(col_means, inds[1])
    std = np.nanstd(matrix, axis=0)
    std[std == 0.0] = 1.0
    matrix = (matrix - col_means) / std
    return np.nan_to_num(matrix)


# pick spread out rows inside one group using greedy
def _diversity_sample(
    group: pd.DataFrame,
    take: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Select a subset that maximizes pairwise distances within a stratum."""
    matrix = group[DIV_FINITE_FEATURES].to_numpy(dtype=float, copy=True)
    matrix = _normalize_matrix(matrix)
    if take <= 0:
        return group.iloc[[]]
    initial = int(rng.integers(0, len(group)))
    chosen = [initial]
    # Initialize squared distances from the first pivot
    diff = matrix - matrix[initial]
    min_dist = np.sum(diff * diff, axis=1)
    min_dist[initial] = -np.inf
    while len(chosen) < take:
        next_idx = int(np.argmax(min_dist))
        if min_dist[next_idx] <= 0:
            # All points identical, fall back to random sampling
            remaining = group.drop(index=group.index[chosen])
            needed = take - len(chosen)
            random_rows = remaining.sample(
                n=needed,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            return pd.concat([group.iloc[chosen], random_rows])
        chosen.append(next_idx)
        diff = matrix - matrix[next_idx]
        new_dist = np.sum(diff * diff, axis=1)
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[chosen] = -np.inf
    return group.iloc[chosen]


# do proportional stratified pick then keep diverse inside each stratum
def stratified_diversity_sample(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    We pick a representative sample from df using stratified proportional allocation
    followed by intra-stratum diversity maximization.
    Returns the sampled rows and a summary of the allocation.
    """
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    sample_size = min(sample_size, len(df))
    rng = np.random.default_rng(random_state)
    working = add_discretized_columns(df)
    strata_cols = ["age_bin", "workclass", "occupation"]
    grouped = working.groupby(strata_cols, dropna=False, observed=False)
    group_sizes = grouped.size()
    allocations = (group_sizes / len(working)) * sample_size
    base = np.floor(allocations).astype(int)
    remainders = allocations - base
    assigned = int(base.sum())
    leftover = sample_size - assigned
    if leftover > 0:
        for key in remainders.sort_values(ascending=False).index:
            base.loc[key] += 1
            leftover -= 1
            if leftover == 0:
                break
    samples: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for key, group in grouped:
        take = int(base.get(key, 0))
        group_len = len(group)
        summary_rows.append(
            {
                "stratum": key,
                "group_size": group_len,
                "allocated_sample": min(take, group_len),
            }
        )
        if take <= 0:
            continue
        take = min(take, group_len)
        if take == group_len:
            subset = group
        elif take == 1 or group_len < 3:
            subset = group.sample(
                n=take,
                random_state=int(rng.integers(0, 1_000_000)),
            )
        else:
            subset = _diversity_sample(group, take, rng)
        samples.append(subset)
    if not samples:
        raise ValueError("Sampling produced no strata. Check input data.")
    sampled = pd.concat(samples).drop(columns=["age_bin", "hours_bin", "gain_bin", "loss_bin"])
    sampled = sampled.loc[~sampled.index.duplicated(keep="first")]
    if len(sampled) < sample_size:
        remaining = df.drop(index=sampled.index)
        needed = sample_size - len(sampled)
        fallback = remaining.sample(
            n=needed,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        sampled = pd.concat([sampled, fallback])
    sampled = sampled.sample(
        n=sample_size,
        random_state=int(rng.integers(0, 1_000_000)),
        replace=False,
    )
    allocation_summary = pd.DataFrame(summary_rows)
    return sampled, allocation_summary


# build preprocessing and logistic model for features
def build_pipeline(features: Sequence[str]) -> Pipeline:
    """Building a preprocessing + logistic regression pipeline."""
    categorical = [col for col in features if col not in DIV_FINITE_FEATURES]
    numeric = [col for col in features if col in DIV_FINITE_FEATURES]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    clf = LogisticRegression(
        max_iter=500,
        n_jobs=None,
        solver="lbfgs",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])


# train full vs sample model then measure metrics
def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[dict[str, float], dict[str, float], str, EvaluationArtifacts]:
    """Fit and compare classifiers trained on the full vs sampled data."""
    pipeline = build_pipeline(feature_cols)
    full_model = pipeline.fit(train_df[feature_cols], train_df[TARGET_COLUMN])
    sample_model = clone(pipeline).fit(sample_df[feature_cols], sample_df[TARGET_COLUMN])
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]
    proba_full = full_model.predict_proba(X_test)[:, 1]
    proba_sample = sample_model.predict_proba(X_test)[:, 1]
    y_pred_full = (proba_full >= 0.5).astype(int)
    y_pred_sample = (proba_sample >= 0.5).astype(int)
    y_test_binary = (y_test == ">50K").astype(int)
    metrics_full = {
        "accuracy": accuracy_score(y_test, full_model.predict(X_test)),
        "precision": precision_score(y_test_binary, y_pred_full),
        "recall": recall_score(y_test_binary, y_pred_full),
        "f1": f1_score(y_test_binary, y_pred_full),
        "roc_auc": roc_auc_score(y_test_binary, proba_full),
    }
    metrics_sample = {
        "accuracy": accuracy_score(y_test, sample_model.predict(X_test)),
        "precision": precision_score(y_test_binary, y_pred_sample),
        "recall": recall_score(y_test_binary, y_pred_sample),
        "f1": f1_score(y_test_binary, y_pred_sample),
        "roc_auc": roc_auc_score(y_test_binary, proba_sample),
    }
    class_report = classification_report(
        y_test,
        sample_model.predict(X_test),
        digits=3,
    )
    artifacts = EvaluationArtifacts(
        y_test=y_test.reset_index(drop=True),
        proba_full=proba_full,
        proba_sample=proba_sample,
        pred_full=full_model.predict(X_test),
        pred_sample=sample_model.predict(X_test),
        classes=sample_model.classes_,
    )
    return metrics_full, metrics_sample, class_report, artifacts


# check how sample distribution drift from full per feature
def feature_distribution_drift(
    full_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute total variation distance for each non-sensitive column."""
    rows = []
    for col in NON_SENSITIVE_COLUMNS:
        full_series = full_df[col]
        sample_series = sample_df[col]
        if pd.api.types.is_numeric_dtype(full_series):
            bins = 10
            full_hist, bin_edges = np.histogram(full_series, bins=bins, density=True)
            sample_hist, _ = np.histogram(sample_series, bins=bin_edges, density=True)
            # Normalize histograms to proper distributions
            full_hist = full_hist / full_hist.sum()
            sample_hist = sample_hist / sample_hist.sum()
        else:
            full_counts = full_series.value_counts(normalize=True)
            sample_counts = sample_series.value_counts(normalize=True)
            union = full_counts.index.union(sample_counts.index)
            full_hist = full_counts.reindex(union, fill_value=0).to_numpy()
            sample_hist = sample_counts.reindex(union, fill_value=0).to_numpy()
        tv_distance = 0.5 * np.abs(full_hist - sample_hist).sum()
        rows.append({"feature": col, "total_variation": tv_distance})
    return pd.DataFrame(rows).sort_values("total_variation")


# bin fnlwgt value so MI calc not explode
def _add_fnlwgt_bins(df: pd.DataFrame, bins: int = 10) -> pd.Series:
    """Create discrete bins for fnlwgt to enable MI calculation."""
    try:
        return pd.qcut(df["fnlwgt"], q=bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.cut(df["fnlwgt"], bins=bins, labels=False, include_lowest=True)


# estimate membership mutual info using discretize columns
def _estimate_membership_mi(
    df: pd.DataFrame,
    sampled_indices: Sequence[int],
) -> float:
    """Estimate I(X; I) where I indicates membership in the sampled subset."""
    discrete = add_discretized_columns(df)
    discrete["fnlwgt_bin"] = _add_fnlwgt_bins(discrete).astype("Int64")
    discrete["in_sample"] = 0
    intersect = discrete.index.intersection(sampled_indices)
    discrete.loc[intersect, "in_sample"] = 1
    mi_columns = [
        "age_bin",
        "workclass",
        "occupation",
        "fnlwgt_bin",
        "hours_bin",
        "gain_bin",
        "loss_bin",
    ]
    for column in mi_columns:
            discrete[column] = discrete[column].astype(str)
    return _mutual_information_discrete(discrete, mi_columns, "in_sample")


# compute MI between feature combo and membership flag
def _mutual_information_discrete(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    membership_col: str,
) -> float:
    """Compute mutual information between feature tuple and membership flag."""
    total = len(df)
    if total == 0:
        return 0.0
    joint = (
        df.groupby(list(feature_cols) + [membership_col], dropna=False)
        .size()
        .div(total)
    )
    p_x = df.groupby(feature_cols, dropna=False).size().div(total)
    p_i = df.groupby(membership_col, dropna=False).size().div(total)
    mi = 0.0
    for key, p_xi in joint.items():
        x_key = key[:-1]
        i_value = key[-1]
        p_x_val = p_x[x_key]
        p_i_val = p_i[i_value]
        if p_xi == 0 or p_x_val == 0 or p_i_val == 0:
            continue
        ratio = p_xi / (p_x_val * p_i_val)
        if ratio <= 0:
            continue
        mi += p_xi * math.log2(ratio)
    return float(mi)


# swap rows inside strata to push MI lower step by step
def mi_guided_refinement(
    full_df: pd.DataFrame,
    initial_sample: pd.DataFrame,
    random_state: int,
    max_iter: int = 30,
) -> tuple[pd.DataFrame, float, list[tuple[int, float]]]:
    """Iteratively swap records within strata to directly minimize membership MI."""
    rng = np.random.default_rng(random_state)
    if initial_sample.empty:
        return initial_sample, 0.0, []
    discretized = add_discretized_columns(full_df)
    strata_cols = ["age_bin", "workclass", "occupation"]
    discretized["stratum_key"] = (
        discretized[strata_cols].astype(str).agg("|".join, axis=1)
    )
    stratum_groups = {
        key: set(group.index.tolist())
        for key, group in discretized.groupby("stratum_key", dropna=False)
    }
    selected: set[int] = set(map(int, initial_sample.index.tolist()))
    best_indices = list(selected)
    best_mi = _estimate_membership_mi(full_df, best_indices)
    history: list[tuple[int, float]] = [(0, best_mi)]
    for step in range(1, max_iter + 1):
        eligible = [
            key
            for key, idxs in stratum_groups.items()
            if len(selected & idxs) > 0 and len(idxs - selected) > 0
        ]
        if not eligible:
            break
        key = eligible[int(rng.integers(0, len(eligible)))]
        stratum_selected = list(selected & stratum_groups[key])
        stratum_candidates = list(stratum_groups[key] - selected)
        if not stratum_selected or not stratum_candidates:
            continue
        to_remove = stratum_selected[int(rng.integers(0, len(stratum_selected)))]
        to_add = stratum_candidates[int(rng.integers(0, len(stratum_candidates)))]
        trial_selected = set(selected)
        trial_selected.remove(to_remove)
        trial_selected.add(to_add)
        trial_indices = list(trial_selected)
        trial_mi = _estimate_membership_mi(full_df, trial_indices)
        if trial_mi + 1e-6 < best_mi:
            selected = trial_selected
            best_indices = trial_indices
            best_mi = trial_mi
        history.append((step, best_mi))
    refined_df = full_df.loc[best_indices].copy()
    refined_df = refined_df.sample(
        frac=1,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    return refined_df, best_mi, history


# draw histogram compare numeric column full vs sample
def _plot_numeric_distribution(
    full_series: pd.Series,
    sample_series: pd.Series,
    column: str,
    output_dir: Path,
) -> Path:
    num_bins = min(20, max(5, full_series.nunique()))
    fig, ax = plt.subplots(figsize=(8, 4))
    common_range = (
        min(full_series.min(), sample_series.min()),
        max(full_series.max(), sample_series.max()),
    )
    bins = np.linspace(common_range[0], common_range[1], num_bins)
    ax.hist(
        full_series,
        bins=bins,
        alpha=0.5,
        density=True,
        label="Original",
        color="#1f77b4",
        edgecolor="#1f77b4",
    )
    ax.hist(
        sample_series,
        bins=bins,
        alpha=0.5,
        density=True,
        label="Sample",
        color="#ff7f0e",
        edgecolor="#ff7f0e",
    )
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.set_title(f"{column} distribution")
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{column}_distribution.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# For data visualization
# To be added in the report
# draw bar chart compare category share full vs sample
def _plot_categorical_distribution(
    full_series: pd.Series,
    sample_series: pd.Series,
    column: str,
    output_dir: Path,
) -> Path:
    full_counts = full_series.value_counts(normalize=True)
    sample_counts = sample_series.value_counts(normalize=True)
    categories = full_counts.index.union(sample_counts.index)
    # Preserve frequency order while appending unseen categories at the end
    for cat in sample_counts.index:
        if cat not in categories:
            categories = categories.append(pd.Index([cat]))
    categories = list(categories)
    full_values = full_counts.reindex(categories, fill_value=0).to_numpy()
    sample_values = sample_counts.reindex(categories, fill_value=0).to_numpy()
    positions = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.5), 4))
    ax.bar(
        positions - width / 2,
        full_values,
        width,
        label="Original",
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
    )
    ax.bar(
        positions + width / 2,
        sample_values,
        width,
        label="Sample",
        color="#ff7f0e",
        alpha=0.85,
        edgecolor="black",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(cat) for cat in categories], rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title(f"{column} distribution")
    max_height = max(
        float(full_values.max()) if len(full_values) else 0.0,
        float(sample_values.max()) if len(sample_values) else 0.0,
    )
    ax.set_ylim(0, max(0.05, max_height * 1.15))
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{column}_distribution.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# create set of compare plots for chosen columns
def create_distribution_plots(
    full_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    columns: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    """Generate histogram/bar comparisons for selected columns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    for column in columns:
        if column not in full_df.columns or column not in sample_df.columns:
            continue
        full_series = full_df[column].dropna()
        sample_series = sample_df[column].dropna()
        if pd.api.types.is_numeric_dtype(full_series):
            path = _plot_numeric_distribution(full_series, sample_series, column, output_dir)
        else:
            path = _plot_categorical_distribution(full_series, sample_series, column, output_dir)
        generated_paths.append(path)
    return generated_paths


# save ROC and confusion matrix to see model behavior
def create_model_diagnostics(
    artifacts: EvaluationArtifacts,
    metrics_full: dict[str, float],
    metrics_sample: dict[str, float],
    output_dir: Path,
) -> list[Path]:
    """Generate ROC curves and confusion matrix visuals."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    y_true_binary = (artifacts.y_test == ">50K").astype(int)
    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 4))
    curve_specs = [
        (
            "Full Data",
            artifacts.proba_full,
            metrics_full.get("roc_auc", float("nan")),
            metrics_full.get("accuracy", float("nan")),
            "#1f77b4",
        ),
        (
            "Sample",
            artifacts.proba_sample,
            metrics_sample.get("roc_auc", float("nan")),
            metrics_sample.get("accuracy", float("nan")),
            "#ff7f0e",
        ),
    ]
    for label, proba, auc_val, acc_val, color in curve_specs:
        fpr, tpr, _ = roc_curve(y_true_binary, proba)
        ax.plot(
            fpr,
            tpr,
            label=f"{label} (AUC={auc_val:.3f}, ACC={acc_val:.3f})",
            color=color,
        )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = output_dir / "roc_curves.png"
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    generated.append(roc_path)
    # Confusion matrix (sample model)
    cm = confusion_matrix(
        artifacts.y_test,
        artifacts.pred_sample,
        labels=artifacts.classes,
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=artifacts.classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix (Sample Model)")
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    generated.append(cm_path)
    return generated


# plot how MI change across refinement loop
def plot_mi_convergence(history: Sequence[tuple[int, float]], output_path: Path) -> Path:
    """Plot MI value across refinement iterations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return output_path
    iterations, values = zip(*history)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iterations, values, marker="o", color="#2ca02c")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Membership MI (bits)")
    ax.set_title("MI-Guided Refinement Progress")
    ax.grid(True, linestyle="--", alpha=0.4)
    start = int(min(iterations))
    stop = int(max(iterations)) + 1
    step = max(1, len(iterations) // 10)
    ax.set_xticks(range(start, stop, step))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


# main pipeline wire: load, sample, train, evaluate
def run_pipeline(
    data_path: Path,
    sample_size: Optional[int],
    random_state: int,
) -> tuple[
    SamplingReport,
    pd.DataFrame,
    pd.DataFrame,
    EvaluationArtifacts,
    list[tuple[int, float]],
]:
    log_stage("Loading dataset", path=data_path)
    df = load_dataset(data_path)
    df = fill_missing_values(df)
    log_stage("Dataset ready", total_records=len(df))
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")
    features = [col for col in NON_SENSITIVE_COLUMNS if col in df.columns]
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )
    log_stage("Split dataset", train=len(train_df), test=len(test_df))
    if sample_size is None or sample_size <= 0:
        effective_sample_size = len(train_df)
    else:
        effective_sample_size = min(sample_size, len(train_df))
    sampled_df, allocation_summary = stratified_diversity_sample(
        train_df,
        sample_size=effective_sample_size,
        random_state=random_state,
    )
    strata_kept = allocation_summary["allocated_sample"].astype(bool).sum()
    log_stage(
        "Sampling completed",
        sample_size=len(sampled_df),
        strata=strata_kept,
        coverage=len(sampled_df) / len(train_df),
    )
    sampled_df, membership_mi, mi_history = mi_guided_refinement(
        train_df,
        sampled_df,
        random_state=random_state,
    )
    log_stage("MI-guided refinement", bits=membership_mi)
    metrics_full, metrics_sample, class_report, eval_artifacts = evaluate_models(
        train_df,
        test_df,
        sampled_df,
        features,
    )
    log_stage(
        "Evaluation complete",
        sample_accuracy=metrics_sample["accuracy"],
        sample_roc_auc=metrics_sample["roc_auc"],
    )
    drift = feature_distribution_drift(train_df, sampled_df)
    log_stage(
        "Distribution drift computed",
        max_drift=float(drift["total_variation"].max()),
        min_drift=float(drift["total_variation"].min()),
    )
    log_stage(
        "Complexity reference",
        overall=COMPLEXITY_INFO["end_to_end"],
        space=COMPLEXITY_INFO["space"],
    )
    report = SamplingReport(
        sample_size=len(sampled_df),
        total_records=len(df),
        strata_kept=strata_kept,
        allocation_summary=allocation_summary,
        distribution_drift=drift,
        metrics_full=metrics_full,
        metrics_sample=metrics_sample,
        classification_summary=class_report,
        used_full_training_set=(effective_sample_size == len(train_df)),
        membership_mi=membership_mi,
    )
    return report, sampled_df, train_df, eval_artifacts, mi_history


# write markdown report of sampling result
def write_report(report: SamplingReport, output_path: Path) -> None:
    """Persist a human-readable summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_note = " (full training set)" if report.used_full_training_set else ""
    lines = [
        "# Sampling Report",
        "",
        f"- Total records: {report.total_records}",
        f"- Sample size: {report.sample_size}{sample_note}",
        f"- Membership MI (bits): {report.membership_mi:.4f}",
        f"- Non-empty strata: {report.strata_kept}",
        "",
        "## Model Performance (Full vs Sample)",
    ]
    header = "| Metric | Full Data | Sample |\n| --- | --- | --- |"
    lines.append(header)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        full_val = report.metrics_full.get(metric, float("nan"))
        sample_val = report.metrics_sample.get(metric, float("nan"))
        lines.append(f"| {metric} | {full_val:.3f} | {sample_val:.3f} |")
    lines.extend(
        [
            "",
            "## Distribution Drift (Total Variation Distance)",
            dataframe_to_markdown(report.distribution_drift, index=False),
            "",
            "## Strata Allocation (top 15 by size)",
            dataframe_to_markdown(
                report.allocation_summary.sort_values("group_size", ascending=False)
                .head(15),
                index=False,
            ),
            "",
            "## Complexity Snapshot",
            "| Component | Complexity |",
            "| --- | --- |",
            f"| Overall | {COMPLEXITY_INFO['end_to_end']} |",
            f"| Discretization | {COMPLEXITY_INFO['discretization']} |",
            f"| Grouping | {COMPLEXITY_INFO['grouping']} |",
            f"| Sampling | {COMPLEXITY_INFO['sampling']} |",
            f"| Space | {COMPLEXITY_INFO['space']} |",
            "",
            "## Sample Classification Report",
            "```\n" + report.classification_summary + "\n```",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


# parse user cli options for pipeline run
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Representative sampling + evaluation for Adult dataset.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("adult.csv"),
        help="Path to adult.csv dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of records to sample (m). If omitted, will ask you on start.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used across sampling/modeling.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("sampling_report.md"),
        help="Where to save the markdown report.",
    )
    parser.add_argument(
        "--export-sample",
        type=Path,
        default=Path("sampled_adult.csv"),
        help="Optional CSV path to persist the sampled records.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for distribution comparison figures.",
    )
    return parser.parse_args()


# program entry run pipeline and export stuff
def main() -> None:
    args = parse_args()
    df_for_bounds = load_dataset(args.data_path)
    max_entries = len(df_for_bounds)
    global DEFAULT_SAMPLE_SIZE
    DEFAULT_SAMPLE_SIZE = prompt_sample_size(max_entries)
    chosen_sample_size = args.sample_size if args.sample_size is not None else DEFAULT_SAMPLE_SIZE
    if chosen_sample_size < 0 or chosen_sample_size > max_entries:
        chosen_sample_size = max_entries
    (
        report,
        sampled_df,
        reference_df,
        eval_artifacts,
        mi_history,
    ) = run_pipeline(
        data_path=args.data_path,
        sample_size=chosen_sample_size,
        random_state=args.random_state,
    )
    write_report(report, args.report_path)
    log_stage("Report written", path=args.report_path)
    if args.export_sample:
        args.export_sample.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(args.export_sample, index=False)
        log_stage("Sample CSV exported", path=args.export_sample, rows=len(sampled_df))
    generated = create_distribution_plots(
        reference_df,
        sampled_df,
        columns=[
            "age",
            "workclass",
            "occupation",
            "hours_per_week",
            "capital_gain",
            "capital_loss",
        ],
        output_dir=args.figures_dir,
    )
    if generated:
        log_stage("Figures generated", count=len(generated), directory=args.figures_dir)
    diag_paths = create_model_diagnostics(
        eval_artifacts,
        report.metrics_full,
        report.metrics_sample,
        output_dir=args.figures_dir,
    )
    if diag_paths:
        log_stage("Model diagnostics saved", count=len(diag_paths))
    mi_path = plot_mi_convergence(
        mi_history,
        output_path=args.figures_dir / "mi_convergence.png",
    )
    if mi_history:
        log_stage("MI convergence plot written", path=mi_path)


if __name__ == "__main__":
    main()
