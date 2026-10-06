from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hp\Industrial-Analytics"
    r"\Mitsubishi-ME-AD-Predictive-Maintenance"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_cycle_manifest.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_cycle_structure_quality.csv"
)


# =============================================================================
# 2. LOAD MASTER MANIFEST
# =============================================================================

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(
        f"Master cycle manifest not found:\n{MANIFEST_FILE}"
    )

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={"operation_code": str},
)

print("=" * 82)
print("ME-AD DATASET — CYCLE STRUCTURE & QUALITY")
print("=" * 82)

print(f"\nManifest rows : {len(manifest):,}")
print(
    f"Operations    : "
    f"{manifest['operation_code'].nunique():,}"
)


# =============================================================================
# 3. BASIC MANIFEST QUALITY
# =============================================================================

duplicate_keys = manifest.duplicated(
    subset=["operation_code", "cycle_index"]
).sum()

missing_values = manifest.isna().sum().sum()

print("\nMANIFEST QUALITY")
print("-" * 82)
print(f"Duplicate operation/cycle keys : {duplicate_keys:,}")
print(f"Missing manifest values         : {missing_values:,}")


# =============================================================================
# 4. PER-OPERATION FILE-SIZE STATISTICS
# =============================================================================

operation_summary = (
    manifest
    .groupby("operation_code")
    .agg(
        cycles=("cycle_index", "count"),
        min_index=("cycle_index", "min"),
        max_index=("cycle_index", "max"),
        mean_pkl_kb=("pkl_bytes", lambda x: x.mean() / 1024),
        std_pkl_kb=("pkl_bytes", lambda x: x.std() / 1024),
        min_pkl_kb=("pkl_bytes", lambda x: x.min() / 1024),
        median_pkl_kb=("pkl_bytes", lambda x: x.median() / 1024),
        max_pkl_kb=("pkl_bytes", lambda x: x.max() / 1024),
    )
    .reset_index()
)


# =============================================================================
# 5. WITHIN-OPERATION ROBUST FILE-SIZE OUTLIERS
# =============================================================================
# Calculate robust file-size scores within each operation.
#
# MAD = Median Absolute Deviation.
# This is used only as a structural screening measure.
# A size outlier does NOT imply a robot fault.

manifest["operation_median_pkl_bytes"] = (
    manifest
    .groupby("operation_code")["pkl_bytes"]
    .transform("median")
)

manifest["absolute_size_deviation"] = (
    manifest["pkl_bytes"]
    - manifest["operation_median_pkl_bytes"]
).abs()

manifest["operation_mad_pkl_bytes"] = (
    manifest
    .groupby("operation_code")["absolute_size_deviation"]
    .transform("median")
)

manifest["robust_size_score"] = np.where(
    manifest["operation_mad_pkl_bytes"] > 0,
    0.6745
    * (
        manifest["pkl_bytes"]
        - manifest["operation_median_pkl_bytes"]
    )
    / manifest["operation_mad_pkl_bytes"],
    0.0,
)

manifest["size_outlier"] = (
    manifest["robust_size_score"].abs() > 3.5
)

manifest = manifest.drop(
    columns=["absolute_size_deviation"]
)


# =============================================================================
# 6. FILE SIZE BY HEALTH REGION
# =============================================================================

region_summary = (
    manifest
    .groupby(
        [
            "operation_code",
            "benchmark_health_region",
        ]
    )
    .agg(
        cycles=("cycle_index", "count"),
        mean_pkl_kb=("pkl_bytes", lambda x: x.mean() / 1024),
        median_pkl_kb=("pkl_bytes", lambda x: x.median() / 1024),
        min_pkl_kb=("pkl_bytes", lambda x: x.min() / 1024),
        max_pkl_kb=("pkl_bytes", lambda x: x.max() / 1024),
        size_outliers=("size_outlier", "sum"),
    )
    .reset_index()
)


# =============================================================================
# 7. CORRELATION: FILE SIZE VS SEQUENCE POSITION
# =============================================================================
# This does not prove degradation. It simply checks whether retained dataset
# size changes systematically from the beginning to the end of each sequence.

correlation_records = []

for operation_code, group in manifest.groupby("operation_code"):

    pearson = group[
        ["sequence_position", "pkl_bytes"]
    ].corr(method="pearson").iloc[0, 1]

    spearman = group[
        ["sequence_position", "pkl_bytes"]
    ].corr(method="spearman").iloc[0, 1]

    correlation_records.append(
        {
            "operation_code": operation_code,
            "pearson_size_vs_sequence": pearson,
            "spearman_size_vs_sequence": spearman,
        }
    )

correlation_summary = pd.DataFrame(
    correlation_records
)


# =============================================================================
# 8. EARLY VS LATE FILE-SIZE COMPARISON
# =============================================================================

comparison_records = []

for operation_code, group in manifest.groupby("operation_code"):

    early = group[
        group["benchmark_health_region"].isin(
            ["training_healthy", "test_healthy"]
        )
    ]["pkl_bytes"]

    late = group[
        group["benchmark_health_region"]
        == "faulty_evident"
    ]["pkl_bytes"]

    early_mean = early.mean()
    late_mean = late.mean()

    percent_change = (
        (late_mean - early_mean)
        / early_mean
        * 100
    )

    comparison_records.append(
        {
            "operation_code": operation_code,
            "early_healthy_mean_kb": early_mean / 1024,
            "faulty_evident_mean_kb": late_mean / 1024,
            "percent_change": percent_change,
        }
    )

early_late_summary = pd.DataFrame(
    comparison_records
)


# =============================================================================
# 9. DISPLAY RESULTS
# =============================================================================

print("\nPKL FILE-SIZE DISTRIBUTION BY OPERATION")
print("-" * 82)

print(
    operation_summary.round(2).to_string(
        index=False
    )
)


print("\nSIZE OUTLIERS BY OPERATION")
print("-" * 82)

outlier_counts = (
    manifest
    .groupby("operation_code")["size_outlier"]
    .sum()
)

print(outlier_counts.to_string())


print("\nFILE SIZE VS SEQUENCE POSITION")
print("-" * 82)

print(
    correlation_summary.round(4).to_string(
        index=False
    )
)


print("\nEARLY HEALTHY VS EVIDENT FAULTY FILE SIZE")
print("-" * 82)

print(
    early_late_summary.round(2).to_string(
        index=False
    )
)


print("\nHEALTH-REGION FILE-SIZE SUMMARY")
print("-" * 82)

print(
    region_summary.round(2).to_string(
        index=False
    )
)


# =============================================================================
# 10. MOST EXTREME WITHIN-OPERATION SIZE OBSERVATIONS
# =============================================================================

extreme = (
    manifest
    .assign(
        absolute_robust_score=lambda x:
        x["robust_size_score"].abs()
    )
    .sort_values(
        "absolute_robust_score",
        ascending=False,
    )
    .head(20)
)

print("\nTOP 20 EXTREME CYCLE FILE SIZES")
print("-" * 82)

print(
    extreme[
        [
            "operation_code",
            "cycle_index",
            "benchmark_health_region",
            "pkl_bytes",
            "robust_size_score",
            "sequence_position",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 11. SAVE ENRICHED MANIFEST
# =============================================================================

manifest.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nEnriched cycle-quality table saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 82)
print("CYCLE STRUCTURE & QUALITY COMPLETE")
print("=" * 82)