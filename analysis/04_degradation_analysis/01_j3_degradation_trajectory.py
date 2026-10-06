from pathlib import Path
from zipfile import ZipFile
import io
import sys
import types

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hp\Industrial-Analytics"
    r"\Mitsubishi-ME-AD-Predictive-Maintenance"
)

ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "me_ad"
    / "ME-AD.zip"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_cycle_structure_quality.csv"
)

HEALTHY_FEATURE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_healthy_cycle_features.csv"
)

OUTPUT_CYCLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_j3_degradation_cycle_features.csv"
)

OUTPUT_BINS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_j3_degradation_bins.csv"
)

SAMPLING_INTERVAL = 0.003555


# =============================================================================
# 2. PICKLE COMPATIBILITY
# =============================================================================

numeric_module = types.ModuleType(
    "pandas.core.indexes.numeric"
)

numeric_module.Int64Index = pd.Index
numeric_module.UInt64Index = pd.Index
numeric_module.Float64Index = pd.Index

sys.modules[
    "pandas.core.indexes.numeric"
] = numeric_module


# =============================================================================
# 3. HELPERS
# =============================================================================

def rms(values):
    values = np.asarray(values)

    if len(values) == 0:
        return np.nan

    return np.sqrt(
        np.mean(values ** 2)
    )


def robust_center_scale(values):

    values = pd.Series(values).dropna()

    median = values.median()

    mad = np.median(
        np.abs(values - median)
    )

    # Convert MAD to robust sigma estimate
    robust_sigma = 1.4826 * mad

    return median, robust_sigma


# =============================================================================
# 4. LOAD METADATA
# =============================================================================

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={"operation_code": str},
)

healthy_features = pd.read_csv(
    HEALTHY_FEATURE_FILE,
    dtype={"operation_code": str},
)

print("=" * 104)
print("ME-AD PHASE 04.1 — J3 PROGRESSIVE DEGRADATION TRAJECTORY")
print("=" * 104)

print(
    f"\nManifest cycles : {len(manifest):,}"
)

print(
    f"Operations      : "
    f"{manifest['operation_code'].nunique()}"
)


# =============================================================================
# 5. BUILD OPERATION-SPECIFIC HEALTHY REFERENCES
# =============================================================================

feature_names = [
    "j3_q_range",
    "j3_dq_rms",
    "j3_ddq_rms",
    "j3_tau_abs_mean",
    "j3_tau_rms",
    "j3_tau_peak",
    "j3_tau_std",
]

reference_records = []

for operation_code, group in healthy_features.groupby(
    "operation_code"
):

    record = {
        "operation_code":
            operation_code,
    }

    for feature in feature_names:

        center, scale = robust_center_scale(
            group[feature]
        )

        record[
            f"{feature}_healthy_median"
        ] = center

        record[
            f"{feature}_healthy_sigma"
        ] = scale

    # Healthy structural duration reference
    duration_median = (
        group["duration_seconds"].median()
    )

    duration_mad = np.median(
        np.abs(
            group["duration_seconds"]
            - duration_median
        )
    )

    record[
        "healthy_duration_median"
    ] = duration_median

    record[
        "healthy_duration_mad"
    ] = duration_mad

    reference_records.append(
        record
    )


references = pd.DataFrame(
    reference_records
)


# =============================================================================
# 6. EXTRACT J3 FEATURES FROM EVERY CYCLE
# =============================================================================

records = []

with ZipFile(ZIP_FILE, "r") as archive:

    total = len(manifest)

    for counter, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):

        operation_code = row.operation_code
        cycle_index = int(row.cycle_index)

        member = (
            f"ME-AD/Pandas/{operation_code}/"
            f"cleaned_dataset_{cycle_index}.pkl"
        )

        raw_bytes = archive.read(member)

        df = pd.read_pickle(
            io.BytesIO(raw_bytes)
        )

        record = {
            "operation_code":
                operation_code,

            "cycle_index":
                cycle_index,

            "operation_family":
                row.operation_family,

            "commanded_speed_group":
                row.commanded_speed_group,

            "benchmark_health_region":
                row.benchmark_health_region,

            "sequence_position":
                row.sequence_position,

            "cycles_from_end":
                row.cycles_from_end,

            "rows":
                len(df),

            "duration_seconds":
                len(df) * SAMPLING_INTERVAL,
        }

        if len(df) == 0:

            for feature in feature_names:
                record[feature] = np.nan

        else:

            q = df["q_filt_3"].to_numpy()
            dq = df["dq_filt_3"].to_numpy()
            ddq = df["ddq_filt_3"].to_numpy()
            tau = df["tau_filt_3"].to_numpy()

            record["j3_q_range"] = np.ptp(q)
            record["j3_dq_rms"] = rms(dq)
            record["j3_ddq_rms"] = rms(ddq)

            record["j3_tau_abs_mean"] = (
                np.mean(
                    np.abs(tau)
                )
            )

            record["j3_tau_rms"] = rms(tau)

            record["j3_tau_peak"] = (
                np.max(
                    np.abs(tau)
                )
            )

            record["j3_tau_std"] = (
                np.std(tau)
            )

        records.append(record)

        if (
            counter % 2000 == 0
            or counter == total
        ):
            print(
                f"Processed "
                f"{counter:,}/{total:,} cycles"
            )


cycles = pd.DataFrame(
    records
)


# =============================================================================
# 7. ADD HEALTHY REFERENCES
# =============================================================================

cycles = cycles.merge(
    references,
    on="operation_code",
    how="left",
    validate="many_to_one",
)


# =============================================================================
# 8. STRUCTURAL QUALITY FLAG
# =============================================================================

# Duration filtering is intentionally broad.
# We want to remove grossly incomplete/prolonged records,
# not legitimate degradation.

cycles[
    "duration_ratio_to_healthy"
] = (
    cycles["duration_seconds"]
    /
    cycles["healthy_duration_median"]
)

cycles[
    "structural_quality"
] = "usable"

cycles.loc[
    cycles["rows"] == 0,
    "structural_quality",
] = "exclude_empty"

cycles.loc[
    (
        cycles["rows"] > 0
    )
    &
    (
        cycles[
            "duration_ratio_to_healthy"
        ] < 0.50
    ),
    "structural_quality",
] = "exclude_too_short"

cycles.loc[
    cycles[
        "duration_ratio_to_healthy"
    ] > 2.00,
    "structural_quality",
] = "exclude_too_long"


# =============================================================================
# 9. ROBUST DEPARTURE FROM HEALTHY BASELINE
# =============================================================================

for feature in feature_names:

    center_col = (
        f"{feature}_healthy_median"
    )

    scale_col = (
        f"{feature}_healthy_sigma"
    )

    z_col = (
        f"{feature}_robust_z"
    )

    numerator = (
        cycles[feature]
        -
        cycles[center_col]
    )

    denominator = cycles[
        scale_col
    ].replace(
        0,
        np.nan,
    )

    cycles[z_col] = (
        numerator
        /
        denominator
    )


# =============================================================================
# 10. SAVE CYCLE-LEVEL TRAJECTORY
# =============================================================================

cycles.to_csv(
    OUTPUT_CYCLES,
    index=False,
)


# =============================================================================
# 11. QUALITY SUMMARY
# =============================================================================

print(
    "\nSTRUCTURAL QUALITY CLASSIFICATION"
)
print("-" * 104)

quality_summary = (
    cycles[
        "structural_quality"
    ]
    .value_counts(
        dropna=False
    )
)

print(
    quality_summary.to_string()
)


# =============================================================================
# 12. CREATE SEQUENCE BINS
# =============================================================================

usable = cycles[
    cycles["structural_quality"]
    == "usable"
].copy()

# 20 equal-width bins across each operation's normalized sequence.
usable[
    "sequence_bin"
] = pd.cut(
    usable[
        "sequence_position"
    ],
    bins=np.linspace(
        0,
        1,
        21,
    ),
    include_lowest=True,
    labels=False,
)

usable[
    "sequence_bin"
] = (
    usable[
        "sequence_bin"
    ]
    + 1
)


# =============================================================================
# 13. SUMMARISE J3 DEGRADATION BY SEQUENCE BIN
# =============================================================================

bin_summary = (
    usable
    .groupby(
        [
            "operation_code",
            "sequence_bin",
        ],
        observed=True,
    )
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),

        median_sequence_position=(
            "sequence_position",
            "median",
        ),

        median_q_range_z=(
            "j3_q_range_robust_z",
            "median",
        ),

        median_dq_rms_z=(
            "j3_dq_rms_robust_z",
            "median",
        ),

        median_ddq_rms_z=(
            "j3_ddq_rms_robust_z",
            "median",
        ),

        median_tau_abs_z=(
            "j3_tau_abs_mean_robust_z",
            "median",
        ),

        median_tau_rms_z=(
            "j3_tau_rms_robust_z",
            "median",
        ),

        median_tau_peak_z=(
            "j3_tau_peak_robust_z",
            "median",
        ),

        median_tau_std_z=(
            "j3_tau_std_robust_z",
            "median",
        ),
    )
    .reset_index()
)

bin_summary.to_csv(
    OUTPUT_BINS,
    index=False,
)


# =============================================================================
# 14. EARLY VS MIDDLE VS LATE J3 DEPARTURE
# =============================================================================

def sequence_stage(position):

    if position <= 0.10:
        return "early_0_10pct"

    if position <= 0.50:
        return "middle_10_50pct"

    if position <= 0.90:
        return "late_50_90pct"

    return "end_90_100pct"


usable[
    "sequence_stage"
] = usable[
    "sequence_position"
].apply(
    sequence_stage
)


stage_summary = (
    usable
    .groupby(
        "sequence_stage"
    )
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),

        median_q_range_z=(
            "j3_q_range_robust_z",
            "median",
        ),

        median_dq_rms_z=(
            "j3_dq_rms_robust_z",
            "median",
        ),

        median_ddq_rms_z=(
            "j3_ddq_rms_robust_z",
            "median",
        ),

        median_tau_abs_z=(
            "j3_tau_abs_mean_robust_z",
            "median",
        ),

        median_tau_rms_z=(
            "j3_tau_rms_robust_z",
            "median",
        ),

        median_tau_peak_z=(
            "j3_tau_peak_robust_z",
            "median",
        ),

        median_tau_std_z=(
            "j3_tau_std_robust_z",
            "median",
        ),
    )
    .reindex(
        [
            "early_0_10pct",
            "middle_10_50pct",
            "late_50_90pct",
            "end_90_100pct",
        ]
    )
    .reset_index()
)


print(
    "\nJ3 ROBUST DEPARTURE FROM OPERATION-SPECIFIC HEALTHY BASELINE"
)
print("-" * 104)

print(
    stage_summary
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 15. OPERATION-LEVEL EARLY VS END COMPARISON
# =============================================================================

operation_stage = (
    usable[
        usable[
            "sequence_stage"
        ].isin(
            [
                "early_0_10pct",
                "end_90_100pct",
            ]
        )
    ]
    .groupby(
        [
            "operation_code",
            "sequence_stage",
        ]
    )
    .agg(
        median_tau_rms_z=(
            "j3_tau_rms_robust_z",
            "median",
        ),

        median_tau_abs_z=(
            "j3_tau_abs_mean_robust_z",
            "median",
        ),

        median_tau_peak_z=(
            "j3_tau_peak_robust_z",
            "median",
        ),

        median_dq_rms_z=(
            "j3_dq_rms_robust_z",
            "median",
        ),
    )
    .reset_index()
)


pivot = operation_stage.pivot(
    index="operation_code",
    columns="sequence_stage",
    values=[
        "median_tau_rms_z",
        "median_tau_abs_z",
        "median_tau_peak_z",
        "median_dq_rms_z",
    ],
)

pivot.columns = [
    f"{metric}_{stage}"
    for metric, stage
    in pivot.columns
]

pivot = pivot.reset_index()


print(
    "\nOPERATION-LEVEL J3 EARLY VS END COMPARISON"
)
print("-" * 104)

print(
    pivot
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 16. BENCHMARK REGION CHECK
# =============================================================================

benchmark_summary = (
    usable
    .groupby(
        "benchmark_health_region"
    )
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),

        median_tau_rms_z=(
            "j3_tau_rms_robust_z",
            "median",
        ),

        median_tau_abs_z=(
            "j3_tau_abs_mean_robust_z",
            "median",
        ),

        median_tau_peak_z=(
            "j3_tau_peak_robust_z",
            "median",
        ),

        median_dq_rms_z=(
            "j3_dq_rms_robust_z",
            "median",
        ),
    )
    .reset_index()
)


print(
    "\nBENCHMARK REGION CHECK"
)
print("-" * 104)

print(
    benchmark_summary
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 17. OUTPUTS
# =============================================================================

print(
    f"\nCycle-level degradation features saved to:\n"
    f"{OUTPUT_CYCLES}"
)

print(
    f"\nSequence-bin degradation summary saved to:\n"
    f"{OUTPUT_BINS}"
)

print("\n" + "=" * 104)
print("PHASE 04.1 J3 DEGRADATION TRAJECTORY COMPLETE")
print("=" * 104)