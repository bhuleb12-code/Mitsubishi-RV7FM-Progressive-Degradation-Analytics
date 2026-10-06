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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_stratified_cycle_quality.csv"
)


# =============================================================================
# 2. PICKLE COMPATIBILITY SHIM
# =============================================================================
# Older pandas pickles may reference pandas.core.indexes.numeric.
# Newer pandas versions removed that module.

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
# 3. LOAD MANIFEST
# =============================================================================

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={"operation_code": str},
)

print("=" * 88)
print("ME-AD PHASE 02 — STRATIFIED CYCLE DATA QUALITY")
print("=" * 88)

print(f"\nAvailable cycles : {len(manifest):,}")
print(
    f"Operations       : "
    f"{manifest['operation_code'].nunique()}"
)


# =============================================================================
# 4. STRATIFIED SAMPLE DESIGN
# =============================================================================
# For every operation we inspect:
#
#   first 5 cycles
#   five cycles around 25%
#   five cycles around 50%
#   five cycles around 75%
#   last 5 cycles
#
# This gives broad coverage of the degradation sequence without loading
# the entire archive.

sample_records = []

for operation_code, group in manifest.groupby("operation_code"):

    group = group.sort_values("cycle_index")

    max_index = int(group["cycle_index"].max())

    centres = [
        0,
        round(max_index * 0.25),
        round(max_index * 0.50),
        round(max_index * 0.75),
        max_index,
    ]

    selected_indices = set()

    for centre in centres:

        if centre == 0:
            candidates = range(
                0,
                min(5, max_index + 1),
            )

        elif centre == max_index:
            candidates = range(
                max(0, max_index - 4),
                max_index + 1,
            )

        else:
            candidates = range(
                max(0, centre - 2),
                min(max_index + 1, centre + 3),
            )

        selected_indices.update(candidates)

    selected = group[
        group["cycle_index"].isin(
            sorted(selected_indices)
        )
    ].copy()

    sample_records.append(selected)


sample_manifest = pd.concat(
    sample_records,
    ignore_index=True,
)

print(
    f"Cycles selected  : "
    f"{len(sample_manifest):,}"
)

print(
    f"Approx per op    : "
    f"{len(sample_manifest) / manifest['operation_code'].nunique():.1f}"
)


# =============================================================================
# 5. EXPECTED CORE SIGNALS
# =============================================================================

signal_prefixes = [
    "q",
    "dq",
    "ddq",
    "q_filt",
    "dq_filt",
    "ddq_filt",
    "tau",
    "tau_MAT",
    "tau_filt",
]

expected_columns = [
    f"{prefix}_{joint}"
    for prefix in signal_prefixes
    for joint in range(1, 7)
]

# =============================================================================
# 6. INSPECT SELECTED PICKLES
# =============================================================================

quality_records = []

with ZipFile(ZIP_FILE, "r") as archive:

    total = len(sample_manifest)

    for counter, row in enumerate(
        sample_manifest.itertuples(index=False),
        start=1,
    ):

        member = row.pkl_member

        try:
            raw_bytes = archive.read(member)

            df = pd.read_pickle(
                io.BytesIO(raw_bytes)
            )

            numeric_df = df.select_dtypes(
                include=[np.number]
            )

            missing_cells = int(
                df.isna().sum().sum()
            )

            infinite_cells = int(
                np.isinf(
                    numeric_df.to_numpy()
                ).sum()
            )

            duplicate_rows = int(
                df.duplicated().sum()
            )

            missing_expected_columns = [
                column
                for column in expected_columns
                if column not in df.columns
            ]

            index_duplicate_count = int(
                df.index.duplicated().sum()
            )

            if len(df) > 0:
                index_min = df.index.min()
                index_max = df.index.max()
            else:
                index_min = np.nan
                index_max = np.nan

            quality_records.append(
                {
                    "operation_code":
                        row.operation_code,

                    "cycle_index":
                        row.cycle_index,

                    "benchmark_health_region":
                        row.benchmark_health_region,

                    "sequence_position":
                        row.sequence_position,

                    "pkl_bytes":
                        row.pkl_bytes,

                    "rows":
                        len(df),

                    "columns":
                        len(df.columns),

                    "numeric_columns":
                        len(numeric_df.columns),

                    "missing_cells":
                        missing_cells,

                    "infinite_cells":
                        infinite_cells,

                    "duplicate_rows":
                        duplicate_rows,

                    "duplicate_index_values":
                        index_duplicate_count,

                    "index_min":
                        index_min,

                    "index_max":
                        index_max,

                    "missing_expected_columns":
                        len(
                            missing_expected_columns
                        ),

                    "missing_expected_names":
                        ",".join(
                            missing_expected_columns
                        ),

                    "load_status":
                        "OK",
                }
            )

        except Exception as exc:

            quality_records.append(
                {
                    "operation_code":
                        row.operation_code,

                    "cycle_index":
                        row.cycle_index,

                    "benchmark_health_region":
                        row.benchmark_health_region,

                    "sequence_position":
                        row.sequence_position,

                    "pkl_bytes":
                        row.pkl_bytes,

                    "rows":
                        np.nan,

                    "columns":
                        np.nan,

                    "numeric_columns":
                        np.nan,

                    "missing_cells":
                        np.nan,

                    "infinite_cells":
                        np.nan,

                    "duplicate_rows":
                        np.nan,

                    "duplicate_index_values":
                        np.nan,

                    "index_min":
                        np.nan,

                    "index_max":
                        np.nan,

                    "missing_expected_columns":
                        np.nan,

                    "missing_expected_names":
                        "",

                    "load_status":
                        f"ERROR: {exc}",
                }
            )

        if (
            counter % 50 == 0
            or counter == total
        ):
            print(
                f"Processed "
                f"{counter:,}/{total:,} cycles"
            )


quality = pd.DataFrame(
    quality_records
)


# =============================================================================
# 7. DERIVED CYCLE DURATION
# =============================================================================

SAMPLING_INTERVAL_SECONDS = 0.003555

quality["approx_duration_seconds"] = (
    quality["rows"]
    * SAMPLING_INTERVAL_SECONDS
)


# =============================================================================
# 8. OVERALL QUALITY SUMMARY
# =============================================================================

print("\nOVERALL SAMPLE QUALITY")
print("-" * 88)

print(
    f"Cycles inspected       : "
    f"{len(quality):,}"
)

print(
    f"Successful loads       : "
    f"{(quality['load_status'] == 'OK').sum():,}"
)

print(
    f"Failed loads           : "
    f"{(quality['load_status'] != 'OK').sum():,}"
)

print(
    f"Cycles with missing    : "
    f"{(quality['missing_cells'] > 0).sum():,}"
)

print(
    f"Cycles with infinities : "
    f"{(quality['infinite_cells'] > 0).sum():,}"
)

print(
    f"Cycles with duplicates : "
    f"{(quality['duplicate_rows'] > 0).sum():,}"
)

print(
    f"Duplicate indices      : "
    f"{(quality['duplicate_index_values'] > 0).sum():,}"
)

print(
    f"Schema problems        : "
    f"{(quality['missing_expected_columns'] > 0).sum():,}"
)


# =============================================================================
# 9. OPERATION-LEVEL SUMMARY
# =============================================================================

operation_quality = (
    quality
    .groupby("operation_code")
    .agg(
        sampled_cycles=("cycle_index", "count"),

        min_rows=("rows", "min"),
        median_rows=("rows", "median"),
        max_rows=("rows", "max"),

        min_duration_s=(
            "approx_duration_seconds",
            "min",
        ),

        median_duration_s=(
            "approx_duration_seconds",
            "median",
        ),

        max_duration_s=(
            "approx_duration_seconds",
            "max",
        ),

        missing_cells=("missing_cells", "sum"),
        infinite_cells=("infinite_cells", "sum"),
        duplicate_rows=("duplicate_rows", "sum"),

        schema_problems=(
            "missing_expected_columns",
            lambda x: (x > 0).sum(),
        ),
    )
    .reset_index()
)

print("\nQUALITY BY OPERATION")
print("-" * 88)

print(
    operation_quality
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 10. QUALITY BY EXPERIMENT POSITION
# =============================================================================

quality["sequence_band"] = pd.cut(
    quality["sequence_position"],
    bins=[
        -0.001,
        0.10,
        0.35,
        0.65,
        0.90,
        1.001,
    ],
    labels=[
        "early",
        "early_middle",
        "middle",
        "late_middle",
        "late",
    ],
)

position_summary = (
    quality
    .groupby(
        "sequence_band",
        observed=True,
    )
    .agg(
        sampled_cycles=("cycle_index", "count"),
        median_rows=("rows", "median"),
        min_rows=("rows", "min"),
        max_rows=("rows", "max"),
        missing_cells=("missing_cells", "sum"),
        infinite_cells=("infinite_cells", "sum"),
    )
    .reset_index()
)

print("\nQUALITY BY SEQUENCE POSITION")
print("-" * 88)

print(
    position_summary
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 11. MOST UNUSUAL SAMPLED CYCLE LENGTHS
# =============================================================================

quality["operation_median_rows"] = (
    quality
    .groupby("operation_code")["rows"]
    .transform("median")
)

quality["row_deviation_from_median"] = (
    quality["rows"]
    - quality["operation_median_rows"]
).abs()

extreme_rows = (
    quality
    .sort_values(
        "row_deviation_from_median",
        ascending=False,
    )
    .head(20)
)

print("\nTOP 20 SAMPLED CYCLE-LENGTH DEVIATIONS")
print("-" * 88)

print(
    extreme_rows[
        [
            "operation_code",
            "cycle_index",
            "benchmark_health_region",
            "sequence_position",
            "rows",
            "approx_duration_seconds",
            "operation_median_rows",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 12. SAVE
# =============================================================================

quality.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nDetailed quality table saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 88)
print("PHASE 02 STRATIFIED DATA QUALITY COMPLETE")
print("=" * 88)