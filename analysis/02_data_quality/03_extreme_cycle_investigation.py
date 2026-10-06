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
    / "me_ad_extreme_cycle_investigation.csv"
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
# 3. EXTREME CYCLES IDENTIFIED IN PHASE 01
# =============================================================================

extreme_cycles = [
    ("2200", 311),
    ("3000", 187),
    ("2400", 976),
    ("3000", 1207),
    ("2400", 1236),
    ("2100", 1205),
    ("2200", 1561),
    ("3100", 1259),
]


# =============================================================================
# 4. LOAD MANIFEST
# =============================================================================

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={"operation_code": str},
)

print("=" * 96)
print("ME-AD PHASE 02.3 — EXTREME CYCLE INVESTIGATION")
print("=" * 96)

print(
    f"\nExtreme cycles selected : "
    f"{len(extreme_cycles)}"
)


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

def load_cycle(archive, operation_code, cycle_index):

    member = (
        f"ME-AD/Pandas/{operation_code}/"
        f"cleaned_dataset_{cycle_index}.pkl"
    )

    raw_bytes = archive.read(member)

    df = pd.read_pickle(
        io.BytesIO(raw_bytes)
    )

    return df, len(raw_bytes)


def rms(values):

    values = np.asarray(values)

    return np.sqrt(
        np.mean(values ** 2)
    )


# =============================================================================
# 6. INSPECT EXTREMES AND LOCAL NEIGHBOURS
# =============================================================================

records = []

with ZipFile(ZIP_FILE, "r") as archive:

    for operation_code, extreme_index in extreme_cycles:

        op_manifest = (
            manifest[
                manifest["operation_code"] == operation_code
            ]
            .sort_values("cycle_index")
            .copy()
        )

        max_index = int(
            op_manifest["cycle_index"].max()
        )

        comparison_indices = sorted(
            set(
                idx
                for idx in [
                    extreme_index - 2,
                    extreme_index - 1,
                    extreme_index,
                    extreme_index + 1,
                    extreme_index + 2,
                ]
                if 0 <= idx <= max_index
            )
        )

        print(
            f"\nOperation {operation_code} | "
            f"Extreme cycle {extreme_index}"
        )

        for cycle_index in comparison_indices:

            df, pkl_bytes = load_cycle(
                archive,
                operation_code,
                cycle_index,
            )

            manifest_row = op_manifest[
                op_manifest["cycle_index"] == cycle_index
            ].iloc[0]

            record = {
                "operation_code": operation_code,
                "target_extreme_cycle": extreme_index,
                "cycle_index": cycle_index,
                "relative_position": cycle_index - extreme_index,
                "is_extreme_target": cycle_index == extreme_index,
                "benchmark_health_region":
                    manifest_row["benchmark_health_region"],
                "sequence_position":
                    manifest_row["sequence_position"],
                "rows": len(df),
                "duration_seconds":
                    len(df) * SAMPLING_INTERVAL,
                "pkl_bytes": pkl_bytes,
                "missing_values":
                    int(df.isna().sum().sum()),
                "duplicate_rows":
                    int(df.duplicated().sum()),
                "duplicate_index":
                    int(df.index.duplicated().sum()),
            }

            # -------------------------------------------------------------
            # Empty processed cycle
            # -------------------------------------------------------------

            if len(df) == 0:

                for joint in range(1, 7):
                    record[f"j{joint}_q_range"] = np.nan
                    record[f"j{joint}_dq_rms"] = np.nan
                    record[f"j{joint}_ddq_rms"] = np.nan
                    record[f"j{joint}_tau_rms"] = np.nan
                    record[f"j{joint}_tau_abs_max"] = np.nan

            # -------------------------------------------------------------
            # Non-empty cycle: calculate physical signal metrics
            # -------------------------------------------------------------

            else:

                for joint in range(1, 7):

                    q = df[
                        f"q_{joint}"
                    ].to_numpy()

                    dq = df[
                        f"dq_{joint}"
                    ].to_numpy()

                    ddq = df[
                        f"ddq_{joint}"
                    ].to_numpy()

                    tau = df[
                        f"tau_{joint}"
                    ].to_numpy()

                    record[
                        f"j{joint}_q_range"
                    ] = np.ptp(q)

                    record[
                        f"j{joint}_dq_rms"
                    ] = rms(dq)

                    record[
                        f"j{joint}_ddq_rms"
                    ] = rms(ddq)

                    record[
                        f"j{joint}_tau_rms"
                    ] = rms(tau)

                    record[
                        f"j{joint}_tau_abs_max"
                    ] = np.max(
                        np.abs(tau)
                    )

            records.append(record)

            marker = (
                " <--- TARGET"
                if cycle_index == extreme_index
                else ""
            )

            empty_marker = (
                " [EMPTY AFTER CLEANING]"
                if len(df) == 0
                else ""
            )

            print(
                f"  cycle={cycle_index:5d} | "
                f"rows={len(df):6d} | "
                f"duration="
                f"{len(df) * SAMPLING_INTERVAL:8.3f}s | "
                f"bytes={pkl_bytes:9d}"
                f"{marker}"
                f"{empty_marker}"
            )
results = pd.DataFrame(records)

# =============================================================================
# 7. LOCAL COMPARISON RATIOS
# =============================================================================

results[
    "row_ratio_vs_local_median"
] = np.nan

results[
    "duration_ratio_vs_local_median"
] = np.nan

results[
    "j3_tau_rms_ratio_vs_local_median"
] = np.nan


for (
    operation_code,
    target_extreme_cycle
), group in results.groupby(
    [
        "operation_code",
        "target_extreme_cycle",
    ]
):

    target_mask = (
        (results["operation_code"] == operation_code)
        &
        (
            results["target_extreme_cycle"]
            == target_extreme_cycle
        )
    )

    group_indices = results[
        target_mask
    ].index

    neighbour_group = results.loc[
        group_indices
    ]

    neighbours = neighbour_group[
        ~neighbour_group[
            "is_extreme_target"
        ]
    ]

    if len(neighbours) == 0:
        continue

    row_median = (
        neighbours["rows"].median()
    )

    duration_median = (
        neighbours[
            "duration_seconds"
        ].median()
    )

    j3_tau_median = (
        neighbours[
            "j3_tau_rms"
        ].median()
    )

    target_index = (
        neighbour_group[
            neighbour_group[
                "is_extreme_target"
            ]
        ].index[0]
    )

    results.loc[
        target_index,
        "row_ratio_vs_local_median",
    ] = (
        results.loc[
            target_index,
            "rows",
        ]
        / row_median
    )

    results.loc[
        target_index,
        "duration_ratio_vs_local_median",
    ] = (
        results.loc[
            target_index,
            "duration_seconds",
        ]
        / duration_median
    )

    results.loc[
        target_index,
        "j3_tau_rms_ratio_vs_local_median",
    ] = (
        results.loc[
            target_index,
            "j3_tau_rms",
        ]
        / j3_tau_median
    )


# =============================================================================
# 8. SAVE FULL RESULTS
# =============================================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =============================================================================
# 9. PRINT TARGET SUMMARY
# =============================================================================

targets = (
    results[
        results["is_extreme_target"]
    ]
    .copy()
)

summary_columns = [
    "operation_code",
    "cycle_index",
    "benchmark_health_region",
    "sequence_position",
    "rows",
    "duration_seconds",
    "pkl_bytes",
    "missing_values",
    "duplicate_rows",
    "duplicate_index",
    "row_ratio_vs_local_median",
    "j3_tau_rms",
    "j3_tau_rms_ratio_vs_local_median",
]

print("\n" + "=" * 96)
print("EXTREME TARGET SUMMARY")
print("=" * 96)

print(
    targets[
        summary_columns
    ]
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 10. J3 PHYSICAL COMPARISON
# =============================================================================

print("\n" + "=" * 96)
print("J3 — EXTREME CYCLES AND LOCAL NEIGHBOURS")
print("=" * 96)

print(
    results[
        [
            "operation_code",
            "target_extreme_cycle",
            "cycle_index",
            "relative_position",
            "is_extreme_target",
            "rows",
            "duration_seconds",
            "j3_q_range",
            "j3_dq_rms",
            "j3_ddq_rms",
            "j3_tau_rms",
            "j3_tau_abs_max",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


print(
    f"\nDetailed results saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 96)
print("PHASE 02.3 EXTREME CYCLE INVESTIGATION COMPLETE")
print("=" * 96)