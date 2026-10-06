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

OUTPUT_CYCLE_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_healthy_cycle_features.csv"
)

OUTPUT_BASELINE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_healthy_operating_baseline.csv"
)

SAMPLING_INTERVAL = 0.003555


# =============================================================================
# 2. OLD-PANDAS PICKLE COMPATIBILITY
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


def load_cycle(
    archive,
    operation_code,
    cycle_index,
):
    member = (
        f"ME-AD/Pandas/{operation_code}/"
        f"cleaned_dataset_{cycle_index}.pkl"
    )

    raw_bytes = archive.read(member)

    return pd.read_pickle(
        io.BytesIO(raw_bytes)
    )


# =============================================================================
# 4. LOAD MANIFEST AND SELECT TRAINING-HEALTHY REGION
# =============================================================================

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={"operation_code": str},
)

healthy = manifest[
    manifest["benchmark_health_region"]
    == "training_healthy"
].copy()

healthy = healthy.sort_values(
    [
        "operation_code",
        "cycle_index",
    ]
)

print("=" * 100)
print("ME-AD PHASE 03.1 — HEALTHY OPERATING BASELINE")
print("=" * 100)

print(
    f"\nHealthy cycles selected : "
    f"{len(healthy):,}"
)

print(
    f"Operations              : "
    f"{healthy['operation_code'].nunique()}"
)

print(
    f"Expected cycles/op       : "
    f"{len(healthy) / healthy['operation_code'].nunique():.1f}"
)


# =============================================================================
# 5. BUILD CYCLE-LEVEL HEALTHY FEATURES
# =============================================================================

records = []

with ZipFile(ZIP_FILE, "r") as archive:

    total = len(healthy)

    for counter, row in enumerate(
        healthy.itertuples(index=False),
        start=1,
    ):

        operation_code = row.operation_code
        cycle_index = int(row.cycle_index)

        df = load_cycle(
            archive,
            operation_code,
            cycle_index,
        )

        if len(df) == 0:
            continue

        record = {
            "operation_code":
                operation_code,

            "cycle_index":
                cycle_index,

            "operation_family":
                row.operation_family,

            "commanded_speed_group":
                row.commanded_speed_group,

            "rows":
                len(df),

            "duration_seconds":
                len(df) * SAMPLING_INTERVAL,
        }

        for joint in range(1, 7):

            q = df[
                f"q_filt_{joint}"
            ].to_numpy()

            dq = df[
                f"dq_filt_{joint}"
            ].to_numpy()

            ddq = df[
                f"ddq_filt_{joint}"
            ].to_numpy()

            tau = df[
                f"tau_filt_{joint}"
            ].to_numpy()

            # ---------------------------------------------------------
            # Motion features
            # ---------------------------------------------------------

            record[
                f"j{joint}_q_min"
            ] = np.min(q)

            record[
                f"j{joint}_q_max"
            ] = np.max(q)

            record[
                f"j{joint}_q_range"
            ] = np.ptp(q)

            record[
                f"j{joint}_dq_rms"
            ] = rms(dq)

            record[
                f"j{joint}_dq_peak"
            ] = np.max(
                np.abs(dq)
            )

            record[
                f"j{joint}_ddq_rms"
            ] = rms(ddq)

            # ---------------------------------------------------------
            # Effort features
            # ---------------------------------------------------------

            record[
                f"j{joint}_tau_mean"
            ] = np.mean(tau)

            record[
                f"j{joint}_tau_abs_mean"
            ] = np.mean(
                np.abs(tau)
            )

            record[
                f"j{joint}_tau_rms"
            ] = rms(tau)

            record[
                f"j{joint}_tau_peak"
            ] = np.max(
                np.abs(tau)
            )

            record[
                f"j{joint}_tau_std"
            ] = np.std(tau)

        records.append(record)

        if (
            counter % 40 == 0
            or counter == total
        ):
            print(
                f"Processed "
                f"{counter:,}/{total:,} "
                f"healthy cycles"
            )


cycle_features = pd.DataFrame(
    records
)

cycle_features.to_csv(
    OUTPUT_CYCLE_FEATURES,
    index=False,
)


# =============================================================================
# 6. BUILD OPERATION × JOINT HEALTHY BASELINE
# =============================================================================

baseline_records = []

for operation_code in sorted(
    cycle_features[
        "operation_code"
    ].unique()
):

    op = cycle_features[
        cycle_features[
            "operation_code"
        ] == operation_code
    ]

    for joint in range(1, 7):

        baseline_records.append(
            {
                "operation_code":
                    operation_code,

                "joint":
                    joint,

                "cycles":
                    len(op),

                "median_duration":
                    op[
                        "duration_seconds"
                    ].median(),

                "median_q_range":
                    op[
                        f"j{joint}_q_range"
                    ].median(),

                "median_dq_rms":
                    op[
                        f"j{joint}_dq_rms"
                    ].median(),

                "median_ddq_rms":
                    op[
                        f"j{joint}_ddq_rms"
                    ].median(),

                "median_tau_abs":
                    op[
                        f"j{joint}_tau_abs_mean"
                    ].median(),

                "median_tau_rms":
                    op[
                        f"j{joint}_tau_rms"
                    ].median(),

                "median_tau_peak":
                    op[
                        f"j{joint}_tau_peak"
                    ].median(),

                "median_tau_std":
                    op[
                        f"j{joint}_tau_std"
                    ].median(),

                "tau_rms_iqr":
                    (
                        op[
                            f"j{joint}_tau_rms"
                        ].quantile(0.75)
                        -
                        op[
                            f"j{joint}_tau_rms"
                        ].quantile(0.25)
                    ),

                "dq_rms_iqr":
                    (
                        op[
                            f"j{joint}_dq_rms"
                        ].quantile(0.75)
                        -
                        op[
                            f"j{joint}_dq_rms"
                        ].quantile(0.25)
                    ),
            }
        )


baseline = pd.DataFrame(
    baseline_records
)

baseline.to_csv(
    OUTPUT_BASELINE,
    index=False,
)


# =============================================================================
# 7. HEALTHY CYCLE DURATION BY OPERATION
# =============================================================================

duration_summary = (
    cycle_features
    .groupby("operation_code")
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),
        duration_min=(
            "duration_seconds",
            "min",
        ),
        duration_median=(
            "duration_seconds",
            "median",
        ),
        duration_max=(
            "duration_seconds",
            "max",
        ),
    )
    .reset_index()
)

print(
    "\nHEALTHY CYCLE DURATION BY OPERATION"
)
print("-" * 100)

print(
    duration_summary
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 8. J3 HEALTHY BASELINE BY OPERATION
# =============================================================================

j3_baseline = baseline[
    baseline["joint"] == 3
].copy()

print(
    "\nJ3 HEALTHY OPERATING BASELINE"
)
print("-" * 100)

print(
    j3_baseline[
        [
            "operation_code",
            "cycles",
            "median_q_range",
            "median_dq_rms",
            "median_ddq_rms",
            "median_tau_abs",
            "median_tau_rms",
            "median_tau_peak",
            "tau_rms_iqr",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 9. ALL-JOINT TORQUE BASELINE
# =============================================================================

joint_summary = (
    baseline
    .groupby("joint")
    .agg(
        operation_median_tau_abs=(
            "median_tau_abs",
            "median",
        ),
        operation_median_tau_rms=(
            "median_tau_rms",
            "median",
        ),
        operation_median_tau_peak=(
            "median_tau_peak",
            "median",
        ),
    )
    .reset_index()
)

print(
    "\nHEALTHY TORQUE BASELINE ACROSS JOINTS"
)
print("-" * 100)

print(
    joint_summary
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 10. J3 NORMAL VARIABILITY
# =============================================================================

j3_variability = (
    baseline[
        baseline["joint"] == 3
    ][
        [
            "operation_code",
            "median_tau_rms",
            "tau_rms_iqr",
            "median_dq_rms",
            "dq_rms_iqr",
        ]
    ]
    .copy()
)

j3_variability[
    "tau_rms_iqr_pct"
] = (
    100
    * j3_variability[
        "tau_rms_iqr"
    ]
    / j3_variability[
        "median_tau_rms"
    ]
)

j3_variability[
    "dq_rms_iqr_pct"
] = (
    100
    * j3_variability[
        "dq_rms_iqr"
    ]
    / j3_variability[
        "median_dq_rms"
    ]
)

print(
    "\nJ3 HEALTHY REPEATABILITY"
)
print("-" * 100)

print(
    j3_variability
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 11. OUTPUTS
# =============================================================================

print(
    f"\nCycle-level healthy features saved to:\n"
    f"{OUTPUT_CYCLE_FEATURES}"
)

print(
    f"\nHealthy operating baseline saved to:\n"
    f"{OUTPUT_BASELINE}"
)

print("\n" + "=" * 100)
print("PHASE 03.1 HEALTHY OPERATING BASELINE COMPLETE")
print("=" * 100)