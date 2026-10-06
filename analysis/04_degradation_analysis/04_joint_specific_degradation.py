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

J3_QUALITY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_j3_degradation_cycle_features.csv"
)

OUTPUT_CYCLE_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_all_joint_degradation_features.csv"
)

OUTPUT_JOINT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_joint_degradation_summary.csv"
)

OUTPUT_LOCALIZATION = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_j3_localization_summary.csv"
)

OUTPUT_LATE_WINDOWS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_joint_late_degradation_windows.csv"
)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

JOINTS = range(1, 7)

EFFORT_METRICS = [
    "tau_abs_mean",
    "tau_rms",
    "tau_peak",
    "tau_std",
]

# Known strongest diagnostic operations from Phase 04.3.
# We still calculate all 16 operations; this group is used only for
# an additional focused summary.
STRONG_DIAGNOSTIC_OPERATIONS = [
    "1050",
    "1051",
    "1052",
    "1053",
    "1100",
    "1101",
]


# =============================================================================
# 3. PICKLE COMPATIBILITY
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
# 4. HELPERS
# =============================================================================

def rms(values):

    values = np.asarray(values)

    if len(values) == 0:
        return np.nan

    return np.sqrt(
        np.mean(values ** 2)
    )


# =============================================================================
# 5. LOAD METADATA AND QUALITY DECISIONS
# =============================================================================

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={"operation_code": str},
)

quality = pd.read_csv(
    J3_QUALITY_FILE,
    dtype={"operation_code": str},
)

quality = quality[
    [
        "operation_code",
        "cycle_index",
        "structural_quality",
    ]
]

manifest = manifest.merge(
    quality,
    on=[
        "operation_code",
        "cycle_index",
    ],
    how="left",
    validate="one_to_one",
)


print("=" * 112)
print("ME-AD PHASE 04.4 — JOINT-SPECIFIC DEGRADATION LOCALIZATION")
print("=" * 112)

print(
    f"\nManifest cycles          : "
    f"{len(manifest):,}"
)

print(
    f"Operations               : "
    f"{manifest['operation_code'].nunique()}"
)

print(
    f"Structurally usable      : "
    f"{(manifest['structural_quality'] == 'usable').sum():,}"
)


# =============================================================================
# 6. EXTRACT ALL-JOINT CYCLE EFFORT FEATURES
# =============================================================================

records = []

with ZipFile(ZIP_FILE, "r") as archive:

    total = len(manifest)

    for counter, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):

        record = {
            "operation_code":
                row.operation_code,

            "cycle_index":
                int(row.cycle_index),

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

            "structural_quality":
                row.structural_quality,
        }

        # Preserve audit trail but do not analyse structurally bad cycles.
        if row.structural_quality != "usable":

            for joint in JOINTS:
                for metric in EFFORT_METRICS:
                    record[
                        f"j{joint}_{metric}"
                    ] = np.nan

            records.append(record)
            continue

        member = (
            f"ME-AD/Pandas/"
            f"{row.operation_code}/"
            f"cleaned_dataset_"
            f"{int(row.cycle_index)}.pkl"
        )

        raw_bytes = archive.read(
            member
        )

        df = pd.read_pickle(
            io.BytesIO(raw_bytes)
        )

        for joint in JOINTS:

            tau = df[
                f"tau_filt_{joint}"
            ].to_numpy()

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

        records.append(
            record
        )

        if (
            counter % 2000 == 0
            or counter == total
        ):
            print(
                f"Processed "
                f"{counter:,}/{total:,} cycles"
            )


features = pd.DataFrame(
    records
)

features.to_csv(
    OUTPUT_CYCLE_FEATURES,
    index=False,
)


# =============================================================================
# 7. HEALTHY VALIDATION ENVELOPES FOR EACH JOINT
# =============================================================================

usable = features[
    features["structural_quality"]
    == "usable"
].copy()

limit_records = []

for operation_code, group in usable.groupby(
    "operation_code"
):

    healthy = group[
        group[
            "benchmark_health_region"
        ] == "test_healthy"
    ]

    record = {
        "operation_code":
            operation_code
    }

    for joint in JOINTS:

        for metric in EFFORT_METRICS:

            feature = (
                f"j{joint}_{metric}"
            )

            values = healthy[
                feature
            ].dropna()

            record[
                f"{feature}_healthy_min"
            ] = values.min()

            record[
                f"{feature}_healthy_max"
            ] = values.max()

            record[
                f"{feature}_healthy_median"
            ] = values.median()

    limit_records.append(
        record
    )


limits = pd.DataFrame(
    limit_records
)

usable = usable.merge(
    limits,
    on="operation_code",
    how="left",
    validate="many_to_one",
)


# =============================================================================
# 8. CYCLE-LEVEL OUTSIDE-HEALTHY FLAGS
# =============================================================================

for joint in JOINTS:

    for metric in EFFORT_METRICS:

        feature = (
            f"j{joint}_{metric}"
        )

        lower = usable[
            f"{feature}_healthy_min"
        ]

        upper = usable[
            f"{feature}_healthy_max"
        ]

        usable[
            f"{feature}_outside"
        ] = (
            (usable[feature] < lower)
            |
            (usable[feature] > upper)
        )


# =============================================================================
# 9. JOINT-LEVEL REGION SUMMARY
# =============================================================================

summary_records = []

region_order = [
    "training_healthy",
    "test_healthy",
    "intermediate_unlabelled",
    "faulty_evident",
]

for operation_code, group in usable.groupby(
    "operation_code"
):

    for region in region_order:

        region_data = group[
            group[
                "benchmark_health_region"
            ] == region
        ]

        if len(region_data) == 0:
            continue

        for joint in JOINTS:

            record = {
                "operation_code":
                    operation_code,

                "benchmark_health_region":
                    region,

                "joint":
                    joint,

                "cycles":
                    len(region_data),
            }

            for metric in EFFORT_METRICS:

                feature = (
                    f"j{joint}_{metric}"
                )

                outside_col = (
                    f"{feature}_outside"
                )

                healthy_median = (
                    region_data[
                        f"{feature}_healthy_median"
                    ].iloc[0]
                )

                current_median = (
                    region_data[
                        feature
                    ].median()
                )

                record[
                    f"median_{metric}"
                ] = current_median

                record[
                    f"{metric}_pct_change_from_healthy"
                ] = (
                    100
                    * (
                        current_median
                        - healthy_median
                    )
                    / healthy_median
                    if healthy_median != 0
                    else np.nan
                )

                record[
                    f"{metric}_outside_pct"
                ] = (
                    100
                    * region_data[
                        outside_col
                    ].mean()
                )

            summary_records.append(
                record
            )


joint_summary = pd.DataFrame(
    summary_records
)

joint_summary.to_csv(
    OUTPUT_JOINT_SUMMARY,
    index=False,
)


# =============================================================================
# 10. FAULTY-REGION LOCALIZATION
# =============================================================================

faulty = joint_summary[
    joint_summary[
        "benchmark_health_region"
    ] == "faulty_evident"
].copy()


localization_records = []

for operation_code, group in faulty.groupby(
    "operation_code"
):

    j3 = group[
        group["joint"] == 3
    ].iloc[0]

    controls = group[
        group["joint"] != 3
    ]

    record = {
        "operation_code":
            operation_code,

        "j3_tau_rms_pct_change":
            j3[
                "tau_rms_pct_change_from_healthy"
            ],

        "control_median_tau_rms_pct_change":
            controls[
                "tau_rms_pct_change_from_healthy"
            ].median(),

        "j3_minus_control_tau_rms_pct_points":
            (
                j3[
                    "tau_rms_pct_change_from_healthy"
                ]
                -
                controls[
                    "tau_rms_pct_change_from_healthy"
                ].median()
            ),

        "j3_tau_rms_outside_pct":
            j3[
                "tau_rms_outside_pct"
            ],

        "control_median_tau_rms_outside_pct":
            controls[
                "tau_rms_outside_pct"
            ].median(),

        "j3_tau_peak_outside_pct":
            j3[
                "tau_peak_outside_pct"
            ],

        "control_median_tau_peak_outside_pct":
            controls[
                "tau_peak_outside_pct"
            ].median(),
    }

    localization_records.append(
        record
    )


localization = pd.DataFrame(
    localization_records
)

localization.to_csv(
    OUTPUT_LOCALIZATION,
    index=False,
)


# =============================================================================
# 11. LAST 200 PRE-FAULT CYCLES — JOINT COMPARISON
# =============================================================================

late_records = []

for operation_code, group in usable.groupby(
    "operation_code"
):

    faulty_cycles = group[
        group[
            "benchmark_health_region"
        ] == "faulty_evident"
    ]

    boundary = int(
        faulty_cycles[
            "cycle_index"
        ].min()
    )

    late = group[
        (
            group["cycle_index"]
            >= boundary - 200
        )
        &
        (
            group["cycle_index"]
            < boundary
        )
    ]

    if len(late) == 0:
        continue

    for joint in JOINTS:

        record = {
            "operation_code":
                operation_code,

            "joint":
                joint,

            "cycles":
                len(late),

            "evident_fault_start_cycle":
                boundary,
        }

        for metric in EFFORT_METRICS:

            outside_col = (
                f"j{joint}_{metric}_outside"
            )

            record[
                f"{metric}_outside_pct"
            ] = (
                100
                * late[
                    outside_col
                ].mean()
            )

        late_records.append(
            record
        )


late_summary = pd.DataFrame(
    late_records
)

late_summary.to_csv(
    OUTPUT_LATE_WINDOWS,
    index=False,
)


# =============================================================================
# 12. PRINT FAULTY-REGION LOCALIZATION
# =============================================================================

print(
    "\nFAULTY-REGION J3 LOCALIZATION"
)
print("-" * 112)

print(
    localization
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 13. ALL-JOINT FAULTY-REGION TORQUE RMS
# =============================================================================

print(
    "\nFAULTY-REGION TORQUE RMS CHANGE FROM HEALTHY BASELINE"
)
print("-" * 112)

faulty_rms = faulty[
    [
        "operation_code",
        "joint",
        "tau_rms_pct_change_from_healthy",
        "tau_rms_outside_pct",
    ]
].copy()

print(
    faulty_rms
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 14. STRONG DIAGNOSTIC OPERATIONS — LAST 200 PRE-FAULT CYCLES
# =============================================================================

strong_late = late_summary[
    late_summary[
        "operation_code"
    ].isin(
        STRONG_DIAGNOSTIC_OPERATIONS
    )
].copy()


print(
    "\nSTRONG DIAGNOSTIC OPERATIONS — LAST 200 PRE-FAULT CYCLES"
)
print("-" * 112)

print(
    strong_late[
        [
            "operation_code",
            "joint",
            "tau_abs_mean_outside_pct",
            "tau_rms_outside_pct",
            "tau_peak_outside_pct",
            "tau_std_outside_pct",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 15. J3 VS CONTROL JOINTS IN STRONG OPERATIONS
# =============================================================================

comparison_records = []

for operation_code, group in strong_late.groupby(
    "operation_code"
):

    j3 = group[
        group["joint"] == 3
    ].iloc[0]

    controls = group[
        group["joint"] != 3
    ]

    comparison_records.append(
        {
            "operation_code":
                operation_code,

            "j3_tau_rms_outside_pct":
                j3[
                    "tau_rms_outside_pct"
                ],

            "control_median_tau_rms_outside_pct":
                controls[
                    "tau_rms_outside_pct"
                ].median(),

            "j3_minus_control_tau_rms_pct_points":
                (
                    j3[
                        "tau_rms_outside_pct"
                    ]
                    -
                    controls[
                        "tau_rms_outside_pct"
                    ].median()
                ),

            "j3_tau_peak_outside_pct":
                j3[
                    "tau_peak_outside_pct"
                ],

            "control_median_tau_peak_outside_pct":
                controls[
                    "tau_peak_outside_pct"
                ].median(),

            "j3_minus_control_tau_peak_pct_points":
                (
                    j3[
                        "tau_peak_outside_pct"
                    ]
                    -
                    controls[
                        "tau_peak_outside_pct"
                    ].median()
                ),
        }
    )


comparison = pd.DataFrame(
    comparison_records
)


print(
    "\nJ3 VS NON-TARGET JOINTS — LAST 200 PRE-FAULT CYCLES"
)
print("-" * 112)

print(
    comparison
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 16. POOLED STRONG-OPERATION LOCALIZATION
# =============================================================================

if len(comparison) > 0:

    print(
        "\nPOOLED STRONG-OPERATION LOCALIZATION"
    )
    print("-" * 112)

    print(
        "Median J3 torque-RMS outside rate        : "
        f"{comparison['j3_tau_rms_outside_pct'].median():.2f}%"
    )

    print(
        "Median control-joint torque-RMS rate     : "
        f"{comparison['control_median_tau_rms_outside_pct'].median():.2f}%"
    )

    print(
        "Median J3-control RMS separation         : "
        f"{comparison['j3_minus_control_tau_rms_pct_points'].median():.2f} percentage points"
    )

    print(
        "Median J3 torque-peak outside rate       : "
        f"{comparison['j3_tau_peak_outside_pct'].median():.2f}%"
    )

    print(
        "Median control-joint torque-peak rate    : "
        f"{comparison['control_median_tau_peak_outside_pct'].median():.2f}%"
    )

    print(
        "Median J3-control peak separation        : "
        f"{comparison['j3_minus_control_tau_peak_pct_points'].median():.2f} percentage points"
    )


# =============================================================================
# 17. OUTPUTS
# =============================================================================

print(
    f"\nAll-joint cycle features saved to:\n"
    f"{OUTPUT_CYCLE_FEATURES}"
)

print(
    f"\nJoint degradation summary saved to:\n"
    f"{OUTPUT_JOINT_SUMMARY}"
)

print(
    f"\nJ3 localization summary saved to:\n"
    f"{OUTPUT_LOCALIZATION}"
)

print(
    f"\nLate pre-fault joint summary saved to:\n"
    f"{OUTPUT_LATE_WINDOWS}"
)

print("\n" + "=" * 112)
print("PHASE 04.4 JOINT-SPECIFIC DEGRADATION LOCALIZATION COMPLETE")
print("=" * 112)