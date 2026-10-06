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

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_healthy_operating_baseline.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_family1_layout_effects.csv"
)


# =============================================================================
# 2. FAMILY-1 OPERATING GROUPS
# =============================================================================

speed_groups = {
    "50_percent": [
        "1050",
        "1051",
        "1052",
        "1053",
    ],
    "100_percent": [
        "1100",
        "1101",
        "1102",
        "1103",
    ],
}


# =============================================================================
# 3. LOAD HEALTHY BASELINE
# =============================================================================

baseline = pd.read_csv(
    INPUT_FILE,
    dtype={"operation_code": str},
)

print("=" * 104)
print("ME-AD PHASE 03.3 — FAMILY-1 HEALTHY TRAJECTORY / LAYOUT EFFECTS")
print("=" * 104)

print(
    "\nObjective:"
    "\nHold commanded speed constant and quantify "
    "healthy differences caused by layout / trajectory."
)


# =============================================================================
# 4. EXTRACT FAMILY-1 BASELINES
# =============================================================================

family1_records = []

for speed_group, operations in speed_groups.items():

    for operation_code in operations:

        for joint in range(1, 7):

            row = baseline[
                (baseline["operation_code"] == operation_code)
                &
                (baseline["joint"] == joint)
            ]

            if len(row) != 1:
                raise ValueError(
                    f"Expected one baseline row for "
                    f"{operation_code}, J{joint}; "
                    f"found {len(row)}."
                )

            row = row.iloc[0]

            family1_records.append(
                {
                    "speed_group":
                        speed_group,

                    "operation_code":
                        operation_code,

                    "joint":
                        joint,

                    "median_q_range":
                        row["median_q_range"],

                    "median_dq_rms":
                        row["median_dq_rms"],

                    "median_ddq_rms":
                        row["median_ddq_rms"],

                    "median_tau_abs":
                        row["median_tau_abs"],

                    "median_tau_rms":
                        row["median_tau_rms"],

                    "median_tau_peak":
                        row["median_tau_peak"],

                    "median_tau_std":
                        row["median_tau_std"],
                }
            )


family1 = pd.DataFrame(
    family1_records
)


# =============================================================================
# 5. WITHIN-SPEED LAYOUT VARIATION
# =============================================================================

summary_records = []

metrics = [
    "median_q_range",
    "median_dq_rms",
    "median_ddq_rms",
    "median_tau_abs",
    "median_tau_rms",
    "median_tau_peak",
]

for speed_group in speed_groups:

    for joint in range(1, 7):

        subset = family1[
            (family1["speed_group"] == speed_group)
            &
            (family1["joint"] == joint)
        ]

        record = {
            "speed_group":
                speed_group,

            "joint":
                joint,
        }

        for metric in metrics:

            values = subset[
                metric
            ].to_numpy()

            minimum = np.min(values)
            maximum = np.max(values)
            mean = np.mean(values)

            record[
                f"{metric}_min"
            ] = minimum

            record[
                f"{metric}_max"
            ] = maximum

            record[
                f"{metric}_range"
            ] = maximum - minimum

            if mean == 0:
                cv_pct = np.nan
                relative_range_pct = np.nan

            else:
                cv_pct = (
                    100
                    * np.std(
                        values,
                        ddof=0,
                    )
                    / mean
                )

                relative_range_pct = (
                    100
                    * (maximum - minimum)
                    / mean
                )

            record[
                f"{metric}_cv_pct"
            ] = cv_pct

            record[
                f"{metric}_relative_range_pct"
            ] = relative_range_pct

        summary_records.append(
            record
        )


layout_summary = pd.DataFrame(
    summary_records
)

layout_summary.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =============================================================================
# 6. J3 BASELINES ACROSS LAYOUTS
# =============================================================================

j3 = family1[
    family1["joint"] == 3
].copy()

print(
    "\nJ3 — HEALTHY BASELINE ACROSS LAYOUTS"
)
print("-" * 104)

print(
    j3[
        [
            "speed_group",
            "operation_code",
            "median_q_range",
            "median_dq_rms",
            "median_ddq_rms",
            "median_tau_abs",
            "median_tau_rms",
            "median_tau_peak",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 7. J3 LAYOUT-INDUCED VARIATION
# =============================================================================

j3_layout = layout_summary[
    layout_summary["joint"] == 3
].copy()

print(
    "\nJ3 — VARIATION ACROSS LAYOUTS AT CONSTANT SPEED"
)
print("-" * 104)

print(
    j3_layout[
        [
            "speed_group",

            "median_q_range_min",
            "median_q_range_max",
            "median_q_range_relative_range_pct",

            "median_dq_rms_min",
            "median_dq_rms_max",
            "median_dq_rms_relative_range_pct",

            "median_tau_rms_min",
            "median_tau_rms_max",
            "median_tau_rms_relative_range_pct",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 8. IDENTIFY LOWEST / HIGHEST J3 EFFORT OPERATIONS
# =============================================================================

print(
    "\nJ3 — LOWEST AND HIGHEST HEALTHY TORQUE RMS WITHIN EACH SPEED GROUP"
)
print("-" * 104)

for speed_group in speed_groups:

    subset = j3[
        j3["speed_group"] == speed_group
    ].copy()

    lowest = subset.loc[
        subset[
            "median_tau_rms"
        ].idxmin()
    ]

    highest = subset.loc[
        subset[
            "median_tau_rms"
        ].idxmax()
    ]

    difference_pct = (
        100
        * (
            highest["median_tau_rms"]
            -
            lowest["median_tau_rms"]
        )
        /
        lowest["median_tau_rms"]
    )

    print(
        f"{speed_group:12s} | "
        f"lowest={lowest['operation_code']} "
        f"({lowest['median_tau_rms']:.3f}) | "
        f"highest={highest['operation_code']} "
        f"({highest['median_tau_rms']:.3f}) | "
        f"difference={difference_pct:.2f}%"
    )


# =============================================================================
# 9. ALL-JOINT LAYOUT SENSITIVITY
# =============================================================================

all_joint_layout = (
    layout_summary
    .groupby("joint")
    .agg(
        median_q_layout_effect_pct=(
            "median_q_range_relative_range_pct",
            "median",
        ),

        median_velocity_layout_effect_pct=(
            "median_dq_rms_relative_range_pct",
            "median",
        ),

        median_acceleration_layout_effect_pct=(
            "median_ddq_rms_relative_range_pct",
            "median",
        ),

        median_tau_layout_effect_pct=(
            "median_tau_rms_relative_range_pct",
            "median",
        ),
    )
    .reset_index()
)

print(
    "\nALL-JOINT MEDIAN LAYOUT SENSITIVITY"
)
print("-" * 104)

print(
    all_joint_layout
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 10. J3 MOTION / EFFORT RANKING
# =============================================================================

print(
    "\nJ3 — OPERATION RANKING BY HEALTHY TORQUE RMS"
)
print("-" * 104)

ranking = (
    j3[
        [
            "speed_group",
            "operation_code",
            "median_q_range",
            "median_dq_rms",
            "median_tau_rms",
        ]
    ]
    .sort_values(
        [
            "speed_group",
            "median_tau_rms",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

print(
    ranking
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 11. OUTPUT
# =============================================================================

print(
    f"\nDetailed layout-effect table saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 104)
print("PHASE 03.3 FAMILY-1 LAYOUT EFFECT ANALYSIS COMPLETE")
print("=" * 104)