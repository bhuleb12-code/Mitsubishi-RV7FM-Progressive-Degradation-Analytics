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
    / "me_ad_family1_speed_effects.csv"
)


# =============================================================================
# 2. MATCHED FAMILY-1 OPERATIONS
# =============================================================================

operation_pairs = {
    "layout_0": ("1050", "1100"),
    "layout_1": ("1051", "1101"),
    "layout_2": ("1052", "1102"),
    "layout_3": ("1053", "1103"),
}


# =============================================================================
# 3. LOAD HEALTHY BASELINE
# =============================================================================

baseline = pd.read_csv(
    INPUT_FILE,
    dtype={"operation_code": str},
)

print("=" * 100)
print("ME-AD PHASE 03.2 — FAMILY-1 HEALTHY SPEED EFFECTS")
print("=" * 100)

print(
    "\nComparison:"
    "\n50% commanded speed → 100% commanded speed"
)

print(
    "\nMatched operation pairs:"
)

for layout, pair in operation_pairs.items():
    print(
        f"  {layout}: "
        f"{pair[0]} -> {pair[1]}"
    )


# =============================================================================
# 4. BUILD MATCHED COMPARISONS
# =============================================================================

records = []

metrics = [
    "median_q_range",
    "median_dq_rms",
    "median_ddq_rms",
    "median_tau_abs",
    "median_tau_rms",
    "median_tau_peak",
    "median_tau_std",
]

for layout, (
    low_speed_op,
    high_speed_op,
) in operation_pairs.items():

    for joint in range(1, 7):

        low = baseline[
            (baseline["operation_code"] == low_speed_op)
            &
            (baseline["joint"] == joint)
        ]

        high = baseline[
            (baseline["operation_code"] == high_speed_op)
            &
            (baseline["joint"] == joint)
        ]

        if len(low) != 1 or len(high) != 1:
            raise ValueError(
                f"Missing or duplicate baseline "
                f"for {layout}, J{joint}"
            )

        low = low.iloc[0]
        high = high.iloc[0]

        record = {
            "layout":
                layout,

            "joint":
                joint,

            "operation_50":
                low_speed_op,

            "operation_100":
                high_speed_op,
        }

        for metric in metrics:

            value_50 = low[metric]
            value_100 = high[metric]

            absolute_change = (
                value_100 - value_50
            )

            if (
                pd.isna(value_50)
                or value_50 == 0
            ):
                percent_change = np.nan
                ratio = np.nan

            else:
                percent_change = (
                    100
                    * absolute_change
                    / value_50
                )

                ratio = (
                    value_100
                    / value_50
                )

            record[
                f"{metric}_50"
            ] = value_50

            record[
                f"{metric}_100"
            ] = value_100

            record[
                f"{metric}_change"
            ] = absolute_change

            record[
                f"{metric}_pct_change"
            ] = percent_change

            record[
                f"{metric}_ratio"
            ] = ratio

        records.append(record)


effects = pd.DataFrame(records)

effects.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =============================================================================
# 5. J3 SPEED EFFECT
# =============================================================================

j3 = effects[
    effects["joint"] == 3
].copy()

print(
    "\nJ3 — EFFECT OF INCREASING COMMANDED SPEED"
)
print("-" * 100)

print(
    j3[
        [
            "layout",
            "operation_50",
            "operation_100",
            "median_q_range_50",
            "median_q_range_100",
            "median_dq_rms_pct_change",
            "median_ddq_rms_pct_change",
            "median_tau_abs_pct_change",
            "median_tau_rms_pct_change",
            "median_tau_peak_pct_change",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 6. ALL-JOINT SPEED RESPONSE
# =============================================================================

joint_effect = (
    effects
    .groupby("joint")
    .agg(
        median_velocity_change_pct=(
            "median_dq_rms_pct_change",
            "median",
        ),

        median_acceleration_change_pct=(
            "median_ddq_rms_pct_change",
            "median",
        ),

        median_tau_abs_change_pct=(
            "median_tau_abs_pct_change",
            "median",
        ),

        median_tau_rms_change_pct=(
            "median_tau_rms_pct_change",
            "median",
        ),

        median_tau_peak_change_pct=(
            "median_tau_peak_pct_change",
            "median",
        ),
    )
    .reset_index()
)

print(
    "\nALL-JOINT MEDIAN RESPONSE TO SPEED INCREASE"
)
print("-" * 100)

print(
    joint_effect
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 7. LAYOUT CONSISTENCY FOR J3
# =============================================================================

print(
    "\nJ3 — SPEED RESPONSE CONSISTENCY ACROSS LAYOUTS"
)
print("-" * 100)

for metric in [
    "median_dq_rms_pct_change",
    "median_ddq_rms_pct_change",
    "median_tau_abs_pct_change",
    "median_tau_rms_pct_change",
]:

    values = j3[
        metric
    ].dropna()

    print(
        f"{metric:35s} | "
        f"min={values.min():8.3f}% | "
        f"median={values.median():8.3f}% | "
        f"max={values.max():8.3f}%"
    )


# =============================================================================
# 8. MOTION VS EFFORT COUPLING
# =============================================================================

print(
    "\nJ3 — MOTION / EFFORT CHANGE BY MATCHED PAIR"
)
print("-" * 100)

print(
    j3[
        [
            "layout",
            "median_dq_rms_pct_change",
            "median_ddq_rms_pct_change",
            "median_tau_rms_pct_change",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# =============================================================================
# 9. CHECK POSITION-RANGE STABILITY
# =============================================================================

j3[
    "q_range_pct_change"
] = (
    100
    * (
        j3["median_q_range_100"]
        -
        j3["median_q_range_50"]
    )
    /
    j3["median_q_range_50"]
)

print(
    "\nJ3 — POSITION RANGE STABILITY"
)
print("-" * 100)

print(
    j3[
        [
            "layout",
            "operation_50",
            "operation_100",
            "median_q_range_50",
            "median_q_range_100",
            "q_range_pct_change",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 10. OUTPUT
# =============================================================================

print(
    f"\nDetailed speed-effect table saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 100)
print("PHASE 03.2 FAMILY-1 SPEED EFFECT ANALYSIS COMPLETE")
print("=" * 100)