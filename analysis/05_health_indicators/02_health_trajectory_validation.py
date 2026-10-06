from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
    / "me_ad_multijoint_health_indicator.csv"
)

OUTPUT_OPERATION_TRENDS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_trajectory_validation.csv"
)

OUTPUT_WINDOWS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_warning_windows.csv"
)

OUTPUT_WARNINGS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_warning_summary.csv"
)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

N_SEQUENCE_BINS = 20

WARNING_WINDOW = 50

# At least 70% of cycles in a window must be below the health threshold.
WARNING_PERSISTENCE = 0.70

# Require three consecutive abnormal windows.
REQUIRED_CONSECUTIVE_WINDOWS = 3

# Conservative lower-tail threshold from independent training-healthy cycles.
HEALTH_THRESHOLD_QUANTILE = 0.05


# =============================================================================
# 3. LOAD DATA
# =============================================================================

health = pd.read_csv(
    INPUT_FILE,
    dtype={"operation_code": str},
)

health = health.sort_values(
    [
        "operation_code",
        "cycle_index",
    ]
).reset_index(
    drop=True
)


print("=" * 116)
print("ME-AD PHASE 05.2 — HEALTH TRAJECTORY VALIDATION")
print("=" * 116)

print(
    f"\nCycles loaded : {len(health):,}"
)

print(
    f"Operations    : "
    f"{health['operation_code'].nunique()}"
)


# =============================================================================
# 4. OPERATION-SPECIFIC INDEPENDENT HEALTH THRESHOLDS
# =============================================================================

threshold_records = []

for operation_code, group in health.groupby(
    "operation_code"
):

    training_healthy = group[
        group[
            "benchmark_health_region"
        ] == "training_healthy"
    ]

    if len(training_healthy) == 0:

        raise ValueError(
            f"No training-healthy cycles for "
            f"{operation_code}"
        )

    threshold = (
        training_healthy[
            "health_index"
        ]
        .quantile(
            HEALTH_THRESHOLD_QUANTILE
        )
    )

    threshold_records.append(
        {
            "operation_code":
                operation_code,

            "training_healthy_cycles":
                len(training_healthy),

            "health_warning_threshold":
                threshold,
        }
    )


thresholds = pd.DataFrame(
    threshold_records
)


print(
    "\nOPERATION-SPECIFIC HEALTH WARNING THRESHOLDS"
)
print("-" * 116)

print(
    thresholds
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 5. 20-BIN HEALTH TRAJECTORIES
# =============================================================================

trajectory_records = []

for operation_code, group in health.groupby(
    "operation_code"
):

    group = group.copy()

    group[
        "sequence_bin"
    ] = pd.cut(
        group[
            "sequence_position"
        ],
        bins=np.linspace(
            0,
            1,
            N_SEQUENCE_BINS + 1,
        ),
        include_lowest=True,
        labels=False,
    )

    group[
        "sequence_bin"
    ] = (
        group[
            "sequence_bin"
        ]
        + 1
    )

    for sequence_bin, window in group.groupby(
        "sequence_bin",
        observed=True,
    ):

        trajectory_records.append(
            {
                "operation_code":
                    operation_code,

                "sequence_bin":
                    int(sequence_bin),

                "cycles":
                    len(window),

                "median_sequence_position":
                    window[
                        "sequence_position"
                    ].median(),

                "median_health":
                    window[
                        "health_index"
                    ].median(),

                "health_q25":
                    window[
                        "health_index"
                    ].quantile(0.25),

                "health_q75":
                    window[
                        "health_index"
                    ].quantile(0.75),

                "median_j3_score":
                    window[
                        "j3_target_score"
                    ].median(),

                "median_system_score":
                    window[
                        "multijoint_system_score"
                    ].median(),

                "median_localization_contrast":
                    window[
                        "j3_localization_contrast"
                    ].median(),
            }
        )


trajectory = pd.DataFrame(
    trajectory_records
)


# =============================================================================
# 6. TREND AND MONOTONICITY METRICS
# =============================================================================

trend_records = []

for operation_code, group in trajectory.groupby(
    "operation_code"
):

    group = group.sort_values(
        "sequence_bin"
    ).reset_index(
        drop=True
    )

    rho, p_value = spearmanr(
        group[
            "median_sequence_position"
        ],
        group[
            "median_health"
        ],
    )

    health_values = group[
        "median_health"
    ].to_numpy()

    differences = np.diff(
        health_values
    )

    decreasing_steps = (
        differences < 0
    ).sum()

    increasing_steps = (
        differences > 0
    ).sum()

    unchanged_steps = (
        differences == 0
    ).sum()

    total_steps = len(
        differences
    )

    decreasing_pct = (
        100
        * decreasing_steps
        / total_steps
        if total_steps > 0
        else np.nan
    )

    first_health = (
        health_values[0]
    )

    last_health = (
        health_values[-1]
    )

    health_drop = (
        first_health
        - last_health
    )

    # Maximum temporary recovery from one bin to the next.
    max_upward_step = (
        differences.max()
        if len(differences) > 0
        else np.nan
    )

    # Largest deterioration between adjacent bins.
    max_downward_step = (
        differences.min()
        if len(differences) > 0
        else np.nan
    )

    trend_records.append(
        {
            "operation_code":
                operation_code,

            "spearman_rho":
                rho,

            "spearman_p_value":
                p_value,

            "decreasing_steps":
                int(decreasing_steps),

            "increasing_steps":
                int(increasing_steps),

            "unchanged_steps":
                int(unchanged_steps),

            "decreasing_step_pct":
                decreasing_pct,

            "first_bin_health":
                first_health,

            "last_bin_health":
                last_health,

            "health_drop_points":
                health_drop,

            "max_temporary_recovery":
                max_upward_step,

            "largest_single_bin_drop":
                max_downward_step,
        }
    )


trend_summary = pd.DataFrame(
    trend_records
)


# =============================================================================
# 7. CLASSIFY TRAJECTORY STRENGTH
# =============================================================================

def classify_trend(row):

    rho = row[
        "spearman_rho"
    ]

    decreasing_pct = row[
        "decreasing_step_pct"
    ]

    if (
        rho <= -0.80
        and decreasing_pct >= 70
    ):
        return "strong_progressive"

    if (
        rho <= -0.60
        and decreasing_pct >= 55
    ):
        return "moderate_progressive"

    if rho < 0:
        return "weak_or_irregular_decline"

    return "no_progressive_decline"


trend_summary[
    "trajectory_class"
] = trend_summary.apply(
    classify_trend,
    axis=1,
)

trend_summary.to_csv(
    OUTPUT_OPERATION_TRENDS,
    index=False,
)


print(
    "\nOPERATION-LEVEL HEALTH TRAJECTORY VALIDATION"
)
print("-" * 116)

print(
    trend_summary[
        [
            "operation_code",
            "spearman_rho",
            "spearman_p_value",
            "decreasing_step_pct",
            "first_bin_health",
            "last_bin_health",
            "health_drop_points",
            "max_temporary_recovery",
            "trajectory_class",
        ]
    ]
    .sort_values(
        "spearman_rho"
    )
    .round(4)
    .to_string(
        index=False
    )
)


# =============================================================================
# 8. PRE-FAULT HEALTH-WARNING WINDOWS
# =============================================================================

window_records = []

for operation_code, group in health.groupby(
    "operation_code"
):

    group = group.sort_values(
        "cycle_index"
    ).copy()

    threshold = float(
        thresholds.loc[
            thresholds[
                "operation_code"
            ] == operation_code,
            "health_warning_threshold",
        ].iloc[0]
    )

    faulty = group[
        group[
            "benchmark_health_region"
        ] == "faulty_evident"
    ]

    if len(faulty) == 0:
        raise ValueError(
            f"No evident-fault region for "
            f"{operation_code}"
        )

    fault_boundary = int(
        faulty[
            "cycle_index"
        ].min()
    )

    # Exclude training/reference healthy regions from warning search.
    #
    # Search starts after cycle 69, i.e. after both known healthy
    # benchmark sections.
    pre_fault = group[
        (
            group["cycle_index"] >= 70
        )
        &
        (
            group["cycle_index"] < fault_boundary
        )
    ].copy()

    pre_fault[
        "warning_window_id"
    ] = (
        pre_fault[
            "cycle_index"
        ]
        // WARNING_WINDOW
    )

    for window_id, window in pre_fault.groupby(
        "warning_window_id"
    ):

        if len(window) < WARNING_WINDOW:
            continue

        below_threshold = (
            window[
                "health_index"
            ]
            < threshold
        )

        fraction_below = (
            below_threshold.mean()
        )

        candidate_warning = (
            fraction_below
            >= WARNING_PERSISTENCE
        )

        window_records.append(
            {
                "operation_code":
                    operation_code,

                "warning_window_id":
                    int(window_id),

                "window_start_cycle":
                    int(
                        window[
                            "cycle_index"
                        ].min()
                    ),

                "window_end_cycle":
                    int(
                        window[
                            "cycle_index"
                        ].max()
                    ),

                "cycles":
                    len(window),

                "health_threshold":
                    threshold,

                "median_health":
                    window[
                        "health_index"
                    ].median(),

                "fraction_below_threshold":
                    fraction_below,

                "candidate_warning":
                    candidate_warning,

                "evident_fault_start_cycle":
                    fault_boundary,

                "median_cycles_before_fault":
                    (
                        fault_boundary
                        -
                        window[
                            "cycle_index"
                        ].median()
                    ),
            }
        )


warning_windows = pd.DataFrame(
    window_records
)

warning_windows.to_csv(
    OUTPUT_WINDOWS,
    index=False,
)


# =============================================================================
# 9. SUSTAINED WARNING DETECTION
# =============================================================================

warning_records = []

for operation_code, group in warning_windows.groupby(
    "operation_code"
):

    group = group.sort_values(
        "window_start_cycle"
    ).reset_index(
        drop=True
    )

    fault_boundary = int(
        group[
            "evident_fault_start_cycle"
        ].iloc[0]
    )

    detected = False
    warning_cycle = np.nan

    for i in range(
        len(group)
        - REQUIRED_CONSECUTIVE_WINDOWS
        + 1
    ):

        run = group.iloc[
            i:
            i + REQUIRED_CONSECUTIVE_WINDOWS
        ]

        all_candidate = (
            run[
                "candidate_warning"
            ].all()
        )

        starts = run[
            "window_start_cycle"
        ].to_numpy()

        expected_starts = (
            starts[0]
            +
            np.arange(
                REQUIRED_CONSECUTIVE_WINDOWS
            )
            * WARNING_WINDOW
        )

        consecutive = np.array_equal(
            starts,
            expected_starts,
        )

        if (
            all_candidate
            and consecutive
        ):

            detected = True

            warning_cycle = int(
                starts[0]
            )

            break

    if detected:

        lead = (
            fault_boundary
            - warning_cycle
        )

    else:

        lead = np.nan

    warning_records.append(
        {
            "operation_code":
                operation_code,

            "warning_detected":
                detected,

            "health_warning_cycle":
                warning_cycle,

            "evident_fault_start_cycle":
                fault_boundary,

            "health_warning_lead_cycles":
                lead,
        }
    )


warning_summary = pd.DataFrame(
    warning_records
)

warning_summary = warning_summary.merge(
    thresholds,
    on="operation_code",
    how="left",
    validate="one_to_one",
)

warning_summary = warning_summary.merge(
    trend_summary[
        [
            "operation_code",
            "spearman_rho",
            "decreasing_step_pct",
            "trajectory_class",
        ]
    ],
    on="operation_code",
    how="left",
    validate="one_to_one",
)

warning_summary.to_csv(
    OUTPUT_WARNINGS,
    index=False,
)


print(
    "\nSUSTAINED HEALTH-WARNING RESULTS"
)
print("-" * 116)

print(
    warning_summary[
        [
            "operation_code",
            "health_warning_threshold",
            "warning_detected",
            "health_warning_cycle",
            "evident_fault_start_cycle",
            "health_warning_lead_cycles",
            "spearman_rho",
            "decreasing_step_pct",
            "trajectory_class",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 10. WARNING SUMMARY
# =============================================================================

detected = warning_summary[
    warning_summary[
        "warning_detected"
    ] == True
]


print(
    "\nHEALTH-WARNING SUMMARY"
)
print("-" * 116)

print(
    f"Operations with sustained warning : "
    f"{len(detected)}/"
    f"{len(warning_summary)}"
)

if len(detected) > 0:

    print(
        f"Median warning lead               : "
        f"{detected['health_warning_lead_cycles'].median():.1f} cycles"
    )

    print(
        f"Minimum warning lead              : "
        f"{detected['health_warning_lead_cycles'].min():.0f} cycles"
    )

    print(
        f"Maximum warning lead              : "
        f"{detected['health_warning_lead_cycles'].max():.0f} cycles"
    )


# =============================================================================
# 11. TRAJECTORY CLASS COUNTS
# =============================================================================

print(
    "\nTRAJECTORY CLASSIFICATION COUNTS"
)
print("-" * 116)

print(
    trend_summary[
        "trajectory_class"
    ]
    .value_counts()
    .to_string()
)


# =============================================================================
# 12. J3 LOCALIZATION EVOLUTION
# =============================================================================

localization_summary = (
    trajectory
    .groupby(
        "sequence_bin"
    )
    .agg(
        median_j3_score=(
            "median_j3_score",
            "median",
        ),

        median_system_score=(
            "median_system_score",
            "median",
        ),

        median_localization_contrast=(
            "median_localization_contrast",
            "median",
        ),

        median_health=(
            "median_health",
            "median",
        ),
    )
    .reset_index()
)


print(
    "\nPOOLED SEQUENCE-BIN EVOLUTION"
)
print("-" * 116)

print(
    localization_summary
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 13. OUTPUTS
# =============================================================================

print(
    f"\nTrajectory validation saved to:\n"
    f"{OUTPUT_OPERATION_TRENDS}"
)

print(
    f"\nWarning windows saved to:\n"
    f"{OUTPUT_WINDOWS}"
)

print(
    f"\nWarning summary saved to:\n"
    f"{OUTPUT_WARNINGS}"
)

print("\n" + "=" * 116)
print("PHASE 05.2 HEALTH TRAJECTORY VALIDATION COMPLETE")
print("=" * 116)