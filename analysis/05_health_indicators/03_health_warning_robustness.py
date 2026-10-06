from pathlib import Path
from itertools import product

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
    / "me_ad_multijoint_health_indicator.csv"
)

OUTPUT_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_warning_sensitivity_results.csv"
)

OUTPUT_OPERATION = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_warning_robustness_by_operation.csv"
)

OUTPUT_CONFIG = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_warning_robustness_by_configuration.csv"
)


# =============================================================================
# 2. SENSITIVITY GRID
# =============================================================================

THRESHOLD_QUANTILES = [
    0.05,
    0.10,
    0.20,
]

WINDOW_SIZES = [
    25,
    50,
    100,
]

PERSISTENCE_LEVELS = [
    0.60,
    0.70,
    0.80,
]

CONSECUTIVE_WINDOWS = [
    2,
    3,
]

CONFIGURATIONS = list(
    product(
        THRESHOLD_QUANTILES,
        WINDOW_SIZES,
        PERSISTENCE_LEVELS,
        CONSECUTIVE_WINDOWS,
    )
)


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


print("=" * 120)
print("ME-AD PHASE 05.3 — HEALTH WARNING ROBUSTNESS & FALSE-WARNING ANALYSIS")
print("=" * 120)

print(
    f"\nCycles loaded              : "
    f"{len(health):,}"
)

print(
    f"Operations                 : "
    f"{health['operation_code'].nunique()}"
)

print(
    f"Sensitivity configurations : "
    f"{len(CONFIGURATIONS)}"
)


# =============================================================================
# 4. HELPER — BUILD COMPLETE WINDOWS
# =============================================================================

def build_windows(
    group,
    start_cycle,
    end_cycle,
    window_size,
    threshold,
    persistence,
):

    data = group[
        (
            group["cycle_index"]
            >= start_cycle
        )
        &
        (
            group["cycle_index"]
            < end_cycle
        )
    ].copy()

    if len(data) == 0:
        return pd.DataFrame()

    # Anchor windows to the requested search start rather than absolute
    # cycle multiples. This avoids mixing known healthy and post-healthy
    # observations in the same window.
    data[
        "window_id"
    ] = (
        (
            data["cycle_index"]
            - start_cycle
        )
        // window_size
    )

    records = []

    for window_id, window in data.groupby(
        "window_id"
    ):

        expected_start = (
            start_cycle
            +
            int(window_id)
            * window_size
        )

        expected_end = (
            expected_start
            +
            window_size
            - 1
        )

        # Structural exclusions may remove cycles.
        # Only complete contiguous windows are admitted.
        expected_cycles = np.arange(
            expected_start,
            expected_end + 1,
        )

        observed_cycles = np.sort(
            window[
                "cycle_index"
            ].to_numpy()
        )

        if (
            len(observed_cycles)
            != window_size
        ):
            continue

        if not np.array_equal(
            observed_cycles,
            expected_cycles,
        ):
            continue

        fraction_below = (
            window[
                "health_index"
            ]
            .lt(threshold)
            .mean()
        )

        records.append(
            {
                "window_id":
                    int(window_id),

                "window_start_cycle":
                    expected_start,

                "window_end_cycle":
                    expected_end,

                "median_health":
                    window[
                        "health_index"
                    ].median(),

                "fraction_below_threshold":
                    fraction_below,

                "candidate_warning":
                    (
                        fraction_below
                        >= persistence
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# 5. HELPER — FIRST SUSTAINED RUN
# =============================================================================

def first_sustained_warning(
    windows,
    required_consecutive,
    window_size,
):

    if len(windows) == 0:
        return np.nan

    windows = windows.sort_values(
        "window_start_cycle"
    ).reset_index(
        drop=True
    )

    for i in range(
        len(windows)
        - required_consecutive
        + 1
    ):

        run = windows.iloc[
            i:
            i + required_consecutive
        ]

        if not run[
            "candidate_warning"
        ].all():
            continue

        starts = run[
            "window_start_cycle"
        ].to_numpy()

        expected_starts = (
            starts[0]
            +
            np.arange(
                required_consecutive
            )
            * window_size
        )

        if np.array_equal(
            starts,
            expected_starts,
        ):
            return int(
                starts[0]
            )

    return np.nan


# =============================================================================
# 6. RUN SENSITIVITY GRID
# =============================================================================

result_records = []

total_configs = len(
    CONFIGURATIONS
)

for config_number, (
    threshold_quantile,
    window_size,
    persistence,
    required_consecutive,
) in enumerate(
    CONFIGURATIONS,
    start=1,
):

    print(
        f"Running configuration "
        f"{config_number:02d}/{total_configs}: "
        f"q={threshold_quantile:.0%}, "
        f"window={window_size}, "
        f"persistence={persistence:.0%}, "
        f"consecutive={required_consecutive}"
    )

    for operation_code, group in health.groupby(
        "operation_code"
    ):

        group = group.sort_values(
            "cycle_index"
        ).copy()

        training = group[
            group[
                "benchmark_health_region"
            ] == "training_healthy"
        ]

        test_healthy = group[
            group[
                "benchmark_health_region"
            ] == "test_healthy"
        ]

        faulty = group[
            group[
                "benchmark_health_region"
            ] == "faulty_evident"
        ]

        if (
            len(training) == 0
            or len(test_healthy) == 0
            or len(faulty) == 0
        ):
            continue

        threshold = (
            training[
                "health_index"
            ]
            .quantile(
                threshold_quantile
            )
        )

        fault_boundary = int(
            faulty[
                "cycle_index"
            ].min()
        )

        # -------------------------------------------------------------
        # FALSE-WARNING CHECK
        # -------------------------------------------------------------
        #
        # We evaluate all known healthy cycles 0–69 cycle-by-cycle.
        #
        # There are only 70 known healthy cycles, so 100-cycle windows
        # cannot be formed here. Rather than silently treating that as
        # "no false warning", we record healthy-window evaluation
        # availability explicitly.

        known_healthy = group[
            group[
                "benchmark_health_region"
            ].isin(
                [
                    "training_healthy",
                    "test_healthy",
                ]
            )
        ].copy()

        healthy_below_fraction = (
            known_healthy[
                "health_index"
            ]
            .lt(threshold)
            .mean()
        )

        healthy_windows = build_windows(
            group=group,
            start_cycle=0,
            end_cycle=70,
            window_size=window_size,
            threshold=threshold,
            persistence=persistence,
        )

        healthy_window_evaluation_available = (
            len(healthy_windows) > 0
        )

        false_warning_cycle = (
            first_sustained_warning(
                windows=healthy_windows,
                required_consecutive=required_consecutive,
                window_size=window_size,
            )
            if healthy_window_evaluation_available
            else np.nan
        )

        false_warning_detected = (
            pd.notna(
                false_warning_cycle
            )
        )

        # -------------------------------------------------------------
        # POST-HEALTHY WARNING SEARCH
        # -------------------------------------------------------------
        #
        # Search begins at cycle 70.
        # This explicitly prevents windows from mixing the known healthy
        # benchmark with the intermediate region.

        warning_windows = build_windows(
            group=group,
            start_cycle=70,
            end_cycle=fault_boundary,
            window_size=window_size,
            threshold=threshold,
            persistence=persistence,
        )

        warning_cycle = (
            first_sustained_warning(
                windows=warning_windows,
                required_consecutive=required_consecutive,
                window_size=window_size,
            )
        )

        warning_detected = (
            pd.notna(
                warning_cycle
            )
        )

        warning_lead = (
            fault_boundary
            - warning_cycle
            if warning_detected
            else np.nan
        )

        result_records.append(
            {
                "configuration":
                    config_number,

                "threshold_quantile":
                    threshold_quantile,

                "window_size":
                    window_size,

                "persistence":
                    persistence,

                "required_consecutive_windows":
                    required_consecutive,

                "operation_code":
                    operation_code,

                "health_threshold":
                    threshold,

                "known_healthy_cycles":
                    len(known_healthy),

                "healthy_below_threshold_fraction":
                    healthy_below_fraction,

                "healthy_window_evaluation_available":
                    healthy_window_evaluation_available,

                "false_warning_detected":
                    false_warning_detected,

                "false_warning_cycle":
                    false_warning_cycle,

                "warning_detected":
                    warning_detected,

                "warning_cycle":
                    warning_cycle,

                "evident_fault_start_cycle":
                    fault_boundary,

                "warning_lead_cycles":
                    warning_lead,
            }
        )


results = pd.DataFrame(
    result_records
)

results.to_csv(
    OUTPUT_RESULTS,
    index=False,
)


# =============================================================================
# 7. OPERATION-LEVEL ROBUSTNESS
# =============================================================================

operation_records = []

for operation_code, group in results.groupby(
    "operation_code"
):

    detections = int(
        group[
            "warning_detected"
        ].sum()
    )

    false_warnings = int(
        group[
            "false_warning_detected"
        ].sum()
    )

    evaluated_false_warning_configs = int(
        group[
            "healthy_window_evaluation_available"
        ].sum()
    )

    detected = group[
        group[
            "warning_detected"
        ]
    ]

    operation_records.append(
        {
            "operation_code":
                operation_code,

            "configurations":
                len(group),

            "detections":
                detections,

            "detection_rate_pct":
                (
                    100
                    * detections
                    / len(group)
                ),

            "false_warning_evaluable_configs":
                evaluated_false_warning_configs,

            "false_warnings":
                false_warnings,

            "false_warning_rate_pct":
                (
                    100
                    * false_warnings
                    / evaluated_false_warning_configs
                    if evaluated_false_warning_configs > 0
                    else np.nan
                ),

            "median_healthy_below_threshold_pct":
                (
                    100
                    * group[
                        "healthy_below_threshold_fraction"
                    ].median()
                ),

            "median_warning_cycle":
                (
                    detected[
                        "warning_cycle"
                    ].median()
                    if len(detected) > 0
                    else np.nan
                ),

            "warning_cycle_iqr":
                (
                    detected[
                        "warning_cycle"
                    ].quantile(0.75)
                    -
                    detected[
                        "warning_cycle"
                    ].quantile(0.25)
                    if len(detected) > 0
                    else np.nan
                ),

            "median_warning_lead":
                (
                    detected[
                        "warning_lead_cycles"
                    ].median()
                    if len(detected) > 0
                    else np.nan
                ),

            "min_warning_lead":
                (
                    detected[
                        "warning_lead_cycles"
                    ].min()
                    if len(detected) > 0
                    else np.nan
                ),

            "max_warning_lead":
                (
                    detected[
                        "warning_lead_cycles"
                    ].max()
                    if len(detected) > 0
                    else np.nan
                ),
        }
    )


operation_summary = pd.DataFrame(
    operation_records
)


def robustness_class(row):

    detection_rate = (
        row[
            "detection_rate_pct"
        ]
    )

    false_rate = (
        row[
            "false_warning_rate_pct"
        ]
    )

    if (
        detection_rate >= 75
        and (
            pd.isna(false_rate)
            or false_rate == 0
        )
    ):
        return "robust"

    if (
        detection_rate >= 25
        and (
            pd.isna(false_rate)
            or false_rate <= 10
        )
    ):
        return "moderate"

    return "weak"


operation_summary[
    "robustness_class"
] = operation_summary.apply(
    robustness_class,
    axis=1,
)

operation_summary.to_csv(
    OUTPUT_OPERATION,
    index=False,
)


# =============================================================================
# 8. CONFIGURATION-LEVEL SUMMARY
# =============================================================================

configuration_summary = (
    results
    .groupby(
        [
            "configuration",
            "threshold_quantile",
            "window_size",
            "persistence",
            "required_consecutive_windows",
        ]
    )
    .agg(
        operations=(
            "operation_code",
            "count",
        ),

        detections=(
            "warning_detected",
            "sum",
        ),

        false_warning_evaluable_operations=(
            "healthy_window_evaluation_available",
            "sum",
        ),

        false_warnings=(
            "false_warning_detected",
            "sum",
        ),

        median_warning_lead=(
            "warning_lead_cycles",
            "median",
        ),

        median_healthy_below_threshold_fraction=(
            "healthy_below_threshold_fraction",
            "median",
        ),
    )
    .reset_index()
)

configuration_summary[
    "detection_rate_pct"
] = (
    100
    * configuration_summary[
        "detections"
    ]
    / configuration_summary[
        "operations"
    ]
)

configuration_summary[
    "false_warning_rate_pct"
] = (
    100
    * configuration_summary[
        "false_warnings"
    ]
    / configuration_summary[
        "false_warning_evaluable_operations"
    ].replace(
        0,
        np.nan,
    )
)

configuration_summary[
    "median_healthy_below_threshold_pct"
] = (
    100
    * configuration_summary[
        "median_healthy_below_threshold_fraction"
    ]
)

configuration_summary.to_csv(
    OUTPUT_CONFIG,
    index=False,
)


# =============================================================================
# 9. PRINT OPERATION ROBUSTNESS
# =============================================================================

print(
    "\nOPERATION-LEVEL HEALTH WARNING ROBUSTNESS"
)
print("-" * 120)

print(
    operation_summary[
        [
            "operation_code",
            "detections",
            "configurations",
            "detection_rate_pct",
            "false_warning_evaluable_configs",
            "false_warnings",
            "false_warning_rate_pct",
            "median_healthy_below_threshold_pct",
            "median_warning_cycle",
            "warning_cycle_iqr",
            "median_warning_lead",
            "min_warning_lead",
            "max_warning_lead",
            "robustness_class",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 10. ROBUSTNESS COUNTS
# =============================================================================

print(
    "\nROBUSTNESS CLASSIFICATION"
)
print("-" * 120)

for classification in [
    "robust",
    "moderate",
    "weak",
]:

    operations = (
        operation_summary.loc[
            operation_summary[
                "robustness_class"
            ] == classification,
            "operation_code",
        ]
        .tolist()
    )

    print(
        f"{classification.upper():8s}: "
        f"{len(operations)} operations"
    )

    if operations:
        print(
            ", ".join(
                operations
            )
        )


# =============================================================================
# 11. MOST STABLE WARNING LOCATIONS
# =============================================================================

detected_operations = (
    operation_summary[
        operation_summary[
            "detections"
        ] >= 5
    ]
    .sort_values(
        [
            "detection_rate_pct",
            "warning_cycle_iqr",
        ],
        ascending=[
            False,
            True,
        ],
    )
)


print(
    "\nMOST STABLE WARNING LOCATIONS"
)
print("-" * 120)

print(
    detected_operations[
        [
            "operation_code",
            "detection_rate_pct",
            "median_warning_cycle",
            "warning_cycle_iqr",
            "median_warning_lead",
            "false_warning_rate_pct",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 12. STRICT CONFIGURATION
# =============================================================================

strict = results[
    (
        results[
            "threshold_quantile"
        ] == 0.05
    )
    &
    (
        results[
            "window_size"
        ] == 100
    )
    &
    (
        results[
            "persistence"
        ] == 0.80
    )
    &
    (
        results[
            "required_consecutive_windows"
        ] == 3
    )
].copy()


print(
    "\nSTRICT CONFIGURATION"
)
print(
    "threshold=5th percentile | "
    "window=100 | persistence=80% | "
    "consecutive=3"
)
print("-" * 120)

print(
    strict[
        [
            "operation_code",
            "health_threshold",
            "healthy_below_threshold_fraction",
            "healthy_window_evaluation_available",
            "false_warning_detected",
            "warning_detected",
            "warning_cycle",
            "warning_lead_cycles",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 13. BEST CONFIGURATIONS WITHOUT OBSERVED FALSE WARNINGS
# =============================================================================

safe_configs = configuration_summary[
    (
        configuration_summary[
            "false_warnings"
        ] == 0
    )
].copy()

safe_configs = safe_configs.sort_values(
    [
        "detections",
        "median_warning_lead",
    ],
    ascending=[
        False,
        False,
    ],
)


print(
    "\nTOP CONFIGURATIONS WITH NO OBSERVED HEALTHY-REGION FALSE WARNING"
)
print("-" * 120)

print(
    safe_configs[
        [
            "configuration",
            "threshold_quantile",
            "window_size",
            "persistence",
            "required_consecutive_windows",
            "detections",
            "false_warning_evaluable_operations",
            "false_warnings",
            "median_warning_lead",
            "median_healthy_below_threshold_pct",
        ]
    ]
    .head(10)
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 14. OUTPUTS
# =============================================================================

print(
    f"\nFull sensitivity results saved to:\n"
    f"{OUTPUT_RESULTS}"
)

print(
    f"\nOperation robustness summary saved to:\n"
    f"{OUTPUT_OPERATION}"
)

print(
    f"\nConfiguration summary saved to:\n"
    f"{OUTPUT_CONFIG}"
)

print("\n" + "=" * 120)
print("PHASE 05.3 HEALTH WARNING ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 120)