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
    / "me_ad_j3_degradation_cycle_features.csv"
)

OUTPUT_CONFIG_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_j3_onset_sensitivity_results.csv"
)

OUTPUT_OPERATION_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_j3_onset_sensitivity_by_operation.csv"
)

OUTPUT_CONFIG_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_j3_onset_sensitivity_by_configuration.csv"
)


# =============================================================================
# 2. SENSITIVITY GRID
# =============================================================================

WINDOW_SIZES = [
    25,
    50,
    100,
]

PERSISTENCE_THRESHOLDS = [
    0.60,
    0.70,
    0.80,
]

MIN_EFFORT_FEATURES = [
    2,
    3,
    4,
]


EFFORT_FEATURES = [
    "j3_tau_abs_mean",
    "j3_tau_rms",
    "j3_tau_peak",
    "j3_tau_std",
]


# =============================================================================
# 3. LOAD PHASE 04.1 FEATURES
# =============================================================================

cycles = pd.read_csv(
    INPUT_FILE,
    dtype={"operation_code": str},
)

usable = cycles[
    cycles["structural_quality"]
    == "usable"
].copy()

usable = usable.sort_values(
    [
        "operation_code",
        "cycle_index",
    ]
)

print("=" * 112)
print("ME-AD PHASE 04.3 — J3 ONSET ROBUSTNESS & SENSITIVITY ANALYSIS")
print("=" * 112)

print(
    f"\nCycle rows loaded           : "
    f"{len(cycles):,}"
)

print(
    f"Structurally usable cycles  : "
    f"{len(usable):,}"
)

print(
    f"Operations                  : "
    f"{usable['operation_code'].nunique()}"
)

total_configurations = (
    len(WINDOW_SIZES)
    * len(PERSISTENCE_THRESHOLDS)
    * len(MIN_EFFORT_FEATURES)
)

print(
    f"Sensitivity configurations  : "
    f"{total_configurations}"
)


# =============================================================================
# 4. HEALTHY VALIDATION ENVELOPES
# =============================================================================

limit_records = []

for operation_code, group in usable.groupby(
    "operation_code"
):

    healthy = group[
        group[
            "benchmark_health_region"
        ] == "test_healthy"
    ]

    if len(healthy) == 0:
        raise ValueError(
            f"No test-healthy cycles for "
            f"{operation_code}"
        )

    record = {
        "operation_code":
            operation_code,
    }

    for feature in EFFORT_FEATURES:

        values = healthy[
            feature
        ].dropna()

        record[
            f"{feature}_healthy_min"
        ] = values.min()

        record[
            f"{feature}_healthy_max"
        ] = values.max()

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
# 5. CYCLE-LEVEL EFFORT DEPARTURE FLAGS
# =============================================================================

for feature in EFFORT_FEATURES:

    lower = usable[
        f"{feature}_healthy_min"
    ]

    upper = usable[
        f"{feature}_healthy_max"
    ]

    usable[
        f"{feature}_outside_healthy"
    ] = (
        (usable[feature] < lower)
        |
        (usable[feature] > upper)
    )


# =============================================================================
# 6. EVIDENT-FAULT BOUNDARIES
# =============================================================================

boundary_records = []

for operation_code, group in usable.groupby(
    "operation_code"
):

    faulty = group[
        group[
            "benchmark_health_region"
        ] == "faulty_evident"
    ]

    if len(faulty) == 0:
        raise ValueError(
            f"No faulty-evident region for "
            f"{operation_code}"
        )

    boundary_records.append(
        {
            "operation_code":
                operation_code,

            "evident_fault_start_cycle":
                int(
                    faulty[
                        "cycle_index"
                    ].min()
                ),
        }
    )


boundaries = pd.DataFrame(
    boundary_records
)


# =============================================================================
# 7. RUN ONE SENSITIVITY CONFIGURATION
# =============================================================================

def detect_onset(
    group,
    boundary,
    window_size,
    persistence_threshold,
    min_effort_features,
):

    pre_fault = group[
        group["cycle_index"]
        < boundary
    ].copy()

    if len(pre_fault) == 0:
        return None

    # Anchor windows to absolute cycle numbering.
    pre_fault[
        "window_id"
    ] = (
        pre_fault[
            "cycle_index"
        ]
        // window_size
    )

    window_records = []

    for window_id, window in pre_fault.groupby(
        "window_id"
    ):

        # Only complete windows.
        if len(window) < window_size:
            continue

        persistent_count = 0

        for feature in EFFORT_FEATURES:

            fraction = (
                window[
                    f"{feature}_outside_healthy"
                ].mean()
            )

            if (
                fraction
                >= persistence_threshold
            ):
                persistent_count += 1

        candidate = (
            persistent_count
            >= min_effort_features
        )

        window_records.append(
            {
                "window_id":
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

                "persistent_effort_features":
                    persistent_count,

                "candidate":
                    candidate,
            }
        )

    if len(window_records) < 2:
        return None

    windows = pd.DataFrame(
        window_records
    ).sort_values(
        "window_start_cycle"
    ).reset_index(
        drop=True
    )

    # Require TWO consecutive candidate windows.
    for i in range(
        len(windows) - 1
    ):

        current = windows.iloc[i]
        next_window = windows.iloc[i + 1]

        consecutive = (
            int(
                next_window[
                    "window_start_cycle"
                ]
            )
            ==
            int(
                current[
                    "window_start_cycle"
                ]
            )
            + window_size
        )

        if (
            bool(
                current[
                    "candidate"
                ]
            )
            and
            bool(
                next_window[
                    "candidate"
                ]
            )
            and
            consecutive
        ):

            onset_cycle = int(
                current[
                    "window_start_cycle"
                ]
            )

            return {
                "candidate_onset_cycle":
                    onset_cycle,

                "early_warning_lead_cycles":
                    boundary
                    - onset_cycle,

                "persistent_effort_features_at_onset":
                    int(
                        current[
                            "persistent_effort_features"
                        ]
                    ),
            }

    return None


# =============================================================================
# 8. RUN ALL 27 CONFIGURATIONS
# =============================================================================

results = []

configuration_number = 0

for window_size in WINDOW_SIZES:

    for persistence_threshold in PERSISTENCE_THRESHOLDS:

        for min_effort_features in MIN_EFFORT_FEATURES:

            configuration_number += 1

            print(
                f"Running configuration "
                f"{configuration_number:02d}/"
                f"{total_configurations}: "
                f"window={window_size}, "
                f"persistence="
                f"{persistence_threshold:.0%}, "
                f"features="
                f"{min_effort_features}"
            )

            for operation_code, group in usable.groupby(
                "operation_code"
            ):

                boundary = int(
                    boundaries.loc[
                        boundaries[
                            "operation_code"
                        ]
                        == operation_code,
                        "evident_fault_start_cycle",
                    ].iloc[0]
                )

                detection = detect_onset(
                    group=group,
                    boundary=boundary,
                    window_size=window_size,
                    persistence_threshold=(
                        persistence_threshold
                    ),
                    min_effort_features=(
                        min_effort_features
                    ),
                )

                record = {
                    "configuration_number":
                        configuration_number,

                    "operation_code":
                        operation_code,

                    "window_size":
                        window_size,

                    "persistence_threshold":
                        persistence_threshold,

                    "min_effort_features":
                        min_effort_features,

                    "evident_fault_start_cycle":
                        boundary,

                    "onset_detected":
                        detection is not None,
                }

                if detection is None:

                    record[
                        "candidate_onset_cycle"
                    ] = np.nan

                    record[
                        "early_warning_lead_cycles"
                    ] = np.nan

                    record[
                        "persistent_effort_features_at_onset"
                    ] = np.nan

                else:

                    record.update(
                        detection
                    )

                results.append(
                    record
                )


results = pd.DataFrame(
    results
)

results.to_csv(
    OUTPUT_CONFIG_RESULTS,
    index=False,
)


# =============================================================================
# 9. OPERATION-LEVEL ROBUSTNESS
# =============================================================================

operation_summary_records = []

for operation_code, group in results.groupby(
    "operation_code"
):

    detected = group[
        group["onset_detected"]
        == True
    ]

    detection_rate = (
        len(detected)
        /
        len(group)
    )

    record = {
        "operation_code":
            operation_code,

        "configurations":
            len(group),

        "detections":
            len(detected),

        "detection_rate_pct":
            100 * detection_rate,
    }

    if len(detected) > 0:

        record[
            "median_onset_cycle"
        ] = (
            detected[
                "candidate_onset_cycle"
            ].median()
        )

        record[
            "min_onset_cycle"
        ] = (
            detected[
                "candidate_onset_cycle"
            ].min()
        )

        record[
            "max_onset_cycle"
        ] = (
            detected[
                "candidate_onset_cycle"
            ].max()
        )

        record[
            "median_warning_lead"
        ] = (
            detected[
                "early_warning_lead_cycles"
            ].median()
        )

        record[
            "min_warning_lead"
        ] = (
            detected[
                "early_warning_lead_cycles"
            ].min()
        )

        record[
            "max_warning_lead"
        ] = (
            detected[
                "early_warning_lead_cycles"
            ].max()
        )

        record[
            "onset_cycle_iqr"
        ] = (
            detected[
                "candidate_onset_cycle"
            ].quantile(0.75)
            -
            detected[
                "candidate_onset_cycle"
            ].quantile(0.25)
        )

    else:

        record[
            "median_onset_cycle"
        ] = np.nan

        record[
            "min_onset_cycle"
        ] = np.nan

        record[
            "max_onset_cycle"
        ] = np.nan

        record[
            "median_warning_lead"
        ] = np.nan

        record[
            "min_warning_lead"
        ] = np.nan

        record[
            "max_warning_lead"
        ] = np.nan

        record[
            "onset_cycle_iqr"
        ] = np.nan

    operation_summary_records.append(
        record
    )


operation_summary = pd.DataFrame(
    operation_summary_records
)

operation_summary.to_csv(
    OUTPUT_OPERATION_SUMMARY,
    index=False,
)


# =============================================================================
# 10. CONFIGURATION-LEVEL SUMMARY
# =============================================================================

config_summary_records = []

for config_number, group in results.groupby(
    "configuration_number"
):

    detected = group[
        group["onset_detected"]
        == True
    ]

    record = {
        "configuration_number":
            int(config_number),

        "window_size":
            int(
                group[
                    "window_size"
                ].iloc[0]
            ),

        "persistence_threshold":
            float(
                group[
                    "persistence_threshold"
                ].iloc[0]
            ),

        "min_effort_features":
            int(
                group[
                    "min_effort_features"
                ].iloc[0]
            ),

        "operations_detected":
            len(detected),

        "detection_rate_pct":
            100
            * len(detected)
            / len(group),
    }

    if len(detected) > 0:

        record[
            "median_warning_lead"
        ] = (
            detected[
                "early_warning_lead_cycles"
            ].median()
        )

        record[
            "min_warning_lead"
        ] = (
            detected[
                "early_warning_lead_cycles"
            ].min()
        )

        record[
            "max_warning_lead"
        ] = (
            detected[
                "early_warning_lead_cycles"
            ].max()
        )

    else:

        record[
            "median_warning_lead"
        ] = np.nan

        record[
            "min_warning_lead"
        ] = np.nan

        record[
            "max_warning_lead"
        ] = np.nan

    config_summary_records.append(
        record
    )


config_summary = pd.DataFrame(
    config_summary_records
)

config_summary.to_csv(
    OUTPUT_CONFIG_SUMMARY,
    index=False,
)


# =============================================================================
# 11. PRINT OPERATION ROBUSTNESS
# =============================================================================

print(
    "\nOPERATION-LEVEL ONSET ROBUSTNESS"
)
print("-" * 112)

display_columns = [
    "operation_code",
    "detections",
    "configurations",
    "detection_rate_pct",
    "median_onset_cycle",
    "onset_cycle_iqr",
    "median_warning_lead",
    "min_warning_lead",
    "max_warning_lead",
]

print(
    operation_summary[
        display_columns
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 12. ROBUSTLY DETECTED OPERATIONS
# =============================================================================

robust = operation_summary[
    operation_summary[
        "detection_rate_pct"
    ] >= 75
].copy()

moderate = operation_summary[
    (
        operation_summary[
            "detection_rate_pct"
        ] >= 25
    )
    &
    (
        operation_summary[
            "detection_rate_pct"
        ] < 75
    )
].copy()

weak = operation_summary[
    operation_summary[
        "detection_rate_pct"
    ] < 25
].copy()


print(
    "\nROBUSTNESS CLASSIFICATION"
)
print("-" * 112)

print(
    f"Robust detection >=75% configs : "
    f"{len(robust)} operations"
)

if len(robust) > 0:
    print(
        ", ".join(
            robust[
                "operation_code"
            ].astype(str)
        )
    )

print(
    f"\nModerate detection 25-<75%     : "
    f"{len(moderate)} operations"
)

if len(moderate) > 0:
    print(
        ", ".join(
            moderate[
                "operation_code"
            ].astype(str)
        )
    )

print(
    f"\nWeak detection <25%            : "
    f"{len(weak)} operations"
)

if len(weak) > 0:
    print(
        ", ".join(
            weak[
                "operation_code"
            ].astype(str)
        )
    )


# =============================================================================
# 13. STRICTEST CONFIGURATION
# =============================================================================

strictest = results[
    (
        results["window_size"]
        == max(WINDOW_SIZES)
    )
    &
    (
        results[
            "persistence_threshold"
        ]
        == max(
            PERSISTENCE_THRESHOLDS
        )
    )
    &
    (
        results[
            "min_effort_features"
        ]
        == max(
            MIN_EFFORT_FEATURES
        )
    )
].copy()


print(
    "\nSTRICTEST CONFIGURATION"
)
print(
    "window=100 cycles | "
    "persistence=80% | "
    "required effort features=4/4"
)
print("-" * 112)

print(
    strictest[
        [
            "operation_code",
            "onset_detected",
            "candidate_onset_cycle",
            "early_warning_lead_cycles",
        ]
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# 14. MOST CONSISTENT ONSET LOCATIONS
# =============================================================================

consistent = operation_summary[
    operation_summary[
        "detections"
    ] >= 5
].copy()

consistent = consistent.sort_values(
    [
        "detection_rate_pct",
        "onset_cycle_iqr",
    ],
    ascending=[
        False,
        True,
    ],
)


print(
    "\nMOST CONSISTENT CANDIDATE ONSET LOCATIONS"
)
print("-" * 112)

if len(consistent) == 0:

    print(
        "No operation had enough detections "
        "for consistency assessment."
    )

else:

    print(
        consistent[
            [
                "operation_code",
                "detection_rate_pct",
                "median_onset_cycle",
                "onset_cycle_iqr",
                "median_warning_lead",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


# =============================================================================
# 15. OUTPUTS
# =============================================================================

print(
    f"\nFull sensitivity results saved to:\n"
    f"{OUTPUT_CONFIG_RESULTS}"
)

print(
    f"\nOperation robustness summary saved to:\n"
    f"{OUTPUT_OPERATION_SUMMARY}"
)

print(
    f"\nConfiguration summary saved to:\n"
    f"{OUTPUT_CONFIG_SUMMARY}"
)

print("\n" + "=" * 112)
print("PHASE 04.3 ONSET ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 112)