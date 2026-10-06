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

OUTPUT_WINDOWS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_j3_onset_windows.csv"
)

OUTPUT_ONSET = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_j3_onset_summary.csv"
)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

# Number of consecutive cycles considered as one local diagnostic window.
WINDOW_SIZE = 25

# Require at least this fraction of cycles in a window to exceed
# the healthy-validation limit before calling that feature persistently abnormal.
PERSISTENCE_FRACTION = 0.60

# Require at least this many J3 effort features to show persistent abnormality.
MIN_ABNORMAL_FEATURES = 2


EFFORT_FEATURES = [
    "j3_tau_abs_mean",
    "j3_tau_rms",
    "j3_tau_peak",
    "j3_tau_std",
]

MOTION_FEATURES = [
    "j3_q_range",
    "j3_dq_rms",
    "j3_ddq_rms",
]


# =============================================================================
# 3. LOAD PHASE 04.1 OUTPUT
# =============================================================================

cycles = pd.read_csv(
    INPUT_FILE,
    dtype={"operation_code": str},
)

print("=" * 108)
print("ME-AD PHASE 04.2 — HIGH-RESOLUTION J3 DEGRADATION ONSET ANALYSIS")
print("=" * 108)

print(
    f"\nCycle rows loaded : {len(cycles):,}"
)

print(
    f"Operations        : "
    f"{cycles['operation_code'].nunique()}"
)


# =============================================================================
# 4. KEEP STRUCTURALLY USABLE CYCLES
# =============================================================================

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

print(
    f"Structurally usable cycles : "
    f"{len(usable):,}"
)


# =============================================================================
# 5. BUILD EMPIRICAL HEALTHY-VALIDATION LIMITS
# =============================================================================

# IMPORTANT:
# These limits are NOT fitted on the first 20 training cycles.
# They are calibrated from independent test-healthy cycles 20–69.

limit_records = []

for operation_code, group in usable.groupby(
    "operation_code"
):

    healthy_validation = group[
        group[
            "benchmark_health_region"
        ] == "test_healthy"
    ]

    if len(healthy_validation) == 0:
        raise ValueError(
            f"No test-healthy cycles found "
            f"for operation {operation_code}"
        )

    record = {
        "operation_code":
            operation_code,

        "healthy_validation_cycles":
            len(healthy_validation),
    }

    for feature in EFFORT_FEATURES + MOTION_FEATURES:

        values = healthy_validation[
            feature
        ].dropna()

        # Empirical healthy envelope.
        # We use observed validation extrema rather than assuming
        # Gaussian behaviour.
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


# =============================================================================
# 6. ATTACH HEALTHY LIMITS
# =============================================================================

usable = usable.merge(
    limits,
    on="operation_code",
    how="left",
    validate="many_to_one",
)


# =============================================================================
# 7. CYCLE-LEVEL HEALTHY-ENVELOPE DEPARTURE
# =============================================================================

# Effort features are treated two-sided.
# Degradation may alter torque upward OR downward depending on trajectory,
# direction, friction regime, and controller compensation.

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


# Motion features are also checked two-sided, but they are diagnostic context.
# They are NOT used by themselves to declare degradation.

for feature in MOTION_FEATURES:

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
# 8. DISTANCE TO EVIDENT-FAULT BOUNDARY
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
            f"No faulty-evident cycles for "
            f"{operation_code}"
        )

    boundary = int(
        faulty[
            "cycle_index"
        ].min()
    )

    boundary_records.append(
        {
            "operation_code":
                operation_code,

            "evident_fault_start_cycle":
                boundary,
        }
    )


boundaries = pd.DataFrame(
    boundary_records
)

usable = usable.merge(
    boundaries,
    on="operation_code",
    how="left",
    validate="many_to_one",
)

usable[
    "cycles_before_evident_fault"
] = (
    usable[
        "evident_fault_start_cycle"
    ]
    -
    usable[
        "cycle_index"
    ]
)


# =============================================================================
# 9. BUILD NON-OVERLAPPING 25-CYCLE WINDOWS
# =============================================================================

window_records = []

for operation_code, group in usable.groupby(
    "operation_code"
):

    group = group.sort_values(
        "cycle_index"
    ).copy()

    # Analyse only cycles BEFORE the official faulty region.
    pre_fault = group[
        group[
            "cycle_index"
        ]
        <
        group[
            "evident_fault_start_cycle"
        ]
    ].copy()

    if len(pre_fault) == 0:
        continue

    pre_fault[
        "window_id"
    ] = (
        pre_fault[
            "cycle_index"
        ]
        // WINDOW_SIZE
    )

    for window_id, window in pre_fault.groupby(
        "window_id"
    ):

        if len(window) < WINDOW_SIZE:
            continue

        record = {
            "operation_code":
                operation_code,

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

            "cycles":
                len(window),

            "median_cycles_before_fault":
                float(
                    window[
                        "cycles_before_evident_fault"
                    ].median()
                ),
        }

        persistent_effort_count = 0

        for feature in EFFORT_FEATURES:

            flag_col = (
                f"{feature}_outside_healthy"
            )

            fraction = (
                window[
                    flag_col
                ].mean()
            )

            persistent = (
                fraction
                >= PERSISTENCE_FRACTION
            )

            record[
                f"{feature}_outside_fraction"
            ] = fraction

            record[
                f"{feature}_persistent"
            ] = persistent

            if persistent:
                persistent_effort_count += 1

        persistent_motion_count = 0

        for feature in MOTION_FEATURES:

            flag_col = (
                f"{feature}_outside_healthy"
            )

            fraction = (
                window[
                    flag_col
                ].mean()
            )

            persistent = (
                fraction
                >= PERSISTENCE_FRACTION
            )

            record[
                f"{feature}_outside_fraction"
            ] = fraction

            record[
                f"{feature}_persistent"
            ] = persistent

            if persistent:
                persistent_motion_count += 1

        record[
            "persistent_effort_features"
        ] = persistent_effort_count

        record[
            "persistent_motion_features"
        ] = persistent_motion_count

        record[
            "candidate_degradation_window"
        ] = (
            persistent_effort_count
            >= MIN_ABNORMAL_FEATURES
        )

        window_records.append(
            record
        )


windows = pd.DataFrame(
    window_records
)

windows.to_csv(
    OUTPUT_WINDOWS,
    index=False,
)


# =============================================================================
# 10. REQUIRE PERSISTENCE ACROSS ADJACENT WINDOWS
# =============================================================================

# A single abnormal 25-cycle block may be transient.
# For onset, require TWO consecutive candidate windows.

onset_records = []

for operation_code, group in windows.groupby(
    "operation_code"
):

    group = group.sort_values(
        "window_start_cycle"
    ).reset_index(
        drop=True
    )

    boundary = int(
        boundaries.loc[
            boundaries[
                "operation_code"
            ] == operation_code,
            "evident_fault_start_cycle",
        ].iloc[0]
    )

    onset_window = None

    for i in range(
        len(group) - 1
    ):

        current = group.iloc[i]
        next_window = group.iloc[i + 1]

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
            + WINDOW_SIZE
        )

        if (
            bool(
                current[
                    "candidate_degradation_window"
                ]
            )
            and
            bool(
                next_window[
                    "candidate_degradation_window"
                ]
            )
            and
            consecutive
        ):
            onset_window = current
            break

    if onset_window is None:

        onset_records.append(
            {
                "operation_code":
                    operation_code,

                "evident_fault_start_cycle":
                    boundary,

                "onset_detected":
                    False,

                "candidate_onset_cycle":
                    np.nan,

                "early_warning_lead_cycles":
                    np.nan,

                "persistent_effort_features_at_onset":
                    np.nan,

                "persistent_motion_features_at_onset":
                    np.nan,
            }
        )

    else:

        onset_cycle = int(
            onset_window[
                "window_start_cycle"
            ]
        )

        lead = (
            boundary
            -
            onset_cycle
        )

        onset_records.append(
            {
                "operation_code":
                    operation_code,

                "evident_fault_start_cycle":
                    boundary,

                "onset_detected":
                    True,

                "candidate_onset_cycle":
                    onset_cycle,

                "early_warning_lead_cycles":
                    lead,

                "persistent_effort_features_at_onset":
                    int(
                        onset_window[
                            "persistent_effort_features"
                        ]
                    ),

                "persistent_motion_features_at_onset":
                    int(
                        onset_window[
                            "persistent_motion_features"
                        ]
                    ),
            }
        )


onset = pd.DataFrame(
    onset_records
)

onset.to_csv(
    OUTPUT_ONSET,
    index=False,
)


# =============================================================================
# 11. HEALTHY FALSE-POSITIVE CHECK
# =============================================================================

healthy_check = usable[
    usable[
        "benchmark_health_region"
    ] == "test_healthy"
].copy()

healthy_flag_records = []

for operation_code, group in healthy_check.groupby(
    "operation_code"
):

    record = {
        "operation_code":
            operation_code,

        "cycles":
            len(group),
    }

    for feature in EFFORT_FEATURES:

        record[
            f"{feature}_outside_pct"
        ] = (
            100
            * group[
                f"{feature}_outside_healthy"
            ].mean()
        )

    healthy_flag_records.append(
        record
    )


healthy_flags = pd.DataFrame(
    healthy_flag_records
)


print(
    "\nHEALTHY-VALIDATION ENVELOPE CHECK"
)
print("-" * 108)

print(
    healthy_flags
    .round(2)
    .to_string(index=False)
)


# =============================================================================
# 12. ONSET SUMMARY
# =============================================================================

print(
    "\nCANDIDATE J3 DEGRADATION ONSET BY OPERATION"
)
print("-" * 108)

print(
    onset
    .sort_values(
        "operation_code"
    )
    .to_string(
        index=False
    )
)


# =============================================================================
# 13. DETECTED-OPERATION SUMMARY
# =============================================================================

detected = onset[
    onset[
        "onset_detected"
    ] == True
].copy()

print(
    "\nEARLY-WARNING LEAD SUMMARY"
)
print("-" * 108)

if len(detected) == 0:

    print(
        "No operation satisfied the "
        "two-window persistence criterion."
    )

else:

    print(
        f"Operations with candidate onset : "
        f"{len(detected)}/"
        f"{len(onset)}"
    )

    print(
        f"Earliest warning lead           : "
        f"{detected['early_warning_lead_cycles'].max():.0f} cycles"
    )

    print(
        f"Median warning lead             : "
        f"{detected['early_warning_lead_cycles'].median():.1f} cycles"
    )

    print(
        f"Latest warning lead             : "
        f"{detected['early_warning_lead_cycles'].min():.0f} cycles"
    )


# =============================================================================
# 14. LAST WINDOWS BEFORE EVIDENT FAULT
# =============================================================================

print(
    "\nLAST THREE PRE-FAULT WINDOWS BY OPERATION"
)
print("-" * 108)

last_windows = (
    windows
    .sort_values(
        [
            "operation_code",
            "window_start_cycle",
        ]
    )
    .groupby(
        "operation_code"
    )
    .tail(3)
)

print(
    last_windows[
        [
            "operation_code",
            "window_start_cycle",
            "window_end_cycle",
            "median_cycles_before_fault",
            "persistent_effort_features",
            "persistent_motion_features",
            "candidate_degradation_window",
        ]
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# 15. OUTPUTS
# =============================================================================

print(
    f"\nWindow-level onset analysis saved to:\n"
    f"{OUTPUT_WINDOWS}"
)

print(
    f"\nOperation-level onset summary saved to:\n"
    f"{OUTPUT_ONSET}"
)

print("\n" + "=" * 108)
print("PHASE 04.2 HIGH-RESOLUTION ONSET ANALYSIS COMPLETE")
print("=" * 108)