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

ANOMALY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_isolation_forest_cycle_scores.csv"
)

HEALTH_WARNING_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_warning_summary.csv"
)

OUTPUT_WINDOWS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_if_warning_windows.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_if_health_warning_timing_comparison.csv"
)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

WINDOW_SIZE = 50
ANOMALY_PERSISTENCE = 0.70
REQUIRED_CONSECUTIVE_WINDOWS = 3

# Known healthy region ends at cycle 69.
WARNING_SEARCH_START = 70

ROBUST_OPERATIONS = [
    "1050",
    "1051",
    "1052",
    "1053",
    "1101",
    "2500",
]

MODERATE_OPERATIONS = [
    "1100",
    "1102",
    "1103",
    "2000",
]

WEAK_OPERATIONS = [
    "2100",
    "2200",
    "2300",
    "2400",
    "3000",
    "3100",
]


# =============================================================================
# 3. LOAD DATA
# =============================================================================

anomaly = pd.read_csv(
    ANOMALY_FILE,
    dtype={"operation_code": str},
)

health_warning = pd.read_csv(
    HEALTH_WARNING_FILE,
    dtype={"operation_code": str},
)

anomaly = anomaly.sort_values(
    [
        "operation_code",
        "cycle_index",
    ]
).reset_index(
    drop=True
)


print("=" * 120)
print("ME-AD PHASE 06.3 — UNSUPERVISED EARLY-WARNING TIMING & LEAD ANALYSIS")
print("=" * 120)

print(
    f"\nAnomaly cycles loaded : "
    f"{len(anomaly):,}"
)

print(
    f"Operations            : "
    f"{anomaly['operation_code'].nunique()}"
)

print(
    f"Window size           : "
    f"{WINDOW_SIZE} cycles"
)

print(
    f"Anomaly persistence   : "
    f"{ANOMALY_PERSISTENCE:.0%}"
)

print(
    f"Consecutive windows   : "
    f"{REQUIRED_CONSECUTIVE_WINDOWS}"
)


# =============================================================================
# 4. DIAGNOSTIC GROUP
# =============================================================================

def diagnostic_group(
    operation_code
):

    if operation_code in ROBUST_OPERATIONS:
        return "robust"

    if operation_code in MODERATE_OPERATIONS:
        return "moderate"

    if operation_code in WEAK_OPERATIONS:
        return "weak"

    return "unclassified"


# =============================================================================
# 5. BUILD POST-HEALTHY WINDOWS
# =============================================================================

window_records = []

for operation_code, group in anomaly.groupby(
    "operation_code"
):

    group = group.sort_values(
        "cycle_index"
    ).copy()

    faulty = group[
        group[
            "benchmark_health_region"
        ] == "faulty_evident"
    ]

    if len(faulty) == 0:
        continue

    fault_boundary = int(
        faulty[
            "cycle_index"
        ].min()
    )

    prefault = group[
        (
            group[
                "cycle_index"
            ] >= WARNING_SEARCH_START
        )
        &
        (
            group[
                "cycle_index"
            ] < fault_boundary
        )
    ].copy()

    # Anchor windows at cycle 70.
    prefault[
        "warning_window_id"
    ] = (
        (
            prefault[
                "cycle_index"
            ]
            - WARNING_SEARCH_START
        )
        // WINDOW_SIZE
    )

    for window_id, window in prefault.groupby(
        "warning_window_id"
    ):

        expected_start = (
            WARNING_SEARCH_START
            +
            int(window_id)
            * WINDOW_SIZE
        )

        expected_end = (
            expected_start
            +
            WINDOW_SIZE
            - 1
        )

        expected_cycles = np.arange(
            expected_start,
            expected_end + 1,
        )

        observed_cycles = np.sort(
            window[
                "cycle_index"
            ].to_numpy()
        )

        # Only complete contiguous windows.
        if len(observed_cycles) != WINDOW_SIZE:
            continue

        if not np.array_equal(
            observed_cycles,
            expected_cycles,
        ):
            continue

        anomaly_fraction = (
            window[
                "if_anomaly_flag"
            ].mean()
        )

        median_score = (
            window[
                "if_anomaly_score"
            ].median()
        )

        median_threshold = (
            window[
                "if_anomaly_threshold"
            ].median()
        )

        median_ratio = (
            median_score
            /
            median_threshold
        )

        window_records.append(
            {
                "operation_code":
                    operation_code,

                "diagnostic_group":
                    diagnostic_group(
                        operation_code
                    ),

                "fault_boundary":
                    fault_boundary,

                "window_id":
                    int(window_id),

                "window_start_cycle":
                    expected_start,

                "window_end_cycle":
                    expected_end,

                "anomaly_fraction":
                    anomaly_fraction,

                "anomaly_rate_pct":
                    100
                    * anomaly_fraction,

                "median_if_score":
                    median_score,

                "median_if_threshold":
                    median_threshold,

                "median_if_threshold_ratio":
                    median_ratio,

                "candidate_if_warning":
                    (
                        anomaly_fraction
                        >= ANOMALY_PERSISTENCE
                    ),
            }
        )


windows = pd.DataFrame(
    window_records
)

windows.to_csv(
    OUTPUT_WINDOWS,
    index=False,
)


# =============================================================================
# 6. FIRST SUSTAINED IF WARNING
# =============================================================================

summary_records = []

for operation_code, group in anomaly.groupby(
    "operation_code"
):

    faulty = group[
        group[
            "benchmark_health_region"
        ] == "faulty_evident"
    ]

    if len(faulty) == 0:
        continue

    fault_boundary = int(
        faulty[
            "cycle_index"
        ].min()
    )

    operation_windows = windows[
        windows[
            "operation_code"
        ] == operation_code
    ].sort_values(
        "window_start_cycle"
    ).reset_index(
        drop=True
    )

    if_warning_cycle = np.nan

    if len(operation_windows) >= REQUIRED_CONSECUTIVE_WINDOWS:

        for i in range(
            len(operation_windows)
            - REQUIRED_CONSECUTIVE_WINDOWS
            + 1
        ):

            run = operation_windows.iloc[
                i:
                i + REQUIRED_CONSECUTIVE_WINDOWS
            ]

            if not run[
                "candidate_if_warning"
            ].all():
                continue

            starts = run[
                "window_start_cycle"
            ].to_numpy()

            expected_starts = (
                starts[0]
                +
                np.arange(
                    REQUIRED_CONSECUTIVE_WINDOWS
                )
                * WINDOW_SIZE
            )

            if np.array_equal(
                starts,
                expected_starts,
            ):

                if_warning_cycle = int(
                    starts[0]
                )

                break

    if_warning_detected = pd.notna(
        if_warning_cycle
    )

    if_warning_lead = (
        fault_boundary
        - if_warning_cycle
        if if_warning_detected
        else np.nan
    )

    summary_records.append(
        {
            "operation_code":
                operation_code,

            "diagnostic_group":
                diagnostic_group(
                    operation_code
                ),

            "evident_fault_start_cycle":
                fault_boundary,

            "if_warning_detected":
                if_warning_detected,

            "if_warning_cycle":
                if_warning_cycle,

            "if_warning_lead_cycles":
                if_warning_lead,
        }
    )


summary = pd.DataFrame(
    summary_records
)


# =============================================================================
# 7. PREPARE PHASE 05 HEALTH WARNING
# =============================================================================

# Support the column names created in Phase 05.2.
required_health_columns = [
    "operation_code",
    "warning_detected",
    "health_warning_cycle",
    "health_warning_lead_cycles",
]

missing_health_columns = [
    column
    for column in required_health_columns
    if column not in health_warning.columns
]

if missing_health_columns:

    raise ValueError(
        "Missing expected columns in Phase 05 health warning file:\n"
        + "\n".join(
            missing_health_columns
        )
    )


health_compare = health_warning[
    required_health_columns
].copy()

health_compare = health_compare.rename(
    columns={
        "warning_detected":
            "health_warning_detected",
    }
)


# =============================================================================
# 8. MERGE IF AND HEALTH WARNING TIMING
# =============================================================================

comparison = summary.merge(
    health_compare,
    on="operation_code",
    how="left",
    validate="one_to_one",
)


# =============================================================================
# 9. WARNING TIMING RELATIONSHIP
# =============================================================================

def timing_relationship(
    row
):

    if (
        not row[
            "if_warning_detected"
        ]
        and
        not row[
            "health_warning_detected"
        ]
    ):
        return "neither_warns"

    if (
        row[
            "if_warning_detected"
        ]
        and
        not row[
            "health_warning_detected"
        ]
    ):
        return "if_only"

    if (
        not row[
            "if_warning_detected"
        ]
        and
        row[
            "health_warning_detected"
        ]
    ):
        return "health_only"

    difference = (
        row[
            "if_warning_cycle"
        ]
        -
        row[
            "health_warning_cycle"
        ]
    )

    if difference < 0:
        return "if_earlier"

    if difference > 0:
        return "health_earlier"

    return "same_cycle"


comparison[
    "timing_relationship"
] = comparison.apply(
    timing_relationship,
    axis=1,
)


comparison[
    "if_minus_health_warning_cycles"
] = (
    comparison[
        "if_warning_cycle"
    ]
    -
    comparison[
        "health_warning_cycle"
    ]
)


comparison[
    "absolute_warning_timing_gap"
] = (
    comparison[
        "if_minus_health_warning_cycles"
    ].abs()
)


comparison.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# =============================================================================
# 10. PRINT WARNING COMPARISON
# =============================================================================

print(
    "\nIF VS PHYSICS-INFORMED HEALTH WARNING TIMING"
)
print("-" * 120)

print(
    comparison[
        [
            "operation_code",
            "diagnostic_group",
            "evident_fault_start_cycle",
            "health_warning_detected",
            "health_warning_cycle",
            "health_warning_lead_cycles",
            "if_warning_detected",
            "if_warning_cycle",
            "if_warning_lead_cycles",
            "if_minus_health_warning_cycles",
            "timing_relationship",
        ]
    ]
    .round(1)
    .to_string(
        index=False
    )
)


# =============================================================================
# 11. IF WARNING SUMMARY
# =============================================================================

detected_if = comparison[
    comparison[
        "if_warning_detected"
    ]
].copy()


print(
    "\nUNSUPERVISED IF WARNING SUMMARY"
)
print("-" * 120)

print(
    f"Operations with IF warning : "
    f"{len(detected_if)}/"
    f"{len(comparison)}"
)

if len(detected_if) > 0:

    print(
        f"Median IF warning lead     : "
        f"{detected_if['if_warning_lead_cycles'].median():.1f} cycles"
    )

    print(
        f"Minimum IF warning lead    : "
        f"{detected_if['if_warning_lead_cycles'].min():.1f} cycles"
    )

    print(
        f"Maximum IF warning lead    : "
        f"{detected_if['if_warning_lead_cycles'].max():.1f} cycles"
    )


# =============================================================================
# 12. AGREEMENT IN WHETHER A WARNING EXISTS
# =============================================================================

warning_presence_agreement = (
    comparison[
        "if_warning_detected"
    ]
    ==
    comparison[
        "health_warning_detected"
    ]
)

print(
    "\nWARNING-PRESENCE AGREEMENT"
)
print("-" * 120)

print(
    f"Same warning/no-warning decision : "
    f"{warning_presence_agreement.sum()}/"
    f"{len(comparison)} "
    f"({100 * warning_presence_agreement.mean():.2f}%)"
)


# =============================================================================
# 13. TIMING AGREEMENT WHEN BOTH METHODS WARN
# =============================================================================

both_warn = comparison[
    (
        comparison[
            "if_warning_detected"
        ]
    )
    &
    (
        comparison[
            "health_warning_detected"
        ]
    )
].copy()


print(
    "\nTIMING AGREEMENT WHEN BOTH METHODS WARN"
)
print("-" * 120)

print(
    f"Operations where both warn : "
    f"{len(both_warn)}"
)

if len(both_warn) > 0:

    print(
        f"Median absolute timing gap : "
        f"{both_warn['absolute_warning_timing_gap'].median():.1f} cycles"
    )

    print(
        f"Minimum absolute timing gap: "
        f"{both_warn['absolute_warning_timing_gap'].min():.1f} cycles"
    )

    print(
        f"Maximum absolute timing gap: "
        f"{both_warn['absolute_warning_timing_gap'].max():.1f} cycles"
    )

    print(
        "\nTiming relationship counts:"
    )

    print(
        both_warn[
            "timing_relationship"
        ]
        .value_counts()
        .to_string()
    )


# =============================================================================
# 14. DIAGNOSTIC-GROUP WARNING SUMMARY
# =============================================================================

group_summary = (
    comparison
    .groupby(
        "diagnostic_group"
    )
    .agg(
        operations=(
            "operation_code",
            "count",
        ),

        health_warnings=(
            "health_warning_detected",
            "sum",
        ),

        if_warnings=(
            "if_warning_detected",
            "sum",
        ),

        median_health_lead=(
            "health_warning_lead_cycles",
            "median",
        ),

        median_if_lead=(
            "if_warning_lead_cycles",
            "median",
        ),

        median_absolute_timing_gap=(
            "absolute_warning_timing_gap",
            "median",
        ),
    )
    .reset_index()
)


print(
    "\nDIAGNOSTIC-GROUP WARNING TIMING"
)
print("-" * 120)

print(
    group_summary
    .round(1)
    .to_string(
        index=False
    )
)


# =============================================================================
# 15. LAST THREE PRE-FAULT WINDOWS
# =============================================================================

print(
    "\nLAST THREE COMPLETE PRE-FAULT IF WINDOWS"
)
print("-" * 120)

last_three = (
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
    last_three[
        [
            "operation_code",
            "window_start_cycle",
            "window_end_cycle",
            "anomaly_rate_pct",
            "median_if_threshold_ratio",
            "candidate_if_warning",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 16. OUTPUTS
# =============================================================================

print(
    f"\nIF warning windows saved to:\n"
    f"{OUTPUT_WINDOWS}"
)

print(
    f"\nIF-health warning timing comparison saved to:\n"
    f"{OUTPUT_SUMMARY}"
)

print("\n" + "=" * 120)
print("PHASE 06.3 UNSUPERVISED EARLY-WARNING TIMING ANALYSIS COMPLETE")
print("=" * 120)