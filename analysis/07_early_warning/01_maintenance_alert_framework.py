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

CONVERGENCE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_anomaly_convergence_cycles.csv"
)

IF_TIMING_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_if_health_warning_timing_comparison.csv"
)

OUTPUT_CYCLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_alert_states.csv"
)

OUTPUT_OPERATIONS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_alert_summary.csv"
)

OUTPUT_TRANSITIONS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_alert_transitions.csv"
)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

df = pd.read_csv(
    CONVERGENCE_FILE,
    dtype={"operation_code": str},
)

timing = pd.read_csv(
    IF_TIMING_FILE,
    dtype={"operation_code": str},
)

df = df.sort_values(
    ["operation_code", "cycle_index"]
).reset_index(drop=True)


print("=" * 120)
print("ME-AD PHASE 07.1 — MULTI-SIGNAL MAINTENANCE ALERT FRAMEWORK")
print("=" * 120)

print(f"\nCycles loaded     : {len(df):,}")
print(f"Operations        : {df['operation_code'].nunique()}")


# =============================================================================
# 3. PREPARE WARNING TIMING
# =============================================================================

timing_keep = timing[
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
    ]
].copy()

df = df.drop(
    columns=["diagnostic_group"],
    errors="ignore",
)

df = df.merge(
    timing_keep,
    on="operation_code",
    how="left",
    validate="many_to_one",
)


# =============================================================================
# 4. ACTIVE WARNING STATES
# =============================================================================

# A warning becomes active from its first sustained warning cycle onward.
#
# IMPORTANT:
# The evident-fault boundary is NOT used to create the operational alert.
# It is retained only for retrospective evaluation.

df["if_warning_active"] = (
    df["if_warning_detected"].fillna(False)
    &
    (
        df["cycle_index"]
        >= df["if_warning_cycle"]
    )
)

df["health_warning_active"] = (
    df["health_warning_detected"].fillna(False)
    &
    (
        df["cycle_index"]
        >= df["health_warning_cycle"]
    )
)


# =============================================================================
# 5. OPERATIONAL MAINTENANCE ALERT
# =============================================================================

# NORMAL
#   Neither persistent warning is active.
#
# WATCH
#   Persistent IF warning only.
#   Statistical deviation exists, but physics-informed degradation
#   has not yet independently confirmed it.
#
# WARNING
#   Persistent health warning only.
#   Structured degradation exists, but IF has not independently
#   confirmed it.
#
# CRITICAL
#   Both independent warning systems are active.
#
# "CRITICAL" here means highest analytical maintenance-alert tier.
# It does NOT mean immediate safety failure.

conditions = [
    (
        ~df["if_warning_active"]
        &
        ~df["health_warning_active"]
    ),
    (
        df["if_warning_active"]
        &
        ~df["health_warning_active"]
    ),
    (
        ~df["if_warning_active"]
        &
        df["health_warning_active"]
    ),
    (
        df["if_warning_active"]
        &
        df["health_warning_active"]
    ),
]

choices = [
    "NORMAL",
    "WATCH",
    "WARNING",
    "CRITICAL",
]

df["maintenance_alert"] = np.select(
    conditions,
    choices,
    default="NORMAL",
)


# =============================================================================
# 6. RETROSPECTIVE FAULT CONTEXT
# =============================================================================

df["is_evident_fault_region"] = (
    df["cycle_index"]
    >= df["evident_fault_start_cycle"]
)


# =============================================================================
# 7. ALERT SEVERITY
# =============================================================================

severity_map = {
    "NORMAL": 0,
    "WATCH": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}

df["alert_severity"] = (
    df["maintenance_alert"]
    .map(severity_map)
)


# =============================================================================
# 8. IDENTIFY FIRST OCCURRENCE OF EACH ALERT STATE
# =============================================================================

summary_records = []

for operation_code, group in df.groupby(
    "operation_code"
):

    group = group.sort_values(
        "cycle_index"
    )

    metadata = group.iloc[0]

    record = {
        "operation_code":
            operation_code,

        "diagnostic_group":
            metadata["diagnostic_group"],

        "evident_fault_start_cycle":
            metadata["evident_fault_start_cycle"],

        "if_warning_detected":
            metadata["if_warning_detected"],

        "health_warning_detected":
            metadata["health_warning_detected"],
    }

    for state in [
        "WATCH",
        "WARNING",
        "CRITICAL",
    ]:

        state_rows = group[
            group["maintenance_alert"] == state
        ]

        if len(state_rows) > 0:

            first_cycle = int(
                state_rows[
                    "cycle_index"
                ].min()
            )

            lead = (
                metadata[
                    "evident_fault_start_cycle"
                ]
                - first_cycle
            )

        else:

            first_cycle = np.nan
            lead = np.nan

        record[
            f"first_{state.lower()}_cycle"
        ] = first_cycle

        record[
            f"{state.lower()}_lead_cycles"
        ] = lead

    summary_records.append(record)


summary = pd.DataFrame(
    summary_records
)


# =============================================================================
# 9. OPERATIONAL RESPONSE CATEGORY
# =============================================================================

def response_category(row):

    if pd.notna(
        row["first_critical_cycle"]
    ):
        return "convergent_degradation"

    if pd.notna(
        row["first_warning_cycle"]
    ):
        return "health_confirmed_only"

    if pd.notna(
        row["first_watch_cycle"]
    ):
        return "statistical_watch_only"

    return "no_persistent_alert"


summary[
    "maintenance_response_category"
] = summary.apply(
    response_category,
    axis=1,
)


# =============================================================================
# 10. TRANSITION TABLE
# =============================================================================

transition_records = []

for operation_code, group in df.groupby(
    "operation_code"
):

    group = group.sort_values(
        "cycle_index"
    ).copy()

    group[
        "previous_alert"
    ] = (
        group[
            "maintenance_alert"
        ]
        .shift(1)
    )

    transitions = group[
        (
            group[
                "maintenance_alert"
            ]
            != group[
                "previous_alert"
            ]
        )
        &
        group[
            "previous_alert"
        ].notna()
    ]

    for _, row in transitions.iterrows():

        transition_records.append(
            {
                "operation_code":
                    operation_code,

                "cycle_index":
                    int(
                        row[
                            "cycle_index"
                        ]
                    ),

                "from_alert":
                    row[
                        "previous_alert"
                    ],

                "to_alert":
                    row[
                        "maintenance_alert"
                    ],

                "health_index":
                    row[
                        "health_index"
                    ],

                "if_threshold_ratio":
                    row[
                        "if_threshold_ratio"
                    ],

                "benchmark_health_region":
                    row[
                        "benchmark_health_region"
                    ],
            }
        )


transitions = pd.DataFrame(
    transition_records
)


# =============================================================================
# 11. SAVE OUTPUTS
# =============================================================================

cycle_columns = [
    "operation_code",
    "cycle_index",
    "benchmark_health_region",
    "diagnostic_group",
    "health_index",
    "health_warning_threshold",
    "if_threshold_ratio",
    "if_anomaly_flag",
    "if_warning_active",
    "health_warning_active",
    "maintenance_alert",
    "alert_severity",
    "is_evident_fault_region",
]

df[
    cycle_columns
].to_csv(
    OUTPUT_CYCLES,
    index=False,
)

summary.to_csv(
    OUTPUT_OPERATIONS,
    index=False,
)

transitions.to_csv(
    OUTPUT_TRANSITIONS,
    index=False,
)


# =============================================================================
# 12. PRINT OPERATION SUMMARY
# =============================================================================

print(
    "\nOPERATION-LEVEL MAINTENANCE ALERT SUMMARY"
)
print("-" * 120)

print(
    summary[
        [
            "operation_code",
            "diagnostic_group",
            "first_watch_cycle",
            "watch_lead_cycles",
            "first_warning_cycle",
            "warning_lead_cycles",
            "first_critical_cycle",
            "critical_lead_cycles",
            "maintenance_response_category",
        ]
    ]
    .sort_values(
        [
            "diagnostic_group",
            "operation_code",
        ]
    )
    .round(1)
    .to_string(
        index=False
    )
)


# =============================================================================
# 13. RESPONSE CATEGORY COUNTS
# =============================================================================

print(
    "\nMAINTENANCE RESPONSE CATEGORIES"
)
print("-" * 120)

print(
    summary[
        "maintenance_response_category"
    ]
    .value_counts()
    .to_string()
)


# =============================================================================
# 14. DIAGNOSTIC-GROUP SUMMARY
# =============================================================================

group_summary = (
    summary
    .groupby(
        "diagnostic_group"
    )
    .agg(
        operations=(
            "operation_code",
            "count",
        ),

        operations_with_watch=(
            "first_watch_cycle",
            lambda x: x.notna().sum(),
        ),

        operations_with_warning=(
            "first_warning_cycle",
            lambda x: x.notna().sum(),
        ),

        operations_with_critical=(
            "first_critical_cycle",
            lambda x: x.notna().sum(),
        ),

        median_watch_lead=(
            "watch_lead_cycles",
            "median",
        ),

        median_warning_lead=(
            "warning_lead_cycles",
            "median",
        ),

        median_critical_lead=(
            "critical_lead_cycles",
            "median",
        ),
    )
    .reset_index()
)


print(
    "\nDIAGNOSTIC-GROUP MAINTENANCE RESPONSE"
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
# 15. ALERT DISTRIBUTION BY BENCHMARK REGION
# =============================================================================

alert_region = (
    df
    .groupby(
        [
            "benchmark_health_region",
            "maintenance_alert",
        ]
    )
    .size()
    .reset_index(
        name="cycles"
    )
)

region_totals = (
    alert_region
    .groupby(
        "benchmark_health_region"
    )[
        "cycles"
    ]
    .transform(
        "sum"
    )
)

alert_region[
    "region_pct"
] = (
    100
    * alert_region[
        "cycles"
    ]
    / region_totals
)


print(
    "\nALERT DISTRIBUTION BY BENCHMARK REGION"
)
print("-" * 120)

print(
    alert_region
    .sort_values(
        [
            "benchmark_health_region",
            "maintenance_alert",
        ]
    )
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 16. TRANSITIONS
# =============================================================================

print(
    "\nALERT TRANSITIONS"
)
print("-" * 120)

if len(transitions) > 0:

    print(
        transitions
        .round(3)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No alert transitions identified."
    )


# =============================================================================
# 17. CRITICAL CONVERGENCE SUMMARY
# =============================================================================

critical = summary[
    summary[
        "first_critical_cycle"
    ].notna()
]

print(
    "\nCONVERGENT CRITICAL-ALERT SUMMARY"
)
print("-" * 120)

print(
    f"Operations reaching CRITICAL : "
    f"{len(critical)}/"
    f"{len(summary)}"
)

if len(critical) > 0:

    print(
        f"Median CRITICAL lead         : "
        f"{critical['critical_lead_cycles'].median():.1f} cycles"
    )

    print(
        f"Minimum CRITICAL lead        : "
        f"{critical['critical_lead_cycles'].min():.1f} cycles"
    )

    print(
        f"Maximum CRITICAL lead        : "
        f"{critical['critical_lead_cycles'].max():.1f} cycles"
    )


# =============================================================================
# 18. OUTPUT PATHS
# =============================================================================

print(
    f"\nCycle alert states saved to:\n"
    f"{OUTPUT_CYCLES}"
)

print(
    f"\nOperation alert summary saved to:\n"
    f"{OUTPUT_OPERATIONS}"
)

print(
    f"\nAlert transitions saved to:\n"
    f"{OUTPUT_TRANSITIONS}"
)

print("\n" + "=" * 120)
print("PHASE 07.1 MULTI-SIGNAL MAINTENANCE ALERT FRAMEWORK COMPLETE")
print("=" * 120)