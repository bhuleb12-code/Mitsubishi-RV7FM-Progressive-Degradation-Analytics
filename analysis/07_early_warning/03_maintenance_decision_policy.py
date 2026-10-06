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

ALERT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_alert_states.csv"
)

PROBE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_diagnostic_probe_ranking.csv"
)

OUTPUT_CYCLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_decisions.csv"
)

OUTPUT_POLICY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_decision_policy.csv"
)

OUTPUT_OPERATION_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_decision_summary.csv"
)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

alerts = pd.read_csv(
    ALERT_FILE,
    dtype={"operation_code": str},
)

probes = pd.read_csv(
    PROBE_FILE,
    dtype={"operation_code": str},
)


print("=" * 120)
print("ME-AD PHASE 07.3 — MAINTENANCE DECISION POLICY")
print("=" * 120)

print(f"\nAlert-state cycles : {len(alerts):,}")
print(f"Operations         : {alerts['operation_code'].nunique()}")
print(f"Probe rankings     : {probes['operation_code'].nunique()}")


# =============================================================================
# 3. VALIDATE REQUIRED COLUMNS
# =============================================================================

def require_columns(
    frame,
    required,
    frame_name,
):

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:

        raise ValueError(
            f"{frame_name} is missing expected columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(frame.columns)
        )


require_columns(
    alerts,
    [
        "operation_code",
        "cycle_index",
        "benchmark_health_region",
        "diagnostic_group",
        "health_index",
        "if_threshold_ratio",
        "maintenance_alert",
        "alert_severity",
    ],
    "Maintenance alert-state table",
)

require_columns(
    probes,
    [
        "operation_code",
        "probe_rank",
        "probe_tier",
        "probe_utility_pct",
        "recommended_maintenance_use",
    ],
    "Diagnostic-probe ranking table",
)


# =============================================================================
# 4. MERGE PROBE QUALITY WITH ALERT STATES
# =============================================================================

probe_keep = probes[
    [
        "operation_code",
        "probe_rank",
        "probe_tier",
        "probe_utility_pct",
        "recommended_maintenance_use",
    ]
].copy()


df = alerts.merge(
    probe_keep,
    on="operation_code",
    how="left",
    validate="many_to_one",
)


if df["probe_tier"].isna().any():

    missing_operations = (
        df.loc[
            df["probe_tier"].isna(),
            "operation_code",
        ]
        .drop_duplicates()
        .tolist()
    )

    raise ValueError(
        "Probe ranking missing for operations: "
        + ", ".join(missing_operations)
    )


# =============================================================================
# 5. MAINTENANCE ACTION
# =============================================================================

def maintenance_action(row):

    alert = row["maintenance_alert"]
    tier = row["probe_tier"]

    if alert == "NORMAL":

        return (
            "CONTINUE_OPERATION"
        )

    if alert == "WATCH":

        if tier in [
            "PRIMARY",
            "SECONDARY",
        ]:

            return (
                "INCREASE_MONITORING"
            )

        return (
            "REPEAT_PRIMARY_PROBE"
        )

    if alert == "WARNING":

        return (
            "TARGETED_INSPECTION"
        )

    if alert == "CRITICAL":

        if tier == "PRIMARY":

            return (
                "PRIORITIZE_MAINTENANCE"
            )

        return (
            "CONFIRM_AND_PRIORITIZE"
        )

    return "REVIEW"


df[
    "maintenance_action"
] = df.apply(
    maintenance_action,
    axis=1,
)


# =============================================================================
# 6. DECISION CONFIDENCE
# =============================================================================

# Confidence here means confidence in the analytical maintenance decision,
# NOT probability of failure.
#
# Probe quality modifies how much trust we place in the observed alert.

def decision_confidence(row):

    alert = row["maintenance_alert"]
    tier = row["probe_tier"]

    if alert == "NORMAL":

        if tier in [
            "PRIMARY",
            "SECONDARY",
        ]:
            return "HIGH"

        return "LIMITED"

    if alert == "WATCH":

        if tier == "PRIMARY":
            return "HIGH"

        if tier == "SECONDARY":
            return "MODERATE"

        return "LIMITED"

    if alert == "WARNING":

        if tier == "PRIMARY":
            return "HIGH"

        if tier in [
            "SECONDARY",
            "SUPPORTING",
        ]:
            return "MODERATE"

        return "LIMITED"

    if alert == "CRITICAL":

        if tier == "PRIMARY":
            return "VERY_HIGH"

        if tier == "SECONDARY":
            return "HIGH"

        return "MODERATE"

    return "LIMITED"


df[
    "decision_confidence"
] = df.apply(
    decision_confidence,
    axis=1,
)


# =============================================================================
# 7. MAINTENANCE PRIORITY
# =============================================================================

priority_map = {
    "CONTINUE_OPERATION": 0,
    "REPEAT_PRIMARY_PROBE": 1,
    "INCREASE_MONITORING": 1,
    "TARGETED_INSPECTION": 2,
    "CONFIRM_AND_PRIORITIZE": 3,
    "PRIORITIZE_MAINTENANCE": 3,
    "REVIEW": 1,
}


df[
    "maintenance_priority"
] = (
    df[
        "maintenance_action"
    ]
    .map(priority_map)
)


# =============================================================================
# 8. HUMAN-READABLE RECOMMENDATION
# =============================================================================

def recommendation(row):

    action = row[
        "maintenance_action"
    ]

    if action == "CONTINUE_OPERATION":

        if row["probe_tier"] in [
            "PRIMARY",
            "SECONDARY",
        ]:

            return (
                "Continue normal operation and routine condition monitoring."
            )

        return (
            "Continue operation, but do not treat this trajectory alone "
            "as strong evidence of healthy mechanical condition."
        )

    if action == "INCREASE_MONITORING":

        return (
            "Continue production while increasing condition-monitoring "
            "frequency and repeat a preferred diagnostic probe."
        )

    if action == "REPEAT_PRIMARY_PROBE":

        return (
            "Do not escalate from this trajectory alone; repeat a PRIMARY "
            "diagnostic probe to obtain stronger degradation evidence."
        )

    if action == "TARGETED_INSPECTION":

        return (
            "Schedule targeted inspection of the Joint 3 actuator/mechanical "
            "drive system and confirm condition using a PRIMARY diagnostic probe."
        )

    if action == "CONFIRM_AND_PRIORITIZE":

        return (
            "Obtain confirmation using a PRIMARY diagnostic probe and "
            "prioritize maintenance assessment if degradation persists."
        )

    if action == "PRIORITIZE_MAINTENANCE":

        return (
            "Prioritize maintenance assessment of the Joint 3 actuator/"
            "mechanical drive system and plan intervention while analytical "
            "warning lead remains."
        )

    return (
        "Engineering review required."
    )


df[
    "maintenance_recommendation"
] = df.apply(
    recommendation,
    axis=1,
)


# =============================================================================
# 9. POLICY TABLE
# =============================================================================

policy = (
    df[
        [
            "maintenance_alert",
            "probe_tier",
            "maintenance_action",
            "decision_confidence",
            "maintenance_priority",
            "maintenance_recommendation",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "maintenance_priority",
            "maintenance_alert",
            "probe_tier",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 10. SAVE CYCLE-LEVEL DECISIONS
# =============================================================================

decision_columns = [
    "operation_code",
    "cycle_index",
    "benchmark_health_region",
    "diagnostic_group",
    "probe_rank",
    "probe_tier",
    "probe_utility_pct",
    "health_index",
    "if_threshold_ratio",
    "maintenance_alert",
    "alert_severity",
    "maintenance_action",
    "maintenance_priority",
    "decision_confidence",
    "maintenance_recommendation",
]


df[
    decision_columns
].to_csv(
    OUTPUT_CYCLES,
    index=False,
)


policy.to_csv(
    OUTPUT_POLICY,
    index=False,
)


# =============================================================================
# 11. OPERATION-LEVEL HIGHEST MAINTENANCE PRIORITY
# =============================================================================

summary_records = []


for operation_code, group in df.groupby(
    "operation_code"
):

    group = group.sort_values(
        "cycle_index"
    )

    max_priority = int(
        group[
            "maintenance_priority"
        ].max()
    )

    highest = group[
        group[
            "maintenance_priority"
        ] == max_priority
    ]

    first_highest = highest.iloc[0]

    summary_records.append(
        {
            "operation_code":
                operation_code,

            "diagnostic_group":
                first_highest[
                    "diagnostic_group"
                ],

            "probe_rank":
                int(
                    first_highest[
                        "probe_rank"
                    ]
                ),

            "probe_tier":
                first_highest[
                    "probe_tier"
                ],

            "probe_utility_pct":
                first_highest[
                    "probe_utility_pct"
                ],

            "highest_alert":
                first_highest[
                    "maintenance_alert"
                ],

            "highest_maintenance_action":
                first_highest[
                    "maintenance_action"
                ],

            "highest_priority":
                max_priority,

            "decision_confidence":
                first_highest[
                    "decision_confidence"
                ],

            "first_highest_priority_cycle":
                int(
                    first_highest[
                        "cycle_index"
                    ]
                ),

            "maintenance_recommendation":
                first_highest[
                    "maintenance_recommendation"
                ],
        }
    )


operation_summary = pd.DataFrame(
    summary_records
)


operation_summary = operation_summary.sort_values(
    [
        "highest_priority",
        "probe_utility_pct",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(
    drop=True
)


operation_summary.to_csv(
    OUTPUT_OPERATION_SUMMARY,
    index=False,
)


# =============================================================================
# 12. PRINT DECISION POLICY
# =============================================================================

print(
    "\nMAINTENANCE DECISION POLICY"
)

print("-" * 120)


print(
    policy[
        [
            "maintenance_alert",
            "probe_tier",
            "maintenance_action",
            "decision_confidence",
            "maintenance_priority",
        ]
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# 13. OPERATION-LEVEL DECISION SUMMARY
# =============================================================================

print(
    "\nOPERATION-LEVEL HIGHEST MAINTENANCE RESPONSE"
)

print("-" * 120)


print(
    operation_summary[
        [
            "operation_code",
            "probe_rank",
            "probe_tier",
            "highest_alert",
            "highest_maintenance_action",
            "highest_priority",
            "decision_confidence",
            "first_highest_priority_cycle",
        ]
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# 14. PRIORITY DISTRIBUTION
# =============================================================================

priority_distribution = (
    operation_summary[
        "highest_priority"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nHIGHEST PRIORITY DISTRIBUTION"
)

print("-" * 120)


for priority, count in priority_distribution.items():

    label = {
        0: "ROUTINE",
        1: "ENHANCED MONITORING",
        2: "TARGETED INSPECTION",
        3: "MAINTENANCE PRIORITY",
    }.get(
        priority,
        "UNKNOWN",
    )

    print(
        f"Priority {priority} "
        f"({label}) : {count} operations"
    )


# =============================================================================
# 15. PRIMARY-PROBE MAINTENANCE ESCALATIONS
# =============================================================================

primary_escalations = operation_summary[
    (
        operation_summary[
            "probe_tier"
        ] == "PRIMARY"
    )
    &
    (
        operation_summary[
            "highest_priority"
        ] >= 2
    )
].copy()


print(
    "\nPRIMARY-PROBE MAINTENANCE ESCALATIONS"
)

print("-" * 120)


if len(primary_escalations) > 0:

    print(
        primary_escalations[
            [
                "operation_code",
                "probe_rank",
                "highest_alert",
                "highest_maintenance_action",
                "decision_confidence",
                "first_highest_priority_cycle",
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "No PRIMARY diagnostic probes reached "
        "maintenance escalation."
    )


# =============================================================================
# 16. NORMAL DOES NOT ALWAYS MEAN STRONG HEALTH EVIDENCE
# =============================================================================

normal_by_tier = (
    df[
        df[
            "maintenance_alert"
        ] == "NORMAL"
    ]
    .groupby(
        "probe_tier"
    )
    .size()
    .reset_index(
        name="normal_cycles"
    )
)


print(
    "\nNORMAL-STATE INTERPRETATION BY PROBE TIER"
)

print("-" * 120)


print(
    normal_by_tier.to_string(
        index=False
    )
)


# =============================================================================
# 17. TOP DIAGNOSTIC PROBE DECISION PATH
# =============================================================================

top_operation = (
    probes
    .sort_values(
        "probe_rank"
    )
    .iloc[0][
        "operation_code"
    ]
)


top_path = (
    df[
        df[
            "operation_code"
        ] == top_operation
    ]
    .sort_values(
        "cycle_index"
    )
    .copy()
)


top_path[
    "previous_action"
] = (
    top_path[
        "maintenance_action"
    ]
    .shift(1)
)


top_transitions = top_path[
    (
        top_path[
            "maintenance_action"
        ]
        != top_path[
            "previous_action"
        ]
    )
    &
    top_path[
        "previous_action"
    ].notna()
]


print(
    f"\nTOP-PROBE DECISION PATH — OPERATION "
    f"{top_operation}"
)

print("-" * 120)


if len(top_transitions) > 0:

    print(
        top_transitions[
            [
                "cycle_index",
                "maintenance_alert",
                "maintenance_action",
                "decision_confidence",
                "health_index",
                "if_threshold_ratio",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No maintenance-action transitions identified."
    )


# =============================================================================
# 18. OUTPUTS
# =============================================================================

print(
    f"\nCycle-level maintenance decisions saved to:\n"
    f"{OUTPUT_CYCLES}"
)

print(
    f"\nDecision policy saved to:\n"
    f"{OUTPUT_POLICY}"
)

print(
    f"\nOperation-level decision summary saved to:\n"
    f"{OUTPUT_OPERATION_SUMMARY}"
)


print("\n" + "=" * 120)
print("PHASE 07.3 MAINTENANCE DECISION POLICY COMPLETE")
print("=" * 120)