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

OPPORTUNITY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_opportunity_windows.csv"
)

DECISION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_decision_summary.csv"
)

PROBE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_diagnostic_probe_ranking.csv"
)

OUTPUT_GOLD = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_predictive_maintenance_gold.csv"
)

OUTPUT_KPI = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_predictive_maintenance_business_kpis.csv"
)

OUTPUT_ACTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_predictive_maintenance_action_matrix.csv"
)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

opportunity = pd.read_csv(
    OPPORTUNITY_FILE,
    dtype={"operation_code": str},
)

decisions = pd.read_csv(
    DECISION_FILE,
    dtype={"operation_code": str},
)

probes = pd.read_csv(
    PROBE_FILE,
    dtype={"operation_code": str},
)


print("=" * 120)
print(
    "ME-AD PHASE 08.2 — PREDICTIVE MAINTENANCE "
    "EVIDENCE & BUSINESS KPI SYNTHESIS"
)
print("=" * 120)

print(
    f"\nOpportunity operations : "
    f"{opportunity['operation_code'].nunique()}"
)

print(
    f"Decision operations    : "
    f"{decisions['operation_code'].nunique()}"
)

print(
    f"Probe operations       : "
    f"{probes['operation_code'].nunique()}"
)


# =============================================================================
# 3. VALIDATE INPUT SCHEMAS
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
    opportunity,
    [
        "operation_code",
        "diagnostic_group",
        "fault_start_cycle",
        "first_persistent_signal_cycle",
        "early_awareness_lead_cycles",
        "actionable_maintenance_cycle",
        "maintenance_opportunity_cycles",
        "maintenance_opportunity_pct",
        "confirmation_delay_cycles",
        "monitoring_lead_cycles",
        "predictive_maintenance_class",
    ],
    "Maintenance opportunity table",
)

require_columns(
    decisions,
    [
        "operation_code",
        "highest_alert",
        "highest_maintenance_action",
        "highest_priority",
        "decision_confidence",
        "first_highest_priority_cycle",
        "maintenance_recommendation",
    ],
    "Maintenance decision summary",
)

require_columns(
    probes,
    [
        "operation_code",
        "probe_rank",
        "probe_tier",
        "probe_utility_pct",
        "health_detection_rate_pct",
        "if_warning_detected",
        "health_warning_detected",
        "faulty_anomaly_rate_pct",
        "intermediate_anomaly_rate_pct",
        "recommended_maintenance_use",
    ],
    "Diagnostic probe ranking",
)


# =============================================================================
# 4. PREPARE INPUT TABLES
# =============================================================================

opportunity_keep = opportunity[
    [
        "operation_code",
        "diagnostic_group",
        "fault_start_cycle",
        "first_persistent_signal_cycle",
        "early_awareness_lead_cycles",
        "actionable_maintenance_cycle",
        "maintenance_opportunity_cycles",
        "maintenance_opportunity_pct",
        "confirmation_delay_cycles",
        "monitoring_lead_cycles",
        "predictive_maintenance_class",
    ]
].copy()


decision_keep = decisions[
    [
        "operation_code",
        "highest_alert",
        "highest_maintenance_action",
        "highest_priority",
        "decision_confidence",
        "first_highest_priority_cycle",
        "maintenance_recommendation",
    ]
].copy()


probe_keep = probes[
    [
        "operation_code",
        "probe_rank",
        "probe_tier",
        "probe_utility_pct",
        "health_detection_rate_pct",
        "if_warning_detected",
        "health_warning_detected",
        "faulty_anomaly_rate_pct",
        "intermediate_anomaly_rate_pct",
        "recommended_maintenance_use",
    ]
].copy()


# =============================================================================
# 5. MERGE INTO GOLD EVIDENCE TABLE
# =============================================================================

gold = (
    opportunity_keep
    .merge(
        decision_keep,
        on="operation_code",
        how="left",
        validate="one_to_one",
    )
    .merge(
        probe_keep,
        on="operation_code",
        how="left",
        validate="one_to_one",
    )
)


if gold["operation_code"].nunique() != 16:
    raise ValueError(
        "Expected 16 operations after merge, found "
        f"{gold['operation_code'].nunique()}."
    )


required_after_merge = [
    "probe_rank",
    "probe_tier",
    "highest_priority",
    "decision_confidence",
]


if gold[
    required_after_merge
].isna().any().any():
    raise ValueError(
        "Missing maintenance-decision or probe-ranking "
        "information after Gold-table merge."
    )


print(
    f"Merged Gold operations : "
    f"{gold['operation_code'].nunique()}"
)


# =============================================================================
# 6. EVIDENCE CONVERGENCE
# =============================================================================

def evidence_convergence(row):

    health = bool(
        row[
            "health_warning_detected"
        ]
    )

    anomaly = bool(
        row[
            "if_warning_detected"
        ]
    )

    if health and anomaly:
        return "CONVERGENT"

    if health and not anomaly:
        return "HEALTH_ONLY"

    if anomaly and not health:
        return "ANOMALY_ONLY"

    return "NO_PERSISTENT_WARNING"


gold[
    "evidence_convergence"
] = gold.apply(
    evidence_convergence,
    axis=1,
)


# =============================================================================
# 7. MAINTENANCE READINESS
# =============================================================================

# This is an operational categorization, not a probability of failure.

def maintenance_readiness(row):

    pm_class = row[
        "predictive_maintenance_class"
    ]

    if pm_class == "STRONG_ACTIONABLE":
        return "READY_FOR_MAINTENANCE_PLANNING"

    if pm_class == "INSPECTION_ACTIONABLE":
        return "READY_FOR_TARGETED_INSPECTION"

    if pm_class == "MONITORING_ONLY":
        return "ENHANCED_MONITORING"

    if pm_class == "INSUFFICIENT_OBSERVABILITY":
        return "REQUIRE_DIAGNOSTIC_PROBE"

    return "ROUTINE_MONITORING"


gold[
    "maintenance_readiness"
] = gold.apply(
    maintenance_readiness,
    axis=1,
)


# =============================================================================
# 8. LEAD-TIME KPI
# =============================================================================

# Select the lead measure appropriate to the decision class.
#
# Strong/inspection actionable:
# confirmed maintenance opportunity.
#
# Monitoring only:
# monitoring lead.
#
# Other:
# no actionable lead KPI.

gold[
    "decision_lead_cycles"
] = np.select(
    [
        gold[
            "predictive_maintenance_class"
        ].isin(
            [
                "STRONG_ACTIONABLE",
                "INSPECTION_ACTIONABLE",
            ]
        ),

        gold[
            "predictive_maintenance_class"
        ] == "MONITORING_ONLY",
    ],
    [
        gold[
            "maintenance_opportunity_cycles"
        ],

        gold[
            "monitoring_lead_cycles"
        ],
    ],
    default=np.nan,
)


# =============================================================================
# 9. BUSINESS PRIORITY LABEL
# =============================================================================

priority_labels = {
    0: "ROUTINE",
    1: "ENHANCED_MONITORING",
    2: "TARGETED_INSPECTION",
    3: "MAINTENANCE_PRIORITY",
}


gold[
    "business_priority"
] = (
    gold[
        "highest_priority"
    ]
    .map(priority_labels)
)


# =============================================================================
# 10. OBSERVABILITY INTERPRETATION
# =============================================================================

def observability_interpretation(row):

    tier = row[
        "probe_tier"
    ]

    if tier == "PRIMARY":
        return (
            "High-value diagnostic trajectory for recurring "
            "condition assessment."
        )

    if tier == "SECONDARY":
        return (
            "Useful supplementary trajectory for condition monitoring."
        )

    if tier == "SUPPORTING":
        return (
            "Provides supporting evidence but should not be relied "
            "upon as the sole diagnostic trajectory."
        )

    return (
        "Poor standalone early-warning observability; use a stronger "
        "diagnostic probe before drawing maintenance conclusions."
    )


gold[
    "observability_interpretation"
] = gold.apply(
    observability_interpretation,
    axis=1,
)


# =============================================================================
# 11. BUSINESS DECISION STATEMENT
# =============================================================================

def business_decision(row):

    readiness = row[
        "maintenance_readiness"
    ]

    operation = row[
        "operation_code"
    ]

    if readiness == "READY_FOR_MAINTENANCE_PLANNING":

        lead = row[
            "maintenance_opportunity_cycles"
        ]

        return (
            f"Operation {operation} provides convergent degradation "
            f"evidence with approximately {lead:.0f} cycles of "
            f"confirmed analytical maintenance opportunity before "
            f"the evident-fault region."
        )

    if readiness == "READY_FOR_TARGETED_INSPECTION":

        lead = row[
            "maintenance_opportunity_cycles"
        ]

        return (
            f"Operation {operation} supports targeted inspection with "
            f"approximately {lead:.0f} cycles of analytical lead, "
            f"but independent anomaly confirmation is limited."
        )

    if readiness == "ENHANCED_MONITORING":

        lead = row[
            "monitoring_lead_cycles"
        ]

        return (
            f"Operation {operation} shows persistent abnormality with "
            f"approximately {lead:.0f} cycles of monitoring lead; "
            f"repeat a PRIMARY diagnostic probe before maintenance escalation."
        )

    if readiness == "REQUIRE_DIAGNOSTIC_PROBE":

        return (
            f"Operation {operation} has insufficient standalone "
            f"early-warning observability; absence of an alert should "
            f"not be interpreted as strong evidence of healthy condition."
        )

    return (
        f"Operation {operation} has not produced persistent actionable "
        f"evidence under the current maintenance policy."
    )


gold[
    "business_decision_statement"
] = gold.apply(
    business_decision,
    axis=1,
)


# =============================================================================
# 12. SORT GOLD TABLE
# =============================================================================

gold = gold.sort_values(
    [
        "highest_priority",
        "probe_rank",
    ],
    ascending=[
        False,
        True,
    ],
).reset_index(
    drop=True
)


# =============================================================================
# 13. SAVE GOLD TABLE
# =============================================================================

gold.to_csv(
    OUTPUT_GOLD,
    index=False,
)


# =============================================================================
# 14. BUILD ACTION MATRIX
# =============================================================================

action_matrix = (
    gold[
        [
            "predictive_maintenance_class",
            "maintenance_readiness",
            "business_priority",
            "highest_maintenance_action",
            "decision_confidence",
            "maintenance_recommendation",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "business_priority",
            "predictive_maintenance_class",
        ]
    )
    .reset_index(
        drop=True
    )
)


action_matrix.to_csv(
    OUTPUT_ACTIONS,
    index=False,
)


# =============================================================================
# 15. BUSINESS KPI CALCULATIONS
# =============================================================================

total_operations = (
    gold[
        "operation_code"
    ]
    .nunique()
)


primary_operations = int(
    (
        gold[
            "probe_tier"
        ] == "PRIMARY"
    )
    .sum()
)


strong_operations = int(
    (
        gold[
            "predictive_maintenance_class"
        ] == "STRONG_ACTIONABLE"
    )
    .sum()
)


inspection_operations = int(
    (
        gold[
            "predictive_maintenance_class"
        ] == "INSPECTION_ACTIONABLE"
    )
    .sum()
)


monitoring_operations = int(
    (
        gold[
            "predictive_maintenance_class"
        ] == "MONITORING_ONLY"
    )
    .sum()
)


low_observability_operations = int(
    (
        gold[
            "probe_tier"
        ] == "LOW_OBSERVABILITY"
    )
    .sum()
)


strong = gold[
    gold[
        "predictive_maintenance_class"
    ] == "STRONG_ACTIONABLE"
].copy()


primary = gold[
    gold[
        "probe_tier"
    ] == "PRIMARY"
].copy()


# =============================================================================
# 16. KPI TABLE
# =============================================================================

kpi_records = []


def add_kpi(
    name,
    value,
    unit,
    interpretation,
):
    kpi_records.append(
        {
            "kpi_name": name,
            "kpi_value": value,
            "unit": unit,
            "interpretation": interpretation,
        }
    )


add_kpi(
    "Total operating conditions assessed",
    total_operations,
    "operations",
    (
        "Number of Mitsubishi operating conditions included "
        "in the final maintenance synthesis."
    ),
)


add_kpi(
    "Primary diagnostic probes",
    primary_operations,
    "operations",
    (
        "Operating trajectories classified as preferred recurring "
        "diagnostic health probes."
    ),
)


add_kpi(
    "Strong actionable conditions",
    strong_operations,
    "operations",
    (
        "Conditions where persistent health and anomaly evidence "
        "supports maintenance planning."
    ),
)


add_kpi(
    "Inspection-actionable conditions",
    inspection_operations,
    "operations",
    (
        "Conditions supporting targeted inspection without full "
        "independent warning convergence."
    ),
)


add_kpi(
    "Monitoring-only conditions",
    monitoring_operations,
    "operations",
    (
        "Conditions showing persistent abnormality but insufficient "
        "evidence for direct maintenance escalation."
    ),
)


add_kpi(
    "Low-observability conditions",
    low_observability_operations,
    "operations",
    (
        "Conditions unsuitable for standalone early-warning "
        "maintenance decisions."
    ),
)


if len(strong) > 0:

    add_kpi(
        "Median early-awareness lead",
        strong[
            "early_awareness_lead_cycles"
        ].median(),
        "cycles",
        (
            "Median analytical lead from the first persistent signal "
            "to the evident-fault boundary for strong actionable conditions."
        ),
    )

    add_kpi(
        "Median confirmation delay",
        strong[
            "confirmation_delay_cycles"
        ].median(),
        "cycles",
        (
            "Median interval between first persistent analytical "
            "awareness and actionable maintenance confirmation."
        ),
    )

    add_kpi(
        "Median confirmed maintenance opportunity",
        strong[
            "maintenance_opportunity_cycles"
        ].median(),
        "cycles",
        (
            "Median confirmed intervention-planning window before "
            "the evident-fault region."
        ),
    )

    add_kpi(
        "Minimum confirmed maintenance opportunity",
        strong[
            "maintenance_opportunity_cycles"
        ].min(),
        "cycles",
        (
            "Smallest confirmed maintenance-planning window among "
            "strong actionable conditions."
        ),
    )

    add_kpi(
        "Maximum confirmed maintenance opportunity",
        strong[
            "maintenance_opportunity_cycles"
        ].max(),
        "cycles",
        (
            "Largest confirmed maintenance-planning window among "
            "strong actionable conditions."
        ),
    )

    add_kpi(
        "Median pre-fault sequence remaining at escalation",
        strong[
            "maintenance_opportunity_pct"
        ].median(),
        "percent",
        (
            "Median proportion of the observed pre-evident-fault "
            "sequence remaining when actionable maintenance evidence "
            "was established."
        ),
    )


if len(primary) > 0:

    add_kpi(
        "Median primary-probe utility",
        primary[
            "probe_utility_pct"
        ].median(),
        "percent",
        (
            "Median diagnostic utility score across PRIMARY "
            "health-probe trajectories."
        ),
    )


kpis = pd.DataFrame(
    kpi_records
)


kpis.to_csv(
    OUTPUT_KPI,
    index=False,
)


# =============================================================================
# 17. PRINT BUSINESS KPI SCORECARD
# =============================================================================

print(
    "\nPREDICTIVE-MAINTENANCE BUSINESS KPI SCORECARD"
)

print("-" * 120)


print(
    kpis[
        [
            "kpi_name",
            "kpi_value",
            "unit",
        ]
    ]
    .round(
        {
            "kpi_value": 2,
        }
    )
    .to_string(
        index=False
    )
)


# =============================================================================
# 18. PRINT OPERATION DECISION MATRIX
# =============================================================================

print(
    "\nOPERATION-LEVEL PREDICTIVE-MAINTENANCE DECISION MATRIX"
)

print("-" * 120)


print(
    gold[
        [
            "operation_code",
            "probe_rank",
            "probe_tier",
            "probe_utility_pct",
            "evidence_convergence",
            "predictive_maintenance_class",
            "decision_lead_cycles",
            "business_priority",
            "decision_confidence",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 19. PRIMARY PROBE BUSINESS VIEW
# =============================================================================

print(
    "\nPRIMARY DIAGNOSTIC PROBE BUSINESS VIEW"
)

print("-" * 120)


print(
    primary[
        [
            "operation_code",
            "probe_rank",
            "probe_utility_pct",
            "evidence_convergence",
            "early_awareness_lead_cycles",
            "confirmation_delay_cycles",
            "maintenance_opportunity_cycles",
            "maintenance_opportunity_pct",
            "decision_confidence",
        ]
    ]
    .sort_values(
        "probe_rank"
    )
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 20. ACTIONABLE MAINTENANCE CASES
# =============================================================================

actionable = gold[
    gold[
        "highest_priority"
    ] >= 2
].copy()


print(
    "\nACTIONABLE MAINTENANCE CASES"
)

print("-" * 120)


if len(actionable) > 0:

    print(
        actionable[
            [
                "operation_code",
                "probe_tier",
                "highest_maintenance_action",
                "maintenance_opportunity_cycles",
                "decision_confidence",
                "business_decision_statement",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No actionable maintenance cases identified."
    )


# =============================================================================
# 21. LOW-OBSERVABILITY CONDITIONS
# =============================================================================

low_observability = gold[
    gold[
        "probe_tier"
    ] == "LOW_OBSERVABILITY"
].copy()


print(
    "\nLOW-OBSERVABILITY CONDITIONS"
)

print("-" * 120)


if len(low_observability) > 0:

    print(
        low_observability[
            [
                "operation_code",
                "probe_rank",
                "faulty_anomaly_rate_pct",
                "business_priority",
                "decision_confidence",
                "business_decision_statement",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No low-observability conditions identified."
    )


# =============================================================================
# 22. TOP DIAGNOSTIC PROBE
# =============================================================================

top_probe = (
    gold.sort_values(
        "probe_rank"
    )
    .iloc[0]
)


print(
    "\nTOP DIAGNOSTIC PROBE — BUSINESS SUMMARY"
)

print("-" * 120)

print(
    f"Operation                         : "
    f"{top_probe['operation_code']}"
)

print(
    f"Probe rank                        : "
    f"{int(top_probe['probe_rank'])}"
)

print(
    f"Probe tier                        : "
    f"{top_probe['probe_tier']}"
)

print(
    f"Probe utility                     : "
    f"{top_probe['probe_utility_pct']:.2f}%"
)

print(
    f"Evidence convergence              : "
    f"{top_probe['evidence_convergence']}"
)

print(
    f"Predictive-maintenance class      : "
    f"{top_probe['predictive_maintenance_class']}"
)

print(
    f"Decision confidence               : "
    f"{top_probe['decision_confidence']}"
)

print(
    f"Early-awareness lead              : "
    f"{top_probe['early_awareness_lead_cycles']:.0f} cycles"
)

print(
    f"Confirmed maintenance opportunity : "
    f"{top_probe['maintenance_opportunity_cycles']:.0f} cycles"
)

print(
    f"Maintenance action                : "
    f"{top_probe['highest_maintenance_action']}"
)

print(
    f"\nBusiness interpretation:\n"
    f"{top_probe['business_decision_statement']}"
)


# =============================================================================
# 23. IMPORTANT INTERPRETATION GUARDRAIL
# =============================================================================

print(
    "\nINTERPRETATION GUARDRAIL"
)

print("-" * 120)

print(
    "Lead and maintenance-opportunity values are measured in observed "
    "operating cycles relative to the dataset-defined evident-fault boundary."
)

print(
    "They are not Remaining Useful Life (RUL), probability of failure, "
    "component life, or time-to-physical-failure estimates."
)

print(
    "Probe utility is a transparent analytical synthesis score and "
    "should not be interpreted as a probability."
)


# =============================================================================
# 24. OUTPUTS
# =============================================================================

print(
    f"\nGold predictive-maintenance table saved to:\n"
    f"{OUTPUT_GOLD}"
)

print(
    f"\nBusiness KPI scorecard saved to:\n"
    f"{OUTPUT_KPI}"
)

print(
    f"\nMaintenance action matrix saved to:\n"
    f"{OUTPUT_ACTIONS}"
)


print("\n" + "=" * 120)
print(
    "PHASE 08.2 PREDICTIVE MAINTENANCE KPI SYNTHESIS COMPLETE"
)
print("=" * 120)