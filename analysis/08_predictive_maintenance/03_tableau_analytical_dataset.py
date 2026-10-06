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

GOLD_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_predictive_maintenance_gold.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dashboard"
    / "me_ad_tableau_analytical_dataset.csv"
)


# =============================================================================
# 2. LOAD FINAL GOLD TABLE
# =============================================================================

gold = pd.read_csv(
    GOLD_FILE,
    dtype={"operation_code": str},
)


print("=" * 120)
print(
    "ME-AD — FINAL TABLEAU ANALYTICAL DATASET"
)
print("=" * 120)

print(
    f"\nInput rows       : {len(gold):,}"
)

print(
    f"Operations       : {gold['operation_code'].nunique()}"
)

print(
    f"Input columns    : {len(gold.columns)}"
)


# =============================================================================
# 3. VALIDATE REQUIRED FIELDS
# =============================================================================

required = [
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
    "highest_alert",
    "highest_maintenance_action",
    "highest_priority",
    "decision_confidence",
    "probe_rank",
    "probe_tier",
    "probe_utility_pct",
    "health_detection_rate_pct",
    "if_warning_detected",
    "health_warning_detected",
    "faulty_anomaly_rate_pct",
    "intermediate_anomaly_rate_pct",
    "recommended_maintenance_use",
    "evidence_convergence",
    "maintenance_readiness",
    "decision_lead_cycles",
    "business_priority",
    "observability_interpretation",
    "business_decision_statement",
]


missing = [
    column
    for column in required
    if column not in gold.columns
]


if missing:
    raise ValueError(
        "Gold table is missing expected columns:\n"
        + "\n".join(missing)
        + "\n\nAvailable columns:\n"
        + "\n".join(gold.columns)
    )


# =============================================================================
# 4. BUILD TABLEAU DATASET
# =============================================================================

tableau = gold[
    required
].copy()


# =============================================================================
# 5. BUSINESS-FRIENDLY OPERATION LABEL
# =============================================================================

tableau[
    "operation_label"
] = (
    "Operation "
    +
    tableau[
        "operation_code"
    ].astype(str)
)


# =============================================================================
# 6. OPERATION FAMILY
# =============================================================================

# Dataset structure:
#
# Family 1:
# 1050–1053 and 1100–1103
#
# Family 2:
# 2000–2500
#
# Family 3:
# 3000–3100

family_1 = {
    "1050",
    "1051",
    "1052",
    "1053",
    "1100",
    "1101",
    "1102",
    "1103",
}

family_2 = {
    "2000",
    "2100",
    "2200",
    "2300",
    "2400",
    "2500",
}

family_3 = {
    "3000",
    "3100",
}


def operation_family(code):

    code = str(code)

    if code in family_1:
        return "Family 1 — Repeated Pick-and-Place"

    if code in family_2:
        return "Family 2 — Factorized Pick-and-Place"

    if code in family_3:
        return "Family 3 — Blended Continuous Trajectory"

    return "Unknown"


tableau[
    "operation_family"
] = tableau[
    "operation_code"
].apply(
    operation_family
)


# =============================================================================
# 7. COMMANDED SPEED GROUP
# =============================================================================

# Family 1 operation-code structure:
#
# 105x = 50% commanded speed
# 110x = 100% commanded speed
#
# Families 2 and 3 do not use this same dashboard grouping.

def speed_group(code):

    code = str(code)

    if code.startswith("105"):
        return "50%"

    if code.startswith("110"):
        return "100%"

    return "Not Applicable"


tableau[
    "commanded_speed_group"
] = tableau[
    "operation_code"
].apply(
    speed_group
)


# =============================================================================
# 8. ACTIONABLE FLAG
# =============================================================================

tableau[
    "actionable_flag"
] = np.where(
    tableau[
        "highest_priority"
    ] >= 2,
    "Actionable",
    "Not Actionable",
)


# =============================================================================
# 9. PERSISTENT WARNING FLAG
# =============================================================================

tableau[
    "persistent_warning_flag"
] = np.where(
    (
        tableau[
            "health_warning_detected"
        ].astype(bool)
    )
    |
    (
        tableau[
            "if_warning_detected"
        ].astype(bool)
    ),
    "Persistent Warning",
    "No Persistent Warning",
)


# =============================================================================
# 10. CONVERGENT EVIDENCE FLAG
# =============================================================================

tableau[
    "convergent_evidence_flag"
] = np.where(
    tableau[
        "evidence_convergence"
    ] == "CONVERGENT",
    "Convergent Evidence",
    "No Full Convergence",
)


# =============================================================================
# 11. PRIMARY PROBE FLAG
# =============================================================================

tableau[
    "primary_probe_flag"
] = np.where(
    tableau[
        "probe_tier"
    ] == "PRIMARY",
    "Primary Probe",
    "Other Probe",
)


# =============================================================================
# 12. OBSERVABILITY FLAG
# =============================================================================

tableau[
    "observability_flag"
] = np.where(
    tableau[
        "probe_tier"
    ] == "LOW_OBSERVABILITY",
    "Low Observability",
    "Usable Observability",
)


# =============================================================================
# 13. MAINTENANCE OPPORTUNITY BAND
# =============================================================================

def opportunity_band(value):

    if pd.isna(value):
        return "No Confirmed Opportunity"

    if value >= 1000:
        return "1000+ Cycles"

    if value >= 500:
        return "500–999 Cycles"

    if value > 0:
        return "Below 500 Cycles"

    return "No Confirmed Opportunity"


tableau[
    "maintenance_opportunity_band"
] = tableau[
    "maintenance_opportunity_cycles"
].apply(
    opportunity_band
)


# =============================================================================
# 14. EARLY-AWARENESS BAND
# =============================================================================

def awareness_band(value):

    if pd.isna(value):
        return "No Persistent Early Warning"

    if value >= 1500:
        return "1500+ Cycles"

    if value >= 1000:
        return "1000–1499 Cycles"

    if value >= 500:
        return "500–999 Cycles"

    if value > 0:
        return "Below 500 Cycles"

    return "No Persistent Early Warning"


tableau[
    "early_awareness_band"
] = tableau[
    "early_awareness_lead_cycles"
].apply(
    awareness_band
)


# =============================================================================
# 15. MAINTENANCE PRIORITY ORDER
# =============================================================================

# Numeric sorting field for Tableau.

tableau[
    "maintenance_priority_order"
] = tableau[
    "highest_priority"
].astype(int)


# =============================================================================
# 16. PROBE-TIER ORDER
# =============================================================================

probe_tier_order = {
    "PRIMARY": 1,
    "SECONDARY": 2,
    "SUPPORTING": 3,
    "LOW_OBSERVABILITY": 4,
}


tableau[
    "probe_tier_order"
] = (
    tableau[
        "probe_tier"
    ]
    .map(
        probe_tier_order
    )
)


# =============================================================================
# 17. PREDICTIVE-MAINTENANCE CLASS ORDER
# =============================================================================

pm_class_order = {
    "STRONG_ACTIONABLE": 1,
    "INSPECTION_ACTIONABLE": 2,
    "MONITORING_ONLY": 3,
    "NO_ACTIONABLE_WARNING": 4,
    "INSUFFICIENT_OBSERVABILITY": 5,
}


tableau[
    "predictive_maintenance_class_order"
] = (
    tableau[
        "predictive_maintenance_class"
    ]
    .map(
        pm_class_order
    )
)


# =============================================================================
# 18. TABLEAU KPI HELPER FIELDS
# =============================================================================

# These integer helper fields make KPI calculations straightforward
# and prevent Tableau from needing complex string logic.

tableau[
    "kpi_operation_count"
] = 1


tableau[
    "kpi_primary_probe"
] = (
    tableau[
        "probe_tier"
    ] == "PRIMARY"
).astype(int)


tableau[
    "kpi_strong_actionable"
] = (
    tableau[
        "predictive_maintenance_class"
    ] == "STRONG_ACTIONABLE"
).astype(int)


tableau[
    "kpi_inspection_actionable"
] = (
    tableau[
        "predictive_maintenance_class"
    ] == "INSPECTION_ACTIONABLE"
).astype(int)


tableau[
    "kpi_monitoring_only"
] = (
    tableau[
        "predictive_maintenance_class"
    ] == "MONITORING_ONLY"
).astype(int)


tableau[
    "kpi_low_observability"
] = (
    tableau[
        "probe_tier"
    ] == "LOW_OBSERVABILITY"
).astype(int)


tableau[
    "kpi_convergent"
] = (
    tableau[
        "evidence_convergence"
    ] == "CONVERGENT"
).astype(int)


tableau[
    "kpi_actionable"
] = (
    tableau[
        "highest_priority"
    ] >= 2
).astype(int)


# =============================================================================
# 19. DISPLAY-SAFE LEAD FIELDS
# =============================================================================

# Keep the original analytical NaN fields untouched.
#
# These additional fields are only for visuals where Tableau benefits
# from explicit zero values.

tableau[
    "display_maintenance_opportunity_cycles"
] = (
    tableau[
        "maintenance_opportunity_cycles"
    ]
    .fillna(0)
)


tableau[
    "display_early_awareness_lead_cycles"
] = (
    tableau[
        "early_awareness_lead_cycles"
    ]
    .fillna(0)
)


tableau[
    "display_monitoring_lead_cycles"
] = (
    tableau[
        "monitoring_lead_cycles"
    ]
    .fillna(0)
)


# =============================================================================
# 20. FINAL COLUMN ORDER
# =============================================================================

final_columns = [

    # -------------------------------------------------------------------------
    # Operation identity
    # -------------------------------------------------------------------------

    "operation_code",
    "operation_label",
    "operation_family",
    "commanded_speed_group",
    "diagnostic_group",

    # -------------------------------------------------------------------------
    # Diagnostic probe quality
    # -------------------------------------------------------------------------

    "probe_rank",
    "probe_tier",
    "probe_tier_order",
    "probe_utility_pct",
    "primary_probe_flag",
    "observability_flag",
    "health_detection_rate_pct",
    "faulty_anomaly_rate_pct",
    "intermediate_anomaly_rate_pct",

    # -------------------------------------------------------------------------
    # Independent warning evidence
    # -------------------------------------------------------------------------

    "health_warning_detected",
    "if_warning_detected",
    "persistent_warning_flag",
    "evidence_convergence",
    "convergent_evidence_flag",

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    "fault_start_cycle",
    "first_persistent_signal_cycle",
    "early_awareness_lead_cycles",
    "early_awareness_band",
    "confirmation_delay_cycles",
    "actionable_maintenance_cycle",
    "maintenance_opportunity_cycles",
    "maintenance_opportunity_pct",
    "maintenance_opportunity_band",
    "monitoring_lead_cycles",
    "decision_lead_cycles",

    # -------------------------------------------------------------------------
    # Maintenance decision
    # -------------------------------------------------------------------------

    "highest_alert",
    "highest_priority",
    "maintenance_priority_order",
    "business_priority",
    "highest_maintenance_action",
    "decision_confidence",
    "actionable_flag",
    "maintenance_readiness",

    # -------------------------------------------------------------------------
    # Final predictive-maintenance classification
    # -------------------------------------------------------------------------

    "predictive_maintenance_class",
    "predictive_maintenance_class_order",

    # -------------------------------------------------------------------------
    # Engineering interpretation
    # -------------------------------------------------------------------------

    "recommended_maintenance_use",
    "observability_interpretation",
    "business_decision_statement",

    # -------------------------------------------------------------------------
    # KPI helper fields
    # -------------------------------------------------------------------------

    "kpi_operation_count",
    "kpi_primary_probe",
    "kpi_strong_actionable",
    "kpi_inspection_actionable",
    "kpi_monitoring_only",
    "kpi_low_observability",
    "kpi_convergent",
    "kpi_actionable",

    # -------------------------------------------------------------------------
    # Display helper fields
    # -------------------------------------------------------------------------

    "display_maintenance_opportunity_cycles",
    "display_early_awareness_lead_cycles",
    "display_monitoring_lead_cycles",
]


tableau = tableau[
    final_columns
].copy()


# =============================================================================
# 21. SORT FINAL TABLE
# =============================================================================

tableau = tableau.sort_values(
    [
        "probe_rank",
        "operation_code",
    ]
).reset_index(
    drop=True
)


# =============================================================================
# 22. FINAL QUALITY CHECKS
# =============================================================================

if len(tableau) != 16:
    raise ValueError(
        f"Expected 16 Tableau rows, found {len(tableau)}."
    )


if tableau[
    "operation_code"
].duplicated().any():
    raise ValueError(
        "Duplicate operation codes found in Tableau dataset."
    )


if tableau[
    "probe_rank"
].isna().any():
    raise ValueError(
        "Missing probe ranks found in Tableau dataset."
    )


# =============================================================================
# 23. SAVE
# =============================================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


tableau.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =============================================================================
# 24. PRINT FINAL DATASET
# =============================================================================

print(
    "\nFINAL TABLEAU ANALYTICAL DATASET"
)

print("-" * 120)


print(
    tableau[
        [
            "operation_code",
            "operation_family",
            "commanded_speed_group",
            "probe_rank",
            "probe_tier",
            "probe_utility_pct",
            "evidence_convergence",
            "predictive_maintenance_class",
            "early_awareness_lead_cycles",
            "maintenance_opportunity_cycles",
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
# 25. DATASET PROFILE
# =============================================================================

print(
    "\nTABLEAU DATASET PROFILE"
)

print("-" * 120)

print(
    f"Rows                    : "
    f"{len(tableau):,}"
)

print(
    f"Columns                 : "
    f"{len(tableau.columns):,}"
)

print(
    f"Unique operations       : "
    f"{tableau['operation_code'].nunique()}"
)

print(
    f"Primary probes          : "
    f"{tableau['kpi_primary_probe'].sum()}"
)

print(
    f"Strong actionable       : "
    f"{tableau['kpi_strong_actionable'].sum()}"
)

print(
    f"Inspection actionable   : "
    f"{tableau['kpi_inspection_actionable'].sum()}"
)

print(
    f"Monitoring only         : "
    f"{tableau['kpi_monitoring_only'].sum()}"
)

print(
    f"Low observability       : "
    f"{tableau['kpi_low_observability'].sum()}"
)

print(
    f"Convergent operations   : "
    f"{tableau['kpi_convergent'].sum()}"
)

print(
    f"Actionable operations   : "
    f"{tableau['kpi_actionable'].sum()}"
)


# =============================================================================
# 26. VALIDATE HEADLINE KPI VALUES
# =============================================================================

strong = tableau[
    tableau[
        "predictive_maintenance_class"
    ] == "STRONG_ACTIONABLE"
]


print(
    "\nHEADLINE KPI VALIDATION"
)

print("-" * 120)


if len(strong) > 0:

    print(
        f"Median early-awareness lead       : "
        f"{strong['early_awareness_lead_cycles'].median():.1f} cycles"
    )

    print(
        f"Median confirmation delay         : "
        f"{strong['confirmation_delay_cycles'].median():.1f} cycles"
    )

    print(
        f"Median maintenance opportunity    : "
        f"{strong['maintenance_opportunity_cycles'].median():.1f} cycles"
    )

    print(
        f"Median pre-fault sequence remaining: "
        f"{strong['maintenance_opportunity_pct'].median():.2f}%"
    )


# =============================================================================
# 27. NULL PROFILE
# =============================================================================

null_profile = (
    tableau
    .isna()
    .sum()
)

null_profile = null_profile[
    null_profile > 0
]


print(
    "\nEXPECTED ANALYTICAL NULLS"
)

print("-" * 120)


if len(null_profile) > 0:

    print(
        null_profile.to_string()
    )

else:

    print(
        "No null values found."
    )


print(
    "\nNOTE:"
)

print(
    "Null timing values are intentional where an operation did not "
    "generate the corresponding persistent warning or maintenance action."
)


# =============================================================================
# 28. INTERPRETATION GUARDRAIL
# =============================================================================

print(
    "\nINTERPRETATION GUARDRAIL"
)

print("-" * 120)

print(
    "Maintenance-opportunity and warning-lead fields are measured "
    "in observed operating cycles relative to the dataset-defined "
    "evident-fault boundary."
)

print(
    "They are not Remaining Useful Life (RUL), probability of failure, "
    "component life, or time-to-physical-failure estimates."
)

print(
    "Probe utility is an analytical synthesis score, not a probability."
)


# =============================================================================
# 29. OUTPUT
# =============================================================================

print(
    f"\nTableau analytical dataset saved to:\n"
    f"{OUTPUT_FILE}"
)


print("\n" + "=" * 120)

print(
    "FINAL TABLEAU ANALYTICAL DATASET COMPLETE"
)

print("=" * 120)