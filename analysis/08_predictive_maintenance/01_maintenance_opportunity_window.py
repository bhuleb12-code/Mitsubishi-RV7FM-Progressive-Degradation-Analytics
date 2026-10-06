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

DECISION_SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_decision_summary.csv"
)

WARNING_TIMING_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_if_health_warning_timing_comparison.csv"
)

PROBE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_diagnostic_probe_ranking.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_opportunity_windows.csv"
)

OUTPUT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_predictive_maintenance_summary.csv"
)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

decisions = pd.read_csv(
    DECISION_SUMMARY_FILE,
    dtype={"operation_code": str},
)

timing = pd.read_csv(
    WARNING_TIMING_FILE,
    dtype={"operation_code": str},
)

probes = pd.read_csv(
    PROBE_FILE,
    dtype={"operation_code": str},
)


print("=" * 120)
print(
    "ME-AD PHASE 08.1 — MAINTENANCE OPPORTUNITY WINDOW "
    "& PREDICTIVE LEAD QUANTIFICATION"
)
print("=" * 120)

print(
    f"\nDecision operations : "
    f"{decisions['operation_code'].nunique()}"
)

print(
    f"Timing operations   : "
    f"{timing['operation_code'].nunique()}"
)

print(
    f"Probe operations    : "
    f"{probes['operation_code'].nunique()}"
)


# =============================================================================
# 3. VALIDATION
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
    decisions,
    [
        "operation_code",
        "probe_rank",
        "probe_tier",
        "probe_utility_pct",
        "highest_alert",
        "highest_maintenance_action",
        "highest_priority",
        "decision_confidence",
        "first_highest_priority_cycle",
    ],
    "Maintenance decision summary",
)

require_columns(
    timing,
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
    ],
    "Warning timing comparison",
)

require_columns(
    probes,
    [
        "operation_code",
        "probe_rank",
        "probe_tier",
        "probe_utility_pct",
    ],
    "Diagnostic probe ranking",
)


# =============================================================================
# 4. PREPARE TABLES
# =============================================================================

decision_keep = decisions[
    [
        "operation_code",
        "highest_alert",
        "highest_maintenance_action",
        "highest_priority",
        "decision_confidence",
        "first_highest_priority_cycle",
    ]
].copy()


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


# Rename the source field internally for shorter calculations.
# IMPORTANT:
# This remains the dataset-defined evident-fault boundary.
# It is NOT a physical component-failure timestamp.

timing_keep = timing_keep.rename(
    columns={
        "evident_fault_start_cycle": "fault_start_cycle"
    }
)


probe_keep = probes[
    [
        "operation_code",
        "probe_rank",
        "probe_tier",
        "probe_utility_pct",
    ]
].copy()


# =============================================================================
# 5. MERGE EVIDENCE
# =============================================================================

df = (
    timing_keep
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


if df["operation_code"].nunique() != 16:
    raise ValueError(
        "Expected 16 operations after merge, found "
        f"{df['operation_code'].nunique()}."
    )


if df[
    [
        "probe_rank",
        "probe_tier",
        "highest_priority",
    ]
].isna().any().any():
    raise ValueError(
        "Missing probe-ranking or maintenance-decision "
        "information after merge."
    )


print(
    f"Merged operations   : "
    f"{df['operation_code'].nunique()}"
)


# =============================================================================
# 6. FIRST PERSISTENT ANALYTICAL SIGNAL
# =============================================================================

# The health-warning and IF-warning pathways were developed independently.
#
# The earliest persistent warning from either pathway represents the first
# point at which sustained analytical evidence becomes available.
#
# This is an AWARENESS point, not yet necessarily a maintenance decision.

df[
    "first_persistent_signal_cycle"
] = df[
    [
        "health_warning_cycle",
        "if_warning_cycle",
    ]
].min(
    axis=1,
    skipna=True,
)


# =============================================================================
# 7. EARLY-AWARENESS LEAD
# =============================================================================

df[
    "early_awareness_lead_cycles"
] = (
    df[
        "fault_start_cycle"
    ]
    -
    df[
        "first_persistent_signal_cycle"
    ]
)


# Prevent negative values from being interpreted as useful lead.

df.loc[
    df[
        "early_awareness_lead_cycles"
    ] < 0,
    "early_awareness_lead_cycles",
] = np.nan


# =============================================================================
# 8. ACTIONABLE MAINTENANCE CYCLE
# =============================================================================

# Phase 07.3 priority levels:
#
# 0 = routine operation
# 1 = enhanced monitoring
# 2 = targeted inspection
# 3 = maintenance priority
#
# A true maintenance-action opportunity is defined only at priority >= 2.
#
# Priority 1 represents enhanced monitoring and is retained separately.

df[
    "actionable_maintenance_cycle"
] = np.where(
    df[
        "highest_priority"
    ] >= 2,
    df[
        "first_highest_priority_cycle"
    ],
    np.nan,
)


# =============================================================================
# 9. CONFIRMED MAINTENANCE OPPORTUNITY
# =============================================================================

df[
    "maintenance_opportunity_cycles"
] = (
    df[
        "fault_start_cycle"
    ]
    -
    df[
        "actionable_maintenance_cycle"
    ]
)


df.loc[
    df[
        "maintenance_opportunity_cycles"
    ] < 0,
    "maintenance_opportunity_cycles",
] = np.nan


# =============================================================================
# 10. MONITORING-ONLY LEAD
# =============================================================================

# Priority-1 operations provide useful monitoring lead but have not reached
# the evidence threshold required for direct inspection/maintenance action.

df[
    "monitoring_lead_cycles"
] = np.where(
    df[
        "highest_priority"
    ] == 1,
    (
        df[
            "fault_start_cycle"
        ]
        -
        df[
            "first_highest_priority_cycle"
        ]
    ),
    np.nan,
)


df.loc[
    df[
        "monitoring_lead_cycles"
    ] < 0,
    "monitoring_lead_cycles",
] = np.nan


# =============================================================================
# 11. CONFIRMATION DELAY
# =============================================================================

# Difference between:
#
# first persistent analytical awareness
#
# and
#
# first actionable maintenance decision.
#
# Example:
#
# Operation 1053
# first persistent signal = cycle 720
# actionable maintenance = cycle 950
#
# confirmation delay = 230 cycles.

df[
    "confirmation_delay_cycles"
] = (
    df[
        "actionable_maintenance_cycle"
    ]
    -
    df[
        "first_persistent_signal_cycle"
    ]
)


df.loc[
    df[
        "confirmation_delay_cycles"
    ] < 0,
    "confirmation_delay_cycles",
] = np.nan


# =============================================================================
# 12. MAINTENANCE OPPORTUNITY AS OBSERVED SEQUENCE FRACTION
# =============================================================================

# This is NOT Remaining Useful Life.
#
# It simply expresses how much of the observed pre-evident-fault sequence
# remained after the actionable maintenance decision was reached.

df[
    "maintenance_opportunity_fraction"
] = (
    df[
        "maintenance_opportunity_cycles"
    ]
    /
    df[
        "fault_start_cycle"
    ]
)


df[
    "maintenance_opportunity_pct"
] = (
    100.0
    *
    df[
        "maintenance_opportunity_fraction"
    ]
)


# =============================================================================
# 13. PREDICTIVE-MAINTENANCE EVIDENCE CLASS
# =============================================================================

def evidence_class(row):

    # Highest-quality case:
    # PRIMARY diagnostic probe + priority-3 convergent maintenance evidence.

    if (
        row["highest_priority"] == 3
        and
        row["probe_tier"] == "PRIMARY"
    ):
        return "STRONG_ACTIONABLE"

    # Structured degradation evidence sufficient for targeted inspection.

    if row["highest_priority"] == 2:
        return "INSPECTION_ACTIONABLE"

    # Persistent statistical abnormality, but not enough independent
    # confirmation for maintenance intervention.

    if row["highest_priority"] == 1:
        return "MONITORING_ONLY"

    # Poor diagnostic trajectories cannot provide strong reassurance merely
    # because no persistent warning was observed.

    if row["probe_tier"] == "LOW_OBSERVABILITY":
        return "INSUFFICIENT_OBSERVABILITY"

    # Remaining operations have some diagnostic value but did not generate
    # an actionable warning under the current policy.

    return "NO_ACTIONABLE_WARNING"


df[
    "predictive_maintenance_class"
] = df.apply(
    evidence_class,
    axis=1,
)


# =============================================================================
# 14. OPERATIONAL INTERPRETATION
# =============================================================================

def interpretation(row):

    pm_class = row[
        "predictive_maintenance_class"
    ]

    if pm_class == "STRONG_ACTIONABLE":
        return (
            "Persistent independent evidence converges early enough "
            "to support maintenance planning before the "
            "dataset-defined evident-fault region."
        )

    if pm_class == "INSPECTION_ACTIONABLE":
        return (
            "Physics-informed degradation evidence supports targeted "
            "inspection before the dataset-defined evident-fault region."
        )

    if pm_class == "MONITORING_ONLY":
        return (
            "Persistent abnormality supports enhanced monitoring, "
            "but evidence is insufficient for direct maintenance escalation."
        )

    if pm_class == "INSUFFICIENT_OBSERVABILITY":
        return (
            "This trajectory provides insufficient early-warning "
            "observability for standalone predictive-maintenance decisions."
        )

    return (
        "No persistent actionable maintenance warning was established "
        "under the current decision policy."
    )


df[
    "predictive_maintenance_interpretation"
] = df.apply(
    interpretation,
    axis=1,
)


# =============================================================================
# 15. SORT RESULTS
# =============================================================================

class_order = {
    "STRONG_ACTIONABLE": 0,
    "INSPECTION_ACTIONABLE": 1,
    "MONITORING_ONLY": 2,
    "NO_ACTIONABLE_WARNING": 3,
    "INSUFFICIENT_OBSERVABILITY": 4,
}


df[
    "_class_order"
] = (
    df[
        "predictive_maintenance_class"
    ]
    .map(class_order)
)


df = df.sort_values(
    [
        "_class_order",
        "probe_rank",
    ],
    ascending=[
        True,
        True,
    ],
).reset_index(
    drop=True
)


df = df.drop(
    columns=[
        "_class_order",
    ]
)


# =============================================================================
# 16. SAVE OPERATION-LEVEL OPPORTUNITY WINDOWS
# =============================================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =============================================================================
# 17. PREDICTIVE-MAINTENANCE CLASS SUMMARY
# =============================================================================

summary = (
    df.groupby(
        "predictive_maintenance_class",
        dropna=False,
    )
    .agg(
        operations=(
            "operation_code",
            "count",
        ),
        median_early_awareness_lead=(
            "early_awareness_lead_cycles",
            "median",
        ),
        median_maintenance_opportunity=(
            "maintenance_opportunity_cycles",
            "median",
        ),
        min_maintenance_opportunity=(
            "maintenance_opportunity_cycles",
            "min",
        ),
        max_maintenance_opportunity=(
            "maintenance_opportunity_cycles",
            "max",
        ),
        median_monitoring_lead=(
            "monitoring_lead_cycles",
            "median",
        ),
        median_confirmation_delay=(
            "confirmation_delay_cycles",
            "median",
        ),
    )
    .reset_index()
)


summary.to_csv(
    OUTPUT_SUMMARY_FILE,
    index=False,
)


# =============================================================================
# 18. PRINT OPPORTUNITY WINDOWS
# =============================================================================

print(
    "\nPREDICTIVE-MAINTENANCE OPPORTUNITY WINDOWS"
)

print("-" * 120)


print(
    df[
        [
            "operation_code",
            "probe_rank",
            "probe_tier",
            "fault_start_cycle",
            "first_persistent_signal_cycle",
            "early_awareness_lead_cycles",
            "actionable_maintenance_cycle",
            "maintenance_opportunity_cycles",
            "confirmation_delay_cycles",
            "predictive_maintenance_class",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 19. STRONG ACTIONABLE OPERATIONS
# =============================================================================

strong = df[
    df[
        "predictive_maintenance_class"
    ] == "STRONG_ACTIONABLE"
].copy()


print(
    "\nSTRONG ACTIONABLE PREDICTIVE-MAINTENANCE OPERATIONS"
)

print("-" * 120)


if len(strong) > 0:

    print(
        strong[
            [
                "operation_code",
                "probe_rank",
                "probe_utility_pct",
                "early_awareness_lead_cycles",
                "confirmation_delay_cycles",
                "maintenance_opportunity_cycles",
                "maintenance_opportunity_pct",
                "decision_confidence",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No operations classified as STRONG_ACTIONABLE."
    )


# =============================================================================
# 20. INSPECTION-ACTIONABLE OPERATIONS
# =============================================================================

inspection = df[
    df[
        "predictive_maintenance_class"
    ] == "INSPECTION_ACTIONABLE"
].copy()


print(
    "\nINSPECTION-ACTIONABLE OPERATIONS"
)

print("-" * 120)


if len(inspection) > 0:

    print(
        inspection[
            [
                "operation_code",
                "probe_rank",
                "probe_tier",
                "early_awareness_lead_cycles",
                "maintenance_opportunity_cycles",
                "maintenance_opportunity_pct",
                "decision_confidence",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No operations classified as INSPECTION_ACTIONABLE."
    )


# =============================================================================
# 21. MONITORING-ONLY OPERATIONS
# =============================================================================

monitoring = df[
    df[
        "predictive_maintenance_class"
    ] == "MONITORING_ONLY"
].copy()


print(
    "\nMONITORING-ONLY OPERATIONS"
)

print("-" * 120)


if len(monitoring) > 0:

    print(
        monitoring[
            [
                "operation_code",
                "probe_tier",
                "first_highest_priority_cycle",
                "monitoring_lead_cycles",
                "decision_confidence",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No operations classified as MONITORING_ONLY."
    )


# =============================================================================
# 22. CLASS SUMMARY
# =============================================================================

print(
    "\nPREDICTIVE-MAINTENANCE CLASS SUMMARY"
)

print("-" * 120)


print(
    summary
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 23. PRIMARY DIAGNOSTIC-PROBE MAINTENANCE WINDOWS
# =============================================================================

primary = df[
    df[
        "probe_tier"
    ] == "PRIMARY"
].copy()


print(
    "\nPRIMARY DIAGNOSTIC-PROBE MAINTENANCE WINDOWS"
)

print("-" * 120)


print(
    primary[
        [
            "operation_code",
            "probe_rank",
            "fault_start_cycle",
            "first_persistent_signal_cycle",
            "actionable_maintenance_cycle",
            "early_awareness_lead_cycles",
            "confirmation_delay_cycles",
            "maintenance_opportunity_cycles",
            "maintenance_opportunity_pct",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 24. KEY PREDICTIVE-MAINTENANCE KPI
# =============================================================================

strong_windows = (
    strong[
        "maintenance_opportunity_cycles"
    ]
    .dropna()
)


strong_awareness = (
    strong[
        "early_awareness_lead_cycles"
    ]
    .dropna()
)


strong_confirmation = (
    strong[
        "confirmation_delay_cycles"
    ]
    .dropna()
)


print(
    "\nKEY PREDICTIVE-MAINTENANCE KPI"
)

print("-" * 120)


if len(strong_windows) > 0:

    print(
        f"Strong actionable operations      : "
        f"{len(strong_windows)}"
    )

    print(
        f"Median early-awareness lead       : "
        f"{strong_awareness.median():.1f} cycles"
    )

    print(
        f"Median confirmation delay         : "
        f"{strong_confirmation.median():.1f} cycles"
    )

    print(
        f"Median maintenance opportunity    : "
        f"{strong_windows.median():.1f} cycles"
    )

    print(
        f"Minimum maintenance opportunity   : "
        f"{strong_windows.min():.1f} cycles"
    )

    print(
        f"Maximum maintenance opportunity   : "
        f"{strong_windows.max():.1f} cycles"
    )

else:

    print(
        "No strong actionable maintenance windows available."
    )


# =============================================================================
# 25. TOP DIAGNOSTIC PROBE
# =============================================================================

top_probe = (
    df.sort_values(
        "probe_rank"
    )
    .iloc[0]
)


print(
    "\nTOP DIAGNOSTIC-PROBE PREDICTIVE-MAINTENANCE WINDOW"
)

print("-" * 120)


print(
    f"Operation                        : "
    f"{top_probe['operation_code']}"
)

print(
    f"Probe rank                       : "
    f"{int(top_probe['probe_rank'])}"
)

print(
    f"Probe tier                       : "
    f"{top_probe['probe_tier']}"
)

print(
    f"Evident-fault boundary           : "
    f"{top_probe['fault_start_cycle']:.0f}"
)

if pd.notna(
    top_probe[
        "first_persistent_signal_cycle"
    ]
):
    print(
        f"First persistent signal          : "
        f"{top_probe['first_persistent_signal_cycle']:.0f}"
    )

if pd.notna(
    top_probe[
        "early_awareness_lead_cycles"
    ]
):
    print(
        f"Early-awareness lead             : "
        f"{top_probe['early_awareness_lead_cycles']:.0f} cycles"
    )

if pd.notna(
    top_probe[
        "actionable_maintenance_cycle"
    ]
):
    print(
        f"Actionable maintenance cycle     : "
        f"{top_probe['actionable_maintenance_cycle']:.0f}"
    )

if pd.notna(
    top_probe[
        "confirmation_delay_cycles"
    ]
):
    print(
        f"Confirmation delay               : "
        f"{top_probe['confirmation_delay_cycles']:.0f} cycles"
    )

if pd.notna(
    top_probe[
        "maintenance_opportunity_cycles"
    ]
):
    print(
        f"Confirmed maintenance opportunity: "
        f"{top_probe['maintenance_opportunity_cycles']:.0f} cycles"
    )

if pd.notna(
    top_probe[
        "maintenance_opportunity_pct"
    ]
):
    print(
        f"Observed pre-fault sequence left : "
        f"{top_probe['maintenance_opportunity_pct']:.2f}%"
    )


# =============================================================================
# 26. IMPORTANT SEMANTIC GUARDRAIL
# =============================================================================

print(
    "\nINTERPRETATION GUARDRAIL"
)

print("-" * 120)

print(
    "Maintenance opportunity cycles measure observed analytical lead "
    "before the dataset-defined evident-fault region."
)

print(
    "They are NOT Remaining Useful Life (RUL), component life, "
    "or time-to-physical-failure estimates."
)


# =============================================================================
# 27. OUTPUTS
# =============================================================================

print(
    f"\nOperation-level opportunity windows saved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nPredictive-maintenance summary saved to:\n"
    f"{OUTPUT_SUMMARY_FILE}"
)


print("\n" + "=" * 120)

print(
    "PHASE 08.1 MAINTENANCE OPPORTUNITY WINDOW ANALYSIS COMPLETE"
)

print("=" * 120)