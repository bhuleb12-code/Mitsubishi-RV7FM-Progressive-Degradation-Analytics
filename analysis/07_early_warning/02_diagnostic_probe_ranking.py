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

HEALTH_ROBUSTNESS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_warning_robustness_by_operation.csv"
)

IF_TIMING_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_if_health_warning_timing_comparison.csv"
)

IF_OPERATION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_isolation_forest_operation_summary.csv"
)

ALERT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_maintenance_alert_summary.csv"
)

OUTPUT_RANKING = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_diagnostic_probe_ranking.csv"
)

OUTPUT_RECOMMENDATIONS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_diagnostic_probe_recommendations.csv"
)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

health = pd.read_csv(
    HEALTH_ROBUSTNESS_FILE,
    dtype={"operation_code": str},
)

timing = pd.read_csv(
    IF_TIMING_FILE,
    dtype={"operation_code": str},
)

if_summary = pd.read_csv(
    IF_OPERATION_FILE,
    dtype={"operation_code": str},
)

alerts = pd.read_csv(
    ALERT_SUMMARY_FILE,
    dtype={"operation_code": str},
)


print("=" * 120)
print(
    "ME-AD PHASE 07.2 — DIAGNOSTIC PROBE SELECTION "
    "& MAINTENANCE OBSERVABILITY RANKING"
)
print("=" * 120)

print(
    f"\nHealth robustness operations : "
    f"{health['operation_code'].nunique()}"
)

print(
    f"IF timing operations         : "
    f"{timing['operation_code'].nunique()}"
)

print(
    f"IF summary operations        : "
    f"{if_summary['operation_code'].nunique()}"
)

print(
    f"Alert summary operations     : "
    f"{alerts['operation_code'].nunique()}"
)


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
    health,
    [
        "operation_code",
        "detection_rate_pct",
        "median_warning_lead",
    ],
    "Health robustness table",
)

require_columns(
    timing,
    [
        "operation_code",
        "diagnostic_group",
        "if_warning_detected",
        "if_warning_lead_cycles",
        "health_warning_detected",
        "health_warning_lead_cycles",
    ],
    "IF timing table",
)

require_columns(
    if_summary,
    [
        "operation_code",
        "known_healthy_anomaly_rate_pct",
        "intermediate_anomaly_rate_pct",
        "faulty_anomaly_rate_pct",
    ],
    "IF operation summary",
)

require_columns(
    alerts,
    [
        "operation_code",
        "maintenance_response_category",
        "first_critical_cycle",
        "critical_lead_cycles",
    ],
    "Maintenance alert summary",
)


# =============================================================================
# 4. PREPARE HEALTH ROBUSTNESS
# =============================================================================

health_keep = health[
    [
        "operation_code",
        "detection_rate_pct",
        "median_warning_lead",
    ]
].copy()

health_keep = health_keep.rename(
    columns={
        "detection_rate_pct":
            "health_detection_rate_pct",

        "median_warning_lead":
            "health_robustness_median_lead",
    }
)


# =============================================================================
# 5. PREPARE IF TIMING
# =============================================================================

timing_keep = timing[
    [
        "operation_code",
        "diagnostic_group",
        "if_warning_detected",
        "if_warning_lead_cycles",
        "health_warning_detected",
        "health_warning_lead_cycles",
    ]
].copy()


# =============================================================================
# 6. PREPARE IF OBSERVABILITY
# =============================================================================

if_keep = if_summary[
    [
        "operation_code",
        "known_healthy_anomaly_rate_pct",
        "intermediate_anomaly_rate_pct",
        "faulty_anomaly_rate_pct",
    ]
].copy()


# =============================================================================
# 7. PREPARE MAINTENANCE ALERT EVIDENCE
# =============================================================================

alert_keep = alerts[
    [
        "operation_code",
        "maintenance_response_category",
        "first_critical_cycle",
        "critical_lead_cycles",
    ]
].copy()


# =============================================================================
# 8. MERGE ALL EVIDENCE
# =============================================================================

ranking = (
    health_keep
    .merge(
        timing_keep,
        on="operation_code",
        how="outer",
        validate="one_to_one",
    )
    .merge(
        if_keep,
        on="operation_code",
        how="outer",
        validate="one_to_one",
    )
    .merge(
        alert_keep,
        on="operation_code",
        how="outer",
        validate="one_to_one",
    )
)


print(
    f"\nMerged operations            : "
    f"{ranking['operation_code'].nunique()}"
)

if ranking["operation_code"].nunique() != 16:

    print(
        "\nWARNING: expected 16 unique operations, "
        f"found {ranking['operation_code'].nunique()}."
    )


# =============================================================================
# 9. COMPONENT 1 — HEALTH-WARNING ROBUSTNESS
# =============================================================================

# Phase 05.3 tested each operation across multiple warning-policy
# configurations.
#
# detection_rate_pct measures how consistently an operation generated
# a health warning across those configurations.
#
# Convert from 0–100% to a 0–1 component score.

ranking[
    "health_robustness_score"
] = (
    ranking[
        "health_detection_rate_pct"
    ]
    .fillna(0)
    / 100.0
)


# =============================================================================
# 10. COMPONENT 2 — PERSISTENT IF WARNING
# =============================================================================

# Phase 06.3 already applied a demanding persistence rule:
#
#   50-cycle windows
#   >= 70% anomalous cycles
#   3 consecutive qualifying windows
#
# Therefore the existence of an IF warning is treated as a binary
# persistent-abnormality evidence component.

ranking[
    "if_persistence_score"
] = (
    ranking[
        "if_warning_detected"
    ]
    .fillna(False)
    .astype(float)
)


# =============================================================================
# 11. COMPONENT 3 — INDEPENDENT SIGNAL CONVERGENCE
# =============================================================================

health_warn = (
    ranking[
        "health_warning_detected"
    ]
    .fillna(False)
    .astype(bool)
)

if_warn = (
    ranking[
        "if_warning_detected"
    ]
    .fillna(False)
    .astype(bool)
)


# 1.0 = both pathways warn
# 0.5 = one pathway warns
# 0.0 = neither pathway warns

ranking[
    "convergence_score"
] = np.select(
    [
        health_warn & if_warn,
        health_warn ^ if_warn,
    ],
    [
        1.0,
        0.5,
    ],
    default=0.0,
)


# =============================================================================
# 12. COMPONENT 4 — EFFECTIVE WARNING LEAD
# =============================================================================

# For operations where both systems warn:
# use the CRITICAL lead from Phase 07.1.
#
# This represents the point where both independent analytical pathways
# have converged.
#
# For single-signal operations:
# retain their warning lead but discount it by 50% because independent
# confirmation is absent.
#
# No warning:
# zero lead evidence.

ranking[
    "effective_warning_lead"
] = 0.0


both = (
    health_warn
    &
    if_warn
)

health_only = (
    health_warn
    &
    ~if_warn
)

if_only = (
    ~health_warn
    &
    if_warn
)


ranking.loc[
    both,
    "effective_warning_lead",
] = (
    ranking.loc[
        both,
        "critical_lead_cycles",
    ]
    .fillna(
        ranking.loc[
            both,
            "health_warning_lead_cycles",
        ]
    )
)


ranking.loc[
    health_only,
    "effective_warning_lead",
] = (
    0.5
    *
    ranking.loc[
        health_only,
        "health_warning_lead_cycles",
    ]
    .fillna(0)
)


ranking.loc[
    if_only,
    "effective_warning_lead",
] = (
    0.5
    *
    ranking.loc[
        if_only,
        "if_warning_lead_cycles",
    ]
    .fillna(0)
)


max_effective_lead = (
    ranking[
        "effective_warning_lead"
    ]
    .max()
)


if max_effective_lead > 0:

    ranking[
        "warning_lead_score"
    ] = (
        ranking[
            "effective_warning_lead"
        ]
        / max_effective_lead
    )

else:

    ranking[
        "warning_lead_score"
    ] = 0.0


# =============================================================================
# 13. SUPPORTING FAULT-OBSERVABILITY METRICS
# =============================================================================

# These metrics are retained as supporting evidence.
# They are NOT directly included in the composite score because the
# evident-fault benchmark should validate probe usefulness rather than
# dominate the selection rule.

ranking[
    "fault_minus_healthy_anomaly_pct_points"
] = (
    ranking[
        "faulty_anomaly_rate_pct"
    ]
    -
    ranking[
        "known_healthy_anomaly_rate_pct"
    ]
)


ranking[
    "intermediate_minus_healthy_anomaly_pct_points"
] = (
    ranking[
        "intermediate_anomaly_rate_pct"
    ]
    -
    ranking[
        "known_healthy_anomaly_rate_pct"
    ]
)


# =============================================================================
# 14. COMPOSITE DIAGNOSTIC-PROBE UTILITY
# =============================================================================

# Equal weighting:
#
#   25% health-warning robustness
#   25% persistent IF evidence
#   25% independent convergence
#   25% warning lead
#
# Equal weighting is deliberately transparent. We do not currently
# have empirical evidence supporting more precise business weights.

component_columns = [
    "health_robustness_score",
    "if_persistence_score",
    "convergence_score",
    "warning_lead_score",
]


ranking[
    "probe_utility_score"
] = (
    ranking[
        component_columns
    ]
    .mean(
        axis=1
    )
)


ranking[
    "probe_utility_pct"
] = (
    100
    * ranking[
        "probe_utility_score"
    ]
)


# =============================================================================
# 15. RANK OPERATIONS
# =============================================================================

ranking = ranking.sort_values(
    [
        "probe_utility_score",
        "convergence_score",
        "health_robustness_score",
        "warning_lead_score",
    ],
    ascending=[
        False,
        False,
        False,
        False,
    ],
).reset_index(
    drop=True
)


ranking[
    "probe_rank"
] = np.arange(
    1,
    len(ranking) + 1,
)


# =============================================================================
# 16. ASSIGN DIAGNOSTIC-PROBE TIER
# =============================================================================

def assign_probe_tier(row):

    # PRIMARY:
    # Both analytical pathways warn and the physics-informed warning
    # is highly robust across Phase 05 sensitivity configurations.

    if (
        row[
            "convergence_score"
        ] == 1.0
        and
        row[
            "health_robustness_score"
        ] >= 0.90
    ):
        return "PRIMARY"

    # SECONDARY:
    # Useful overall evidence, but not as strongly corroborated.

    if (
        row[
            "probe_utility_score"
        ] >= 0.50
    ):
        return "SECONDARY"

    # SUPPORTING:
    # Some diagnostic evidence exists, but the operation should not
    # be relied upon as a standalone health probe.

    if (
        row[
            "probe_utility_score"
        ] > 0
    ):
        return "SUPPORTING"

    # LOW_OBSERVABILITY:
    # Neither analytical pathway provides useful persistent warning.

    return "LOW_OBSERVABILITY"


ranking[
    "probe_tier"
] = ranking.apply(
    assign_probe_tier,
    axis=1,
)


# =============================================================================
# 17. MAINTENANCE-USE RECOMMENDATION
# =============================================================================

def maintenance_use(row):

    tier = row[
        "probe_tier"
    ]

    if tier == "PRIMARY":

        return (
            "Preferred recurring diagnostic health-probe trajectory"
        )

    if tier == "SECONDARY":

        return (
            "Useful secondary condition-monitoring trajectory"
        )

    if tier == "SUPPORTING":

        return (
            "Use as supporting evidence; do not rely on alone"
        )

    return (
        "Poor standalone observability for this degradation mechanism"
    )


ranking[
    "recommended_maintenance_use"
] = ranking.apply(
    maintenance_use,
    axis=1,
)


# =============================================================================
# 18. SAVE FULL RANKING
# =============================================================================

ranking.to_csv(
    OUTPUT_RANKING,
    index=False,
)


recommendation_columns = [
    "probe_rank",
    "operation_code",
    "diagnostic_group",
    "probe_tier",
    "probe_utility_pct",
    "health_detection_rate_pct",
    "health_robustness_median_lead",
    "if_warning_detected",
    "health_warning_detected",
    "if_warning_lead_cycles",
    "health_warning_lead_cycles",
    "critical_lead_cycles",
    "faulty_anomaly_rate_pct",
    "intermediate_anomaly_rate_pct",
    "maintenance_response_category",
    "recommended_maintenance_use",
]


ranking[
    recommendation_columns
].to_csv(
    OUTPUT_RECOMMENDATIONS,
    index=False,
)


# =============================================================================
# 19. PRINT DIAGNOSTIC-PROBE RANKING
# =============================================================================

print(
    "\nDIAGNOSTIC PROBE RANKING"
)

print("-" * 120)


print(
    ranking[
        [
            "probe_rank",
            "operation_code",
            "diagnostic_group",
            "probe_tier",
            "health_robustness_score",
            "if_persistence_score",
            "convergence_score",
            "warning_lead_score",
            "probe_utility_pct",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 20. SUPPORTING FAULT-OBSERVABILITY EVIDENCE
# =============================================================================

print(
    "\nFAULT OBSERVABILITY SUPPORT"
)

print("-" * 120)


print(
    ranking[
        [
            "probe_rank",
            "operation_code",
            "known_healthy_anomaly_rate_pct",
            "intermediate_anomaly_rate_pct",
            "faulty_anomaly_rate_pct",
            "intermediate_minus_healthy_anomaly_pct_points",
            "fault_minus_healthy_anomaly_pct_points",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 21. TIER SUMMARY
# =============================================================================

tier_summary = (
    ranking[
        "probe_tier"
    ]
    .value_counts()
)


print(
    "\nPROBE TIER COUNTS"
)

print("-" * 120)

print(
    tier_summary.to_string()
)


# =============================================================================
# 22. PRIMARY DIAGNOSTIC PROBES
# =============================================================================

primary = ranking[
    ranking[
        "probe_tier"
    ] == "PRIMARY"
].copy()


print(
    "\nPRIMARY DIAGNOSTIC HEALTH PROBES"
)

print("-" * 120)


if len(primary) > 0:

    print(
        primary[
            [
                "probe_rank",
                "operation_code",
                "probe_utility_pct",
                "health_detection_rate_pct",
                "health_robustness_median_lead",
                "critical_lead_cycles",
                "faulty_anomaly_rate_pct",
                "recommended_maintenance_use",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No operations satisfied the PRIMARY criteria."
    )


# =============================================================================
# 23. SECONDARY DIAGNOSTIC PROBES
# =============================================================================

secondary = ranking[
    ranking[
        "probe_tier"
    ] == "SECONDARY"
].copy()


print(
    "\nSECONDARY DIAGNOSTIC HEALTH PROBES"
)

print("-" * 120)


if len(secondary) > 0:

    print(
        secondary[
            [
                "probe_rank",
                "operation_code",
                "probe_utility_pct",
                "health_detection_rate_pct",
                "if_warning_detected",
                "health_warning_detected",
                "recommended_maintenance_use",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No operations classified as SECONDARY."
    )


# =============================================================================
# 24. SUPPORTING DIAGNOSTIC OPERATIONS
# =============================================================================

supporting = ranking[
    ranking[
        "probe_tier"
    ] == "SUPPORTING"
].copy()


print(
    "\nSUPPORTING DIAGNOSTIC OPERATIONS"
)

print("-" * 120)


if len(supporting) > 0:

    print(
        supporting[
            [
                "probe_rank",
                "operation_code",
                "probe_utility_pct",
                "health_detection_rate_pct",
                "if_warning_detected",
                "health_warning_detected",
                "recommended_maintenance_use",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No operations classified as SUPPORTING."
    )


# =============================================================================
# 25. LOW-OBSERVABILITY OPERATIONS
# =============================================================================

low = ranking[
    ranking[
        "probe_tier"
    ] == "LOW_OBSERVABILITY"
].copy()


print(
    "\nLOW-OBSERVABILITY OPERATIONS"
)

print("-" * 120)


if len(low) > 0:

    print(
        low[
            [
                "operation_code",
                "diagnostic_group",
                "health_detection_rate_pct",
                "faulty_anomaly_rate_pct",
                "recommended_maintenance_use",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No operations classified as LOW_OBSERVABILITY."
    )


# =============================================================================
# 26. TOP-PROBE SUMMARY
# =============================================================================

top_probe = ranking.iloc[0]


print(
    "\nTOP DIAGNOSTIC-PROBE CANDIDATE"
)

print("-" * 120)

print(
    f"Operation                     : "
    f"{top_probe['operation_code']}"
)

print(
    f"Probe rank                    : "
    f"{int(top_probe['probe_rank'])}"
)

print(
    f"Probe tier                    : "
    f"{top_probe['probe_tier']}"
)

print(
    f"Probe utility                 : "
    f"{top_probe['probe_utility_pct']:.2f}%"
)

print(
    f"Health robustness             : "
    f"{top_probe['health_detection_rate_pct']:.2f}%"
)

print(
    f"Persistent IF warning         : "
    f"{top_probe['if_warning_detected']}"
)

print(
    f"Persistent health warning     : "
    f"{top_probe['health_warning_detected']}"
)

if pd.notna(
    top_probe[
        "critical_lead_cycles"
    ]
):

    print(
        f"Convergent warning lead       : "
        f"{top_probe['critical_lead_cycles']:.1f} cycles"
    )

print(
    f"Evident-fault anomaly rate    : "
    f"{top_probe['faulty_anomaly_rate_pct']:.2f}%"
)


# =============================================================================
# 27. OUTPUT PATHS
# =============================================================================

print(
    f"\nFull diagnostic-probe ranking saved to:\n"
    f"{OUTPUT_RANKING}"
)

print(
    f"\nMaintenance recommendations saved to:\n"
    f"{OUTPUT_RECOMMENDATIONS}"
)


print("\n" + "=" * 120)
print("PHASE 07.2 DIAGNOSTIC PROBE RANKING COMPLETE")
print("=" * 120)