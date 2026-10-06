from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hp\Industrial-Analytics"
    r"\Mitsubishi-ME-AD-Predictive-Maintenance"
)

HEALTH_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_multijoint_health_indicator.csv"
)

ANOMALY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_isolation_forest_cycle_scores.csv"
)

OUTPUT_CYCLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_anomaly_convergence_cycles.csv"
)

OUTPUT_OPERATIONS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_anomaly_convergence_by_operation.csv"
)

OUTPUT_GROUPS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_anomaly_convergence_by_group.csv"
)

OUTPUT_HEALTH_BANDS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_anomaly_by_health_band.csv"
)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

HEALTH_THRESHOLD_QUANTILE = 0.05

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

health = pd.read_csv(
    HEALTH_FILE,
    dtype={"operation_code": str},
)

anomaly = pd.read_csv(
    ANOMALY_FILE,
    dtype={"operation_code": str},
)


print("=" * 120)
print("ME-AD PHASE 06.2 — HEALTH–ANOMALY CONVERGENCE ANALYSIS")
print("=" * 120)

print(
    f"\nHealth cycles loaded  : "
    f"{len(health):,}"
)

print(
    f"Anomaly cycles loaded : "
    f"{len(anomaly):,}"
)


# =============================================================================
# 4. PREPARE ANOMALY SCORE
# =============================================================================

anomaly[
    "if_threshold_ratio"
] = (
    anomaly[
        "if_anomaly_score"
    ]
    /
    anomaly[
        "if_anomaly_threshold"
    ]
)


# =============================================================================
# 5. MERGE INDEPENDENT ANALYTICAL PATHWAYS
# =============================================================================

anomaly_keep = anomaly[
    [
        "operation_code",
        "cycle_index",
        "if_anomaly_score",
        "if_anomaly_threshold",
        "if_threshold_ratio",
        "if_anomaly_flag",
    ]
].copy()

merged = health.merge(
    anomaly_keep,
    on=[
        "operation_code",
        "cycle_index",
    ],
    how="inner",
    validate="one_to_one",
)


print(
    f"Merged cycles         : "
    f"{len(merged):,}"
)

if len(merged) != len(health):

    print(
        "WARNING: merged cycle count differs "
        "from health-indicator table."
    )


# =============================================================================
# 6. DIAGNOSTIC GROUP
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


merged[
    "diagnostic_group"
] = (
    merged[
        "operation_code"
    ]
    .apply(
        diagnostic_group
    )
)


# =============================================================================
# 7. OPERATION-SPECIFIC HEALTH WARNING THRESHOLD
# =============================================================================

threshold_records = []

for operation_code, group in merged.groupby(
    "operation_code"
):

    training = group[
        group[
            "benchmark_health_region"
        ] == "training_healthy"
    ]

    threshold = (
        training[
            "health_index"
        ]
        .quantile(
            HEALTH_THRESHOLD_QUANTILE
        )
    )

    threshold_records.append(
        {
            "operation_code":
                operation_code,

            "health_warning_threshold":
                threshold,
        }
    )


health_thresholds = pd.DataFrame(
    threshold_records
)

merged = merged.merge(
    health_thresholds,
    on="operation_code",
    how="left",
    validate="many_to_one",
)

merged[
    "health_low_flag"
] = (
    merged[
        "health_index"
    ]
    <
    merged[
        "health_warning_threshold"
    ]
)


# =============================================================================
# 8. HEALTH BANDS
# =============================================================================

# These bands are descriptive only.
# They do NOT represent failure probabilities.

merged[
    "health_band"
] = pd.cut(
    merged[
        "health_index"
    ],
    bins=[
        -np.inf,
        30,
        50,
        70,
        85,
        np.inf,
    ],
    labels=[
        "<30",
        "30-50",
        "50-70",
        "70-85",
        "85+",
    ],
    right=False,
)


# =============================================================================
# 9. CYCLE-LEVEL AGREEMENT STATE
# =============================================================================

conditions = [
    (
        merged["health_low_flag"]
        &
        merged["if_anomaly_flag"]
    ),
    (
        merged["health_low_flag"]
        &
        ~merged["if_anomaly_flag"]
    ),
    (
        ~merged["health_low_flag"]
        &
        merged["if_anomaly_flag"]
    ),
]

choices = [
    "both_abnormal",
    "health_only",
    "if_only",
]

merged[
    "agreement_state"
] = np.select(
    conditions,
    choices,
    default="both_normal",
)


# =============================================================================
# 10. OPERATION-LEVEL CORRELATION AND AGREEMENT
# =============================================================================

operation_records = []

for operation_code, group in merged.groupby(
    "operation_code"
):

    rho_score, p_score = spearmanr(
        group[
            "health_index"
        ],
        group[
            "if_anomaly_score"
        ],
    )

    rho_ratio, p_ratio = spearmanr(
        group[
            "health_index"
        ],
        group[
            "if_threshold_ratio"
        ],
    )

    health_low = group[
        "health_low_flag"
    ]

    if_abnormal = group[
        "if_anomaly_flag"
    ]

    both_abnormal = (
        health_low
        &
        if_abnormal
    ).sum()

    health_only = (
        health_low
        &
        ~if_abnormal
    ).sum()

    if_only = (
        ~health_low
        &
        if_abnormal
    ).sum()

    both_normal = (
        ~health_low
        &
        ~if_abnormal
    ).sum()

    total = len(group)

    overall_agreement = (
        (
            both_abnormal
            +
            both_normal
        )
        /
        total
    )

    # Of cycles called low-health by the physics-informed indicator,
    # how many are also anomalous to Isolation Forest?
    health_to_if_overlap = (
        both_abnormal
        /
        (
            both_abnormal
            +
            health_only
        )
        if (
            both_abnormal
            +
            health_only
        ) > 0
        else np.nan
    )

    # Of IF anomalies, how many also have low health?
    if_to_health_overlap = (
        both_abnormal
        /
        (
            both_abnormal
            +
            if_only
        )
        if (
            both_abnormal
            +
            if_only
        ) > 0
        else np.nan
    )

    operation_records.append(
        {
            "operation_code":
                operation_code,

            "diagnostic_group":
                group[
                    "diagnostic_group"
                ].iloc[0],

            "cycles":
                total,

            "spearman_health_vs_if_score":
                rho_score,

            "spearman_score_p_value":
                p_score,

            "spearman_health_vs_if_ratio":
                rho_ratio,

            "spearman_ratio_p_value":
                p_ratio,

            "both_abnormal":
                both_abnormal,

            "health_only":
                health_only,

            "if_only":
                if_only,

            "both_normal":
                both_normal,

            "overall_agreement_pct":
                100
                * overall_agreement,

            "health_to_if_overlap_pct":
                100
                * health_to_if_overlap,

            "if_to_health_overlap_pct":
                100
                * if_to_health_overlap,
        }
    )


operation_summary = pd.DataFrame(
    operation_records
)

operation_summary.to_csv(
    OUTPUT_OPERATIONS,
    index=False,
)


# =============================================================================
# 11. PRINT OPERATION RESULTS
# =============================================================================

print(
    "\nOPERATION-LEVEL HEALTH–ANOMALY CONVERGENCE"
)
print("-" * 120)

print(
    operation_summary[
        [
            "operation_code",
            "diagnostic_group",
            "spearman_health_vs_if_ratio",
            "spearman_ratio_p_value",
            "overall_agreement_pct",
            "health_to_if_overlap_pct",
            "if_to_health_overlap_pct",
        ]
    ]
    .sort_values(
        "spearman_health_vs_if_ratio"
    )
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 12. DIAGNOSTIC-GROUP CORRELATION
# =============================================================================

group_records = []

for diagnostic_group_name, group in merged.groupby(
    "diagnostic_group"
):

    rho, p_value = spearmanr(
        group[
            "health_index"
        ],
        group[
            "if_threshold_ratio"
        ],
    )

    state_counts = (
        group[
            "agreement_state"
        ]
        .value_counts()
    )

    both_abnormal = int(
        state_counts.get(
            "both_abnormal",
            0,
        )
    )

    health_only = int(
        state_counts.get(
            "health_only",
            0,
        )
    )

    if_only = int(
        state_counts.get(
            "if_only",
            0,
        )
    )

    both_normal = int(
        state_counts.get(
            "both_normal",
            0,
        )
    )

    total = len(group)

    group_records.append(
        {
            "diagnostic_group":
                diagnostic_group_name,

            "operations":
                group[
                    "operation_code"
                ].nunique(),

            "cycles":
                total,

            "spearman_health_vs_if_ratio":
                rho,

            "spearman_p_value":
                p_value,

            "both_abnormal":
                both_abnormal,

            "health_only":
                health_only,

            "if_only":
                if_only,

            "both_normal":
                both_normal,

            "overall_agreement_pct":
                100
                * (
                    both_abnormal
                    +
                    both_normal
                )
                / total,

            "health_to_if_overlap_pct":
                100
                * both_abnormal
                / (
                    both_abnormal
                    +
                    health_only
                )
                if (
                    both_abnormal
                    +
                    health_only
                ) > 0
                else np.nan,

            "if_to_health_overlap_pct":
                100
                * both_abnormal
                / (
                    both_abnormal
                    +
                    if_only
                )
                if (
                    both_abnormal
                    +
                    if_only
                ) > 0
                else np.nan,
        }
    )


group_summary = pd.DataFrame(
    group_records
)

group_summary.to_csv(
    OUTPUT_GROUPS,
    index=False,
)


print(
    "\nDIAGNOSTIC-GROUP CONVERGENCE"
)
print("-" * 120)

print(
    group_summary
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 13. ANOMALY BEHAVIOUR BY HEALTH BAND
# =============================================================================

health_band_summary = (
    merged
    .groupby(
        "health_band",
        observed=True,
    )
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),

        median_health=(
            "health_index",
            "median",
        ),

        median_if_threshold_ratio=(
            "if_threshold_ratio",
            "median",
        ),

        anomaly_rate=(
            "if_anomaly_flag",
            "mean",
        ),
    )
    .reset_index()
)

health_band_summary[
    "anomaly_rate_pct"
] = (
    100
    * health_band_summary[
        "anomaly_rate"
    ]
)

health_band_summary.to_csv(
    OUTPUT_HEALTH_BANDS,
    index=False,
)


print(
    "\nISOLATION FOREST BEHAVIOUR ACROSS HEALTH BANDS"
)
print("-" * 120)

print(
    health_band_summary[
        [
            "health_band",
            "cycles",
            "median_health",
            "median_if_threshold_ratio",
            "anomaly_rate_pct",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 14. BENCHMARK REGION CONVERGENCE
# =============================================================================

region_summary = (
    merged
    .groupby(
        "benchmark_health_region"
    )
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),

        median_health=(
            "health_index",
            "median",
        ),

        median_if_ratio=(
            "if_threshold_ratio",
            "median",
        ),

        health_low_rate=(
            "health_low_flag",
            "mean",
        ),

        if_anomaly_rate=(
            "if_anomaly_flag",
            "mean",
        ),
    )
    .reset_index()
)

region_summary[
    "health_low_rate_pct"
] = (
    100
    * region_summary[
        "health_low_rate"
    ]
)

region_summary[
    "if_anomaly_rate_pct"
] = (
    100
    * region_summary[
        "if_anomaly_rate"
    ]
)


print(
    "\nBENCHMARK-REGION CONVERGENCE"
)
print("-" * 120)

print(
    region_summary[
        [
            "benchmark_health_region",
            "cycles",
            "median_health",
            "median_if_ratio",
            "health_low_rate_pct",
            "if_anomaly_rate_pct",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 15. SAVE CYCLE-LEVEL CONVERGENCE TABLE
# =============================================================================

output_columns = [
    "operation_code",
    "cycle_index",
    "benchmark_health_region",
    "sequence_position",
    "diagnostic_group",
    "health_index",
    "health_warning_threshold",
    "health_low_flag",
    "health_band",
    "if_anomaly_score",
    "if_anomaly_threshold",
    "if_threshold_ratio",
    "if_anomaly_flag",
    "agreement_state",
]

merged[
    output_columns
].to_csv(
    OUTPUT_CYCLES,
    index=False,
)


# =============================================================================
# 16. POOLED CORRELATION
# =============================================================================

pooled_rho, pooled_p = spearmanr(
    merged[
        "health_index"
    ],
    merged[
        "if_threshold_ratio"
    ],
)


print(
    "\nPOOLED HEALTH–ANOMALY ASSOCIATION"
)
print("-" * 120)

print(
    f"Spearman rho : "
    f"{pooled_rho:.4f}"
)

print(
    f"p-value      : "
    f"{pooled_p:.6g}"
)


# =============================================================================
# 17. OUTPUTS
# =============================================================================

print(
    f"\nCycle-level convergence saved to:\n"
    f"{OUTPUT_CYCLES}"
)

print(
    f"\nOperation convergence saved to:\n"
    f"{OUTPUT_OPERATIONS}"
)

print(
    f"\nDiagnostic-group convergence saved to:\n"
    f"{OUTPUT_GROUPS}"
)

print(
    f"\nHealth-band anomaly summary saved to:\n"
    f"{OUTPUT_HEALTH_BANDS}"
)

print("\n" + "=" * 120)
print("PHASE 06.2 HEALTH–ANOMALY CONVERGENCE ANALYSIS COMPLETE")
print("=" * 120)