from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


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
    / "me_ad_all_joint_degradation_features.csv"
)

OUTPUT_CYCLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_isolation_forest_cycle_scores.csv"
)

OUTPUT_REGIONS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_isolation_forest_region_summary.csv"
)

OUTPUT_OPERATIONS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_isolation_forest_operation_summary.csv"
)

OUTPUT_TRAJECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_isolation_forest_trajectory.csv"
)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

JOINTS = range(1, 7)

EFFORT_METRICS = [
    "tau_abs_mean",
    "tau_rms",
    "tau_peak",
    "tau_std",
]

FEATURES = [
    f"j{joint}_{metric}"
    for joint in JOINTS
    for metric in EFFORT_METRICS
]

RANDOM_STATE = 42

N_ESTIMATORS = 500

# No assumed contamination rate.
# We calibrate abnormality empirically from healthy model scores instead.
MODEL_CONTAMINATION = "auto"

# A cycle becomes anomalous if its anomaly score exceeds the 95th percentile
# of scores observed in that operation's healthy training data.
HEALTHY_SCORE_QUANTILE = 0.95

N_SEQUENCE_BINS = 20


# =============================================================================
# 3. LOAD DATA
# =============================================================================

df = pd.read_csv(
    INPUT_FILE,
    dtype={"operation_code": str},
)

df = df[
    df["structural_quality"] == "usable"
].copy()

df = df.sort_values(
    [
        "operation_code",
        "cycle_index",
    ]
).reset_index(
    drop=True
)


print("=" * 118)
print("ME-AD PHASE 06.1 — OPERATION-AWARE UNSUPERVISED ANOMALY DETECTION")
print("=" * 118)

print(
    f"\nUsable cycles          : "
    f"{len(df):,}"
)

print(
    f"Operations             : "
    f"{df['operation_code'].nunique()}"
)

print(
    f"Physical input features: "
    f"{len(FEATURES)}"
)


# =============================================================================
# 4. CHECK FEATURE COMPLETENESS
# =============================================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing required features:\n"
        + "\n".join(
            missing_features
        )
    )


# =============================================================================
# 5. OPERATION-AWARE MODEL FITTING
# =============================================================================

scored_groups = []

model_records = []

for operation_code, group in df.groupby(
    "operation_code"
):

    group = group.copy()

    healthy_mask = (
        group[
            "benchmark_health_region"
        ].isin(
            [
                "training_healthy",
                "test_healthy",
            ]
        )
    )

    healthy = group.loc[
        healthy_mask,
        FEATURES,
    ].copy()

    if len(healthy) != 70:

        print(
            f"WARNING: {operation_code} "
            f"has {len(healthy)} usable known-healthy cycles"
        )

    # -------------------------------------------------------------
    # ROBUST SCALING
    # -------------------------------------------------------------
    #
    # Fit only on known healthy cycles.
    # This prevents later degraded observations from influencing
    # the definition of normal behaviour.

    scaler = RobustScaler()

    X_train = scaler.fit_transform(
        healthy
    )

    X_all = scaler.transform(
        group[
            FEATURES
        ]
    )

    # -------------------------------------------------------------
    # ISOLATION FOREST
    # -------------------------------------------------------------

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=MODEL_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train
    )

    # sklearn score_samples:
    # larger = more normal.
    #
    # Multiply by -1 so larger values mean MORE anomalous.
    anomaly_score = (
        -model.score_samples(
            X_all
        )
    )

    train_anomaly_score = (
        -model.score_samples(
            X_train
        )
    )

    threshold = np.quantile(
        train_anomaly_score,
        HEALTHY_SCORE_QUANTILE,
    )

    group[
        "if_anomaly_score"
    ] = anomaly_score

    group[
        "if_anomaly_threshold"
    ] = threshold

    group[
        "if_anomaly_flag"
    ] = (
        group[
            "if_anomaly_score"
        ]
        >
        threshold
    )

    healthy_flag_rate = (
        group.loc[
            healthy_mask,
            "if_anomaly_flag",
        ].mean()
    )

    faulty_mask = (
        group[
            "benchmark_health_region"
        ] == "faulty_evident"
    )

    faulty_flag_rate = (
        group.loc[
            faulty_mask,
            "if_anomaly_flag",
        ].mean()
    )

    intermediate_mask = (
        group[
            "benchmark_health_region"
        ] == "intermediate_unlabelled"
    )

    intermediate_flag_rate = (
        group.loc[
            intermediate_mask,
            "if_anomaly_flag",
        ].mean()
    )

    model_records.append(
        {
            "operation_code":
                operation_code,

            "healthy_training_cycles":
                int(
                    healthy_mask.sum()
                ),

            "anomaly_threshold":
                threshold,

            "known_healthy_anomaly_rate":
                healthy_flag_rate,

            "intermediate_anomaly_rate":
                intermediate_flag_rate,

            "faulty_anomaly_rate":
                faulty_flag_rate,
        }
    )

    scored_groups.append(
        group
    )


scored = pd.concat(
    scored_groups,
    ignore_index=True,
)

model_summary = pd.DataFrame(
    model_records
)


# =============================================================================
# 6. SAVE CYCLE-LEVEL SCORES
# =============================================================================

cycle_columns = [
    "operation_code",
    "cycle_index",
    "operation_family",
    "commanded_speed_group",
    "benchmark_health_region",
    "sequence_position",
    "cycles_from_end",
    "if_anomaly_score",
    "if_anomaly_threshold",
    "if_anomaly_flag",
]

scored[
    cycle_columns
].to_csv(
    OUTPUT_CYCLES,
    index=False,
)


# =============================================================================
# 7. BENCHMARK-REGION SUMMARY
# =============================================================================

region_summary = (
    scored
    .groupby(
        "benchmark_health_region"
    )
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),

        median_anomaly_score=(
            "if_anomaly_score",
            "median",
        ),

        anomaly_rate=(
            "if_anomaly_flag",
            "mean",
        ),

        anomaly_score_q25=(
            "if_anomaly_score",
            lambda x: x.quantile(0.25),
        ),

        anomaly_score_q75=(
            "if_anomaly_score",
            lambda x: x.quantile(0.75),
        ),
    )
    .reset_index()
)

region_summary[
    "anomaly_rate_pct"
] = (
    100
    * region_summary[
        "anomaly_rate"
    ]
)

region_summary.to_csv(
    OUTPUT_REGIONS,
    index=False,
)


print(
    "\nANOMALY BEHAVIOUR BY BENCHMARK REGION"
)
print("-" * 118)

print(
    region_summary[
        [
            "benchmark_health_region",
            "cycles",
            "median_anomaly_score",
            "anomaly_rate_pct",
            "anomaly_score_q25",
            "anomaly_score_q75",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# =============================================================================
# 8. OPERATION-LEVEL SUMMARY
# =============================================================================

model_summary[
    "known_healthy_anomaly_rate_pct"
] = (
    100
    * model_summary[
        "known_healthy_anomaly_rate"
    ]
)

model_summary[
    "intermediate_anomaly_rate_pct"
] = (
    100
    * model_summary[
        "intermediate_anomaly_rate"
    ]
)

model_summary[
    "faulty_anomaly_rate_pct"
] = (
    100
    * model_summary[
        "faulty_anomaly_rate"
    ]
)

model_summary[
    "fault_minus_healthy_pct_points"
] = (
    model_summary[
        "faulty_anomaly_rate_pct"
    ]
    -
    model_summary[
        "known_healthy_anomaly_rate_pct"
    ]
)

model_summary.to_csv(
    OUTPUT_OPERATIONS,
    index=False,
)


print(
    "\nOPERATION-SPECIFIC ANOMALY DETECTION"
)
print("-" * 118)

print(
    model_summary[
        [
            "operation_code",
            "known_healthy_anomaly_rate_pct",
            "intermediate_anomaly_rate_pct",
            "faulty_anomaly_rate_pct",
            "fault_minus_healthy_pct_points",
        ]
    ]
    .sort_values(
        "faulty_anomaly_rate_pct",
        ascending=False,
    )
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 9. NORMALIZED ANOMALY SCORE
# =============================================================================

# Different operation-specific Isolation Forests naturally have different
# raw score scales.
#
# Normalize by each operation's healthy threshold:
#
# ratio > 1 means score is beyond the calibrated healthy anomaly boundary.

scored[
    "if_threshold_ratio"
] = (
    scored[
        "if_anomaly_score"
    ]
    /
    scored[
        "if_anomaly_threshold"
    ]
)


# =============================================================================
# 10. SEQUENCE TRAJECTORY
# =============================================================================

scored[
    "sequence_bin"
] = pd.cut(
    scored[
        "sequence_position"
    ],
    bins=np.linspace(
        0,
        1,
        N_SEQUENCE_BINS + 1,
    ),
    include_lowest=True,
    labels=False,
)

scored[
    "sequence_bin"
] = (
    scored[
        "sequence_bin"
    ]
    + 1
)


trajectory = (
    scored
    .groupby(
        [
            "operation_code",
            "sequence_bin",
        ],
        observed=True,
    )
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),

        median_sequence_position=(
            "sequence_position",
            "median",
        ),

        median_anomaly_score=(
            "if_anomaly_score",
            "median",
        ),

        median_threshold_ratio=(
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

trajectory[
    "anomaly_rate_pct"
] = (
    100
    * trajectory[
        "anomaly_rate"
    ]
)

trajectory.to_csv(
    OUTPUT_TRAJECTORY,
    index=False,
)


# =============================================================================
# 11. FIRST VS LAST SEQUENCE BIN
# =============================================================================

first_last = trajectory[
    trajectory[
        "sequence_bin"
    ].isin(
        [
            1,
            20,
        ]
    )
].pivot(
    index="operation_code",
    columns="sequence_bin",
    values=[
        "median_threshold_ratio",
        "anomaly_rate_pct",
    ],
)

first_last.columns = [
    f"{metric}_bin_{sequence_bin}"
    for metric, sequence_bin
    in first_last.columns
]

first_last = (
    first_last
    .reset_index()
)


print(
    "\nANOMALY TRAJECTORY — FIRST 5% VS LAST 5%"
)
print("-" * 118)

print(
    first_last
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 12. POOLED SEQUENCE EVOLUTION
# =============================================================================

pooled = (
    trajectory
    .groupby(
        "sequence_bin"
    )
    .agg(
        median_threshold_ratio=(
            "median_threshold_ratio",
            "median",
        ),

        median_anomaly_rate_pct=(
            "anomaly_rate_pct",
            "median",
        ),
    )
    .reset_index()
)


print(
    "\nPOOLED SEQUENCE-BIN ANOMALY EVOLUTION"
)
print("-" * 118)

print(
    pooled
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 13. DIAGNOSTIC GROUP COMPARISON
# =============================================================================

ROBUST_HEALTH_OPERATIONS = [
    "1050",
    "1051",
    "1052",
    "1053",
    "1101",
    "2500",
]

MODERATE_HEALTH_OPERATIONS = [
    "1100",
    "1102",
    "1103",
    "2000",
]

WEAK_HEALTH_OPERATIONS = [
    "2100",
    "2200",
    "2300",
    "2400",
    "3000",
    "3100",
]


def diagnostic_group(
    operation_code
):

    if operation_code in ROBUST_HEALTH_OPERATIONS:
        return "health_robust"

    if operation_code in MODERATE_HEALTH_OPERATIONS:
        return "health_moderate"

    if operation_code in WEAK_HEALTH_OPERATIONS:
        return "health_weak"

    return "unclassified"


model_summary[
    "health_diagnostic_group"
] = (
    model_summary[
        "operation_code"
    ]
    .apply(
        diagnostic_group
    )
)


group_comparison = (
    model_summary
    .groupby(
        "health_diagnostic_group"
    )
    .agg(
        operations=(
            "operation_code",
            "count",
        ),

        median_healthy_anomaly_rate_pct=(
            "known_healthy_anomaly_rate_pct",
            "median",
        ),

        median_intermediate_anomaly_rate_pct=(
            "intermediate_anomaly_rate_pct",
            "median",
        ),

        median_faulty_anomaly_rate_pct=(
            "faulty_anomaly_rate_pct",
            "median",
        ),

        median_fault_minus_healthy_pct_points=(
            "fault_minus_healthy_pct_points",
            "median",
        ),
    )
    .reset_index()
)


print(
    "\nPHASE 05 HEALTH-DIAGNOSTIC GROUP VS UNSUPERVISED ANOMALY DETECTION"
)
print("-" * 118)

print(
    group_comparison
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 14. OUTPUTS
# =============================================================================

print(
    f"\nCycle anomaly scores saved to:\n"
    f"{OUTPUT_CYCLES}"
)

print(
    f"\nRegion summary saved to:\n"
    f"{OUTPUT_REGIONS}"
)

print(
    f"\nOperation summary saved to:\n"
    f"{OUTPUT_OPERATIONS}"
)

print(
    f"\nAnomaly trajectory saved to:\n"
    f"{OUTPUT_TRAJECTORY}"
)

print("\n" + "=" * 118)
print("PHASE 06.1 OPERATION-AWARE ISOLATION FOREST COMPLETE")
print("=" * 118)