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
    / "me_ad_all_joint_degradation_features.csv"
)

OUTPUT_CYCLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_multijoint_health_indicator.csv"
)

OUTPUT_REGIONS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_indicator_regions.csv"
)

OUTPUT_BINS = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_health_indicator_trajectory.csv"
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

EPSILON = 1e-9


# =============================================================================
# 3. LOAD DATA
# =============================================================================

df = pd.read_csv(
    INPUT_FILE,
    dtype={"operation_code": str},
)

df = df[
    df["structural_quality"]
    == "usable"
].copy()

df = df.sort_values(
    [
        "operation_code",
        "cycle_index",
    ]
).reset_index(
    drop=True
)


print("=" * 112)
print("ME-AD PHASE 05.1 — MULTIJOINT HEALTH INDICATOR ENGINEERING")
print("=" * 112)

print(
    f"\nUsable cycles : {len(df):,}"
)

print(
    f"Operations    : "
    f"{df['operation_code'].nunique()}"
)


# =============================================================================
# 4. BUILD OPERATION-SPECIFIC HEALTHY REFERENCES
# =============================================================================

# Use independent healthy-validation cycles 20–69.
#
# Median = normal operating centre.
# MAD    = normal healthy variability.
#
# A minimum scale floor is added later because some robot signals are
# extraordinarily repeatable. Without it, tiny harmless differences could
# produce enormous standardized departures.

reference_records = []

for operation_code, group in df.groupby(
    "operation_code"
):

    healthy = group[
        group[
            "benchmark_health_region"
        ] == "test_healthy"
    ]

    record = {
        "operation_code":
            operation_code
    }

    for joint in JOINTS:

        for metric in EFFORT_METRICS:

            feature = (
                f"j{joint}_{metric}"
            )

            values = healthy[
                feature
            ].dropna()

            median = values.median()

            mad = np.median(
                np.abs(
                    values - median
                )
            )

            robust_sigma = (
                1.4826 * mad
            )

            record[
                f"{feature}_healthy_median"
            ] = median

            record[
                f"{feature}_healthy_sigma"
            ] = robust_sigma

    reference_records.append(
        record
    )


references = pd.DataFrame(
    reference_records
)

df = df.merge(
    references,
    on="operation_code",
    how="left",
    validate="many_to_one",
)


# =============================================================================
# 5. ROBUST NORMALIZED DEPARTURES
# =============================================================================

# Critical safeguard:
#
# sigma_floor = 2% of the healthy median magnitude.
#
# This prevents extremely stable signals from receiving absurdly large
# standardized scores for physically tiny changes.
#
# The floor is an engineering regularization parameter, not a claim that
# 2% represents a universal mechanical tolerance.

departure_columns = []

for joint in JOINTS:

    for metric in EFFORT_METRICS:

        feature = (
            f"j{joint}_{metric}"
        )

        median_col = (
            f"{feature}_healthy_median"
        )

        sigma_col = (
            f"{feature}_healthy_sigma"
        )

        floor = (
            0.02
            * df[
                median_col
            ].abs()
        )

        effective_sigma = np.maximum(
            df[sigma_col],
            floor,
        )

        effective_sigma = np.maximum(
            effective_sigma,
            EPSILON,
        )

        departure_col = (
            f"{feature}_departure"
        )

        df[
            departure_col
        ] = (
            np.abs(
                df[feature]
                -
                df[median_col]
            )
            /
            effective_sigma
        )

        departure_columns.append(
            departure_col
        )


# =============================================================================
# 6. JOINT-LEVEL HEALTH DEPARTURE
# =============================================================================

# Each joint receives one score:
# median departure across its four effort metrics.
#
# Median rather than mean prevents one extreme feature from dominating.

joint_score_columns = []

for joint in JOINTS:

    cols = [
        f"j{joint}_{metric}_departure"
        for metric in EFFORT_METRICS
    ]

    score_col = (
        f"j{joint}_effort_departure"
    )

    df[
        score_col
    ] = df[
        cols
    ].median(
        axis=1
    )

    joint_score_columns.append(
        score_col
    )


# =============================================================================
# 7. J3 TARGET SCORE
# =============================================================================

df[
    "j3_target_score"
] = df[
    "j3_effort_departure"
]


# =============================================================================
# 8. NON-TARGET SYSTEM RESPONSE
# =============================================================================

control_joint_columns = [
    "j1_effort_departure",
    "j2_effort_departure",
    "j4_effort_departure",
    "j5_effort_departure",
    "j6_effort_departure",
]

df[
    "control_system_score"
] = df[
    control_joint_columns
].median(
    axis=1
)


# =============================================================================
# 9. MULTIJOINT SYSTEM SCORE
# =============================================================================

# Median across all six joints.
#
# This represents broad redistribution of robot effort.

df[
    "multijoint_system_score"
] = df[
    joint_score_columns
].median(
    axis=1
)


# =============================================================================
# 10. J3 LOCALIZATION CONTRAST
# =============================================================================

# Positive:
# J3 is more abnormal than the typical non-target joint.
#
# Negative:
# the rest of the robot is changing as much as or more than J3.

df[
    "j3_localization_contrast"
] = (
    df[
        "j3_target_score"
    ]
    -
    df[
        "control_system_score"
    ]
)


# =============================================================================
# 11. COMPOSITE DEGRADATION INDEX
# =============================================================================

# We deliberately keep this simple and interpretable.
#
# 50% = J3 target condition
# 50% = multijoint system response
#
# This is NOT yet the final production health score.
# Phase 05 will validate and refine it.

df[
    "degradation_index"
] = (
    0.50
    * df[
        "j3_target_score"
    ]
    +
    0.50
    * df[
        "multijoint_system_score"
    ]
)


# =============================================================================
# 12. HEALTH INDEX
# =============================================================================

# Convert degradation into an intuitive bounded score:
#
# degradation = 0  -> health = 100
# degradation ↑    -> health approaches 0
#
# This transformation is monotonic and intended for interpretability.

df[
    "health_index"
] = (
    100
    /
    (
        1
        +
        df[
            "degradation_index"
        ]
    )
)


# =============================================================================
# 13. SAVE CYCLE-LEVEL HEALTH TABLE
# =============================================================================

keep_columns = [
    "operation_code",
    "cycle_index",
    "operation_family",
    "commanded_speed_group",
    "benchmark_health_region",
    "sequence_position",
    "cycles_from_end",
    "j1_effort_departure",
    "j2_effort_departure",
    "j3_effort_departure",
    "j4_effort_departure",
    "j5_effort_departure",
    "j6_effort_departure",
    "j3_target_score",
    "control_system_score",
    "multijoint_system_score",
    "j3_localization_contrast",
    "degradation_index",
    "health_index",
]

health = df[
    keep_columns
].copy()

health.to_csv(
    OUTPUT_CYCLES,
    index=False,
)


# =============================================================================
# 14. BENCHMARK-REGION SUMMARY
# =============================================================================

region_summary = (
    health
    .groupby(
        "benchmark_health_region"
    )
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),

        median_j3_score=(
            "j3_target_score",
            "median",
        ),

        median_control_score=(
            "control_system_score",
            "median",
        ),

        median_multijoint_score=(
            "multijoint_system_score",
            "median",
        ),

        median_localization_contrast=(
            "j3_localization_contrast",
            "median",
        ),

        median_degradation_index=(
            "degradation_index",
            "median",
        ),

        median_health_index=(
            "health_index",
            "median",
        ),

        health_q25=(
            "health_index",
            lambda x: x.quantile(0.25),
        ),

        health_q75=(
            "health_index",
            lambda x: x.quantile(0.75),
        ),
    )
    .reset_index()
)

region_summary.to_csv(
    OUTPUT_REGIONS,
    index=False,
)


print(
    "\nHEALTH INDICATOR BY BENCHMARK REGION"
)
print("-" * 112)

print(
    region_summary
    .round(3)
    .to_string(
        index=False
    )
)


# =============================================================================
# 15. OPERATION-LEVEL HEALTH SUMMARY
# =============================================================================

operation_region = (
    health
    .groupby(
        [
            "operation_code",
            "benchmark_health_region",
        ]
    )
    .agg(
        cycles=(
            "cycle_index",
            "count",
        ),

        median_j3_score=(
            "j3_target_score",
            "median",
        ),

        median_system_score=(
            "multijoint_system_score",
            "median",
        ),

        median_degradation_index=(
            "degradation_index",
            "median",
        ),

        median_health_index=(
            "health_index",
            "median",
        ),
    )
    .reset_index()
)


print(
    "\nOPERATION-SPECIFIC HEALTH — TEST HEALTHY VS EVIDENT FAULT"
)
print("-" * 112)

comparison = operation_region[
    operation_region[
        "benchmark_health_region"
    ].isin(
        [
            "test_healthy",
            "faulty_evident",
        ]
    )
].copy()

pivot = comparison.pivot(
    index="operation_code",
    columns="benchmark_health_region",
    values="median_health_index",
).reset_index()

pivot.columns.name = None

pivot[
    "health_drop_points"
] = (
    pivot[
        "test_healthy"
    ]
    -
    pivot[
        "faulty_evident"
    ]
)

print(
    pivot
    .sort_values(
        "health_drop_points",
        ascending=False,
    )
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 16. NORMALIZED SEQUENCE TRAJECTORY
# =============================================================================

health[
    "sequence_bin"
] = pd.cut(
    health[
        "sequence_position"
    ],
    bins=np.linspace(
        0,
        1,
        21,
    ),
    include_lowest=True,
    labels=False,
)

health[
    "sequence_bin"
] = (
    health[
        "sequence_bin"
    ]
    + 1
)


trajectory = (
    health
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

        median_j3_score=(
            "j3_target_score",
            "median",
        ),

        median_system_score=(
            "multijoint_system_score",
            "median",
        ),

        median_degradation_index=(
            "degradation_index",
            "median",
        ),

        median_health_index=(
            "health_index",
            "median",
        ),
    )
    .reset_index()
)

trajectory.to_csv(
    OUTPUT_BINS,
    index=False,
)


# =============================================================================
# 17. FIRST VS LAST SEQUENCE BIN
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
    values="median_health_index",
).reset_index()

first_last.columns.name = None

first_last = first_last.rename(
    columns={
        1: "health_first_5pct",
        20: "health_last_5pct",
    }
)

first_last[
    "health_change_points"
] = (
    first_last[
        "health_last_5pct"
    ]
    -
    first_last[
        "health_first_5pct"
    ]
)


print(
    "\nHEALTH TRAJECTORY — FIRST 5% VS LAST 5%"
)
print("-" * 112)

print(
    first_last
    .sort_values(
        "health_change_points"
    )
    .round(2)
    .to_string(
        index=False
    )
)


# =============================================================================
# 18. HEALTH SEPARATION CHECK
# =============================================================================

healthy_values = health[
    health[
        "benchmark_health_region"
    ] == "test_healthy"
][
    "health_index"
]

faulty_values = health[
    health[
        "benchmark_health_region"
    ] == "faulty_evident"
][
    "health_index"
]


print(
    "\nHEALTH-SEPARATION CHECK"
)
print("-" * 112)

print(
    f"Test-healthy median health : "
    f"{healthy_values.median():.2f}"
)

print(
    f"Test-healthy Q25-Q75       : "
    f"{healthy_values.quantile(0.25):.2f} - "
    f"{healthy_values.quantile(0.75):.2f}"
)

print(
    f"Evident-fault median health: "
    f"{faulty_values.median():.2f}"
)

print(
    f"Evident-fault Q25-Q75      : "
    f"{faulty_values.quantile(0.25):.2f} - "
    f"{faulty_values.quantile(0.75):.2f}"
)


# =============================================================================
# 19. OUTPUTS
# =============================================================================

print(
    f"\nCycle-level health indicator saved to:\n"
    f"{OUTPUT_CYCLES}"
)

print(
    f"\nBenchmark-region summary saved to:\n"
    f"{OUTPUT_REGIONS}"
)

print(
    f"\nHealth trajectory saved to:\n"
    f"{OUTPUT_BINS}"
)

print("\n" + "=" * 112)
print("PHASE 05.1 MULTIJOINT HEALTH INDICATOR COMPLETE")
print("=" * 112)