from pathlib import Path
from zipfile import ZipFile
import io
import sys
import types

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hp\Industrial-Analytics"
    r"\Mitsubishi-ME-AD-Predictive-Maintenance"
)

ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "me_ad"
    / "ME-AD.zip"
)

QUALITY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_stratified_cycle_quality.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_physical_signal_summary.csv"
)

OUTPUT_CYCLES = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_cycle_signal_metrics.csv"
)


# =============================================================================
# 2. PICKLE COMPATIBILITY
# =============================================================================

numeric_module = types.ModuleType(
    "pandas.core.indexes.numeric"
)

numeric_module.Int64Index = pd.Index
numeric_module.UInt64Index = pd.Index
numeric_module.Float64Index = pd.Index

sys.modules[
    "pandas.core.indexes.numeric"
] = numeric_module


# =============================================================================
# 3. LOAD THE SAME 400-CYCLE SAMPLE
# =============================================================================

sample = pd.read_csv(
    QUALITY_FILE,
    dtype={"operation_code": str},
)

sample = sample[
    sample["load_status"] == "OK"
].copy()

print("=" * 92)
print("ME-AD PHASE 02.2 — PHYSICAL SIGNAL SANITY")
print("=" * 92)

print(f"\nCycles to inspect : {len(sample):,}")
print(
    f"Operations        : "
    f"{sample['operation_code'].nunique()}"
)


# =============================================================================
# 4. SIGNAL METRICS
# =============================================================================

records = []

with ZipFile(ZIP_FILE, "r") as archive:

    total = len(sample)

    for counter, row in enumerate(
        sample.itertuples(index=False),
        start=1,
    ):

        member = (
            f"ME-AD/Pandas/{row.operation_code}/"
            f"cleaned_dataset_{int(row.cycle_index)}.pkl"
        )

        raw_bytes = archive.read(member)

        df = pd.read_pickle(
            io.BytesIO(raw_bytes)
        )

        for joint in range(1, 7):

            q = df[f"q_{joint}"].to_numpy()
            dq = df[f"dq_{joint}"].to_numpy()
            ddq = df[f"ddq_{joint}"].to_numpy()
            tau = df[f"tau_{joint}"].to_numpy()

            qf = df[f"q_filt_{joint}"].to_numpy()
            dqf = df[f"dq_filt_{joint}"].to_numpy()
            ddqf = df[f"ddq_filt_{joint}"].to_numpy()
            tauf = df[f"tau_filt_{joint}"].to_numpy()

            # -------------------------------------------------------------
            # Raw-vs-filtered differences
            # -------------------------------------------------------------

            q_diff = q - qf
            dq_diff = dq - dqf
            ddq_diff = ddq - ddqf
            tau_diff = tau - tauf

            # -------------------------------------------------------------
            # Correlations
            # -------------------------------------------------------------

            def safe_corr(a, b):

                if (
                    np.std(a) == 0
                    or np.std(b) == 0
                ):
                    return np.nan

                return np.corrcoef(a, b)[0, 1]

            records.append(
                {
                    "operation_code":
                        row.operation_code,

                    "cycle_index":
                        int(row.cycle_index),

                    "benchmark_health_region":
                        row.benchmark_health_region,

                    "sequence_position":
                        row.sequence_position,

                    "joint":
                        joint,

                    "rows":
                        len(df),

                    # Position
                    "q_min":
                        np.min(q),

                    "q_max":
                        np.max(q),

                    "q_range":
                        np.ptp(q),

                    # Velocity
                    "dq_min":
                        np.min(dq),

                    "dq_max":
                        np.max(dq),

                    "dq_rms":
                        np.sqrt(np.mean(dq ** 2)),

                    # Acceleration
                    "ddq_min":
                        np.min(ddq),

                    "ddq_max":
                        np.max(ddq),

                    "ddq_rms":
                        np.sqrt(np.mean(ddq ** 2)),

                    # Torque
                    "tau_min":
                        np.min(tau),

                    "tau_max":
                        np.max(tau),

                    "tau_mean_abs":
                        np.mean(np.abs(tau)),

                    "tau_rms":
                        np.sqrt(np.mean(tau ** 2)),

                    # Filter comparison
                    "q_raw_filt_rmse":
                        np.sqrt(
                            np.mean(q_diff ** 2)
                        ),

                    "dq_raw_filt_rmse":
                        np.sqrt(
                            np.mean(dq_diff ** 2)
                        ),

                    "ddq_raw_filt_rmse":
                        np.sqrt(
                            np.mean(ddq_diff ** 2)
                        ),

                    "tau_raw_filt_rmse":
                        np.sqrt(
                            np.mean(tau_diff ** 2)
                        ),

                    "q_raw_filt_corr":
                        safe_corr(q, qf),

                    "dq_raw_filt_corr":
                        safe_corr(dq, dqf),

                    "ddq_raw_filt_corr":
                        safe_corr(ddq, ddqf),

                    "tau_raw_filt_corr":
                        safe_corr(tau, tauf),
                }
            )

        if (
            counter % 50 == 0
            or counter == total
        ):
            print(
                f"Processed "
                f"{counter:,}/{total:,} cycles"
            )


metrics = pd.DataFrame(records)

metrics.to_csv(
    OUTPUT_CYCLES,
    index=False,
)


# =============================================================================
# 5. JOINT-LEVEL PHYSICAL SUMMARY
# =============================================================================

summary = (
    metrics
    .groupby("joint")
    .agg(
        cycles=("cycle_index", "count"),

        q_min=("q_min", "min"),
        q_max=("q_max", "max"),

        dq_min=("dq_min", "min"),
        dq_max=("dq_max", "max"),
        median_dq_rms=("dq_rms", "median"),

        ddq_min=("ddq_min", "min"),
        ddq_max=("ddq_max", "max"),
        median_ddq_rms=("ddq_rms", "median"),

        tau_min=("tau_min", "min"),
        tau_max=("tau_max", "max"),
        median_tau_abs=("tau_mean_abs", "median"),
        median_tau_rms=("tau_rms", "median"),

        median_q_filter_rmse=(
            "q_raw_filt_rmse",
            "median",
        ),

        median_dq_filter_rmse=(
            "dq_raw_filt_rmse",
            "median",
        ),

        median_ddq_filter_rmse=(
            "ddq_raw_filt_rmse",
            "median",
        ),

        median_tau_filter_rmse=(
            "tau_raw_filt_rmse",
            "median",
        ),

        median_q_filter_corr=(
            "q_raw_filt_corr",
            "median",
        ),

        median_dq_filter_corr=(
            "dq_raw_filt_corr",
            "median",
        ),

        median_ddq_filter_corr=(
            "ddq_raw_filt_corr",
            "median",
        ),

        median_tau_filter_corr=(
            "tau_raw_filt_corr",
            "median",
        ),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


print("\nJOINT-LEVEL PHYSICAL SUMMARY")
print("-" * 92)

print(
    summary
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 6. TORQUE BY HEALTH REGION
# =============================================================================

torque_health = (
    metrics
    .groupby(
        [
            "joint",
            "benchmark_health_region",
        ]
    )
    .agg(
        cycles=("cycle_index", "count"),
        median_tau_abs=(
            "tau_mean_abs",
            "median",
        ),
        median_tau_rms=(
            "tau_rms",
            "median",
        ),
    )
    .reset_index()
)

print("\nTORQUE BY BENCHMARK HEALTH REGION")
print("-" * 92)

print(
    torque_health
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 7. J3 VERSUS OTHER JOINTS
# =============================================================================

joint_torque = (
    metrics
    .groupby("joint")
    .agg(
        median_tau_abs=(
            "tau_mean_abs",
            "median",
        ),
        median_tau_rms=(
            "tau_rms",
            "median",
        ),
    )
    .reset_index()
)

print("\nTORQUE EFFORT BY JOINT")
print("-" * 92)

print(
    joint_torque
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 8. MOST EXTREME TORQUE OBSERVATIONS
# =============================================================================

extreme_torque = (
    metrics
    .assign(
        absolute_tau_extreme=lambda x:
        np.maximum(
            np.abs(x["tau_min"]),
            np.abs(x["tau_max"]),
        )
    )
    .sort_values(
        "absolute_tau_extreme",
        ascending=False,
    )
    .head(20)
)

print("\nTOP 20 EXTREME TORQUE CYCLE/JOINT COMBINATIONS")
print("-" * 92)

print(
    extreme_torque[
        [
            "operation_code",
            "cycle_index",
            "benchmark_health_region",
            "sequence_position",
            "joint",
            "tau_min",
            "tau_max",
            "tau_mean_abs",
            "tau_rms",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# =============================================================================
# 9. RAW/FILTERED AGREEMENT
# =============================================================================

filter_summary = (
    metrics
    .groupby("joint")
    .agg(
        q_corr=(
            "q_raw_filt_corr",
            "median",
        ),
        dq_corr=(
            "dq_raw_filt_corr",
            "median",
        ),
        ddq_corr=(
            "ddq_raw_filt_corr",
            "median",
        ),
        tau_corr=(
            "tau_raw_filt_corr",
            "median",
        ),
    )
    .reset_index()
)

print("\nMEDIAN RAW/FILTERED CORRELATION BY JOINT")
print("-" * 92)

print(
    filter_summary
    .round(5)
    .to_string(index=False)
)


print(
    f"\nCycle-level metrics saved to:\n"
    f"{OUTPUT_CYCLES}"
)

print(
    f"\nJoint summary saved to:\n"
    f"{OUTPUT_SUMMARY}"
)

print("\n" + "=" * 92)
print("PHASE 02.2 PHYSICAL SIGNAL SANITY COMPLETE")
print("=" * 92)