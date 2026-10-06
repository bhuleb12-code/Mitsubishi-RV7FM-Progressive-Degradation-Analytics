from pathlib import Path
from zipfile import ZipFile
import re

import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hp\Industrial-Analytics"
    r"\Mitsubishi-ME-AD-Predictive-Maintenance"
)

ARCHIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "me_ad"
    / "ME-AD.zip"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "me_ad_cycle_manifest.csv"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. OPERATION METADATA
# =============================================================================

OPERATION_METADATA = {
    "1050": ("family_1", "50_percent"),
    "1051": ("family_1", "50_percent"),
    "1052": ("family_1", "50_percent"),
    "1053": ("family_1", "50_percent"),

    "1100": ("family_1", "100_percent"),
    "1101": ("family_1", "100_percent"),
    "1102": ("family_1", "100_percent"),
    "1103": ("family_1", "100_percent"),

    "2000": ("family_2", "not_applicable"),
    "2100": ("family_2", "not_applicable"),
    "2200": ("family_2", "not_applicable"),
    "2300": ("family_2", "not_applicable"),
    "2400": ("family_2", "not_applicable"),
    "2500": ("family_2", "not_applicable"),

    "3000": ("family_3", "not_applicable"),
    "3100": ("family_3", "not_applicable"),
}


# =============================================================================
# 3. READ PKL INVENTORY
# =============================================================================

records = []

pattern = re.compile(
    r"^ME-AD/Pandas/(\d{4})/cleaned_dataset_(\d+)\.pkl$"
)

with ZipFile(ARCHIVE_PATH, "r") as archive:

    for entry in archive.infolist():

        if entry.is_dir():
            continue

        match = pattern.match(entry.filename)

        if match is None:
            continue

        operation_code = match.group(1)
        cycle_index = int(match.group(2))

        records.append(
            {
                "operation_code": operation_code,
                "cycle_index": cycle_index,
                "pkl_member": entry.filename,
                "pkl_bytes": entry.file_size,
            }
        )


manifest = pd.DataFrame(records)

if manifest.empty:
    raise RuntimeError("No ME-AD processed PKL datasets were found.")


# =============================================================================
# 4. DETERMINE OPERATION SEQUENCE LENGTH
# =============================================================================

operation_counts = (
    manifest
    .groupby("operation_code")["cycle_index"]
    .count()
    .to_dict()
)

operation_max_index = (
    manifest
    .groupby("operation_code")["cycle_index"]
    .max()
    .to_dict()
)


manifest["operation_cycle_count"] = (
    manifest["operation_code"].map(operation_counts)
)

manifest["operation_max_index"] = (
    manifest["operation_code"].map(operation_max_index)
)


# =============================================================================
# 5. ADD PHYSICAL OPERATION METADATA
# =============================================================================

manifest["operation_family"] = manifest["operation_code"].map(
    lambda x: OPERATION_METADATA[x][0]
)

manifest["commanded_speed_group"] = manifest["operation_code"].map(
    lambda x: OPERATION_METADATA[x][1]
)


# =============================================================================
# 6. ASSIGN BENCHMARK HEALTH REGION
# =============================================================================

def assign_health_region(row):

    cycle = row["cycle_index"]
    count = row["operation_cycle_count"]

    if 0 <= cycle < 20:
        return "training_healthy"

    if 20 <= cycle < 70:
        return "test_healthy"

    if cycle >= count - 50:
        return "faulty_evident"

    return "intermediate_unlabelled"


manifest["benchmark_health_region"] = manifest.apply(
    assign_health_region,
    axis=1,
)


# =============================================================================
# 7. NORMALIZED DEGRADATION POSITION
# =============================================================================

manifest["sequence_position"] = (
    manifest["cycle_index"]
    / manifest["operation_max_index"]
)


# =============================================================================
# 8. DISTANCE FROM END OF EXPERIMENT
# =============================================================================

manifest["cycles_from_end"] = (
    manifest["operation_max_index"]
    - manifest["cycle_index"]
)


# =============================================================================
# 9. SORT
# =============================================================================

manifest = manifest.sort_values(
    ["operation_code", "cycle_index"]
).reset_index(drop=True)


# =============================================================================
# 10. VALIDATION
# =============================================================================

print("=" * 78)
print("ME-AD DATASET — MASTER CYCLE MANIFEST")
print("=" * 78)

print(f"\nManifest rows : {len(manifest):,}")
print(
    f"Operations    : "
    f"{manifest['operation_code'].nunique():,}"
)

print("\nHEALTH REGION COUNTS")
print("-" * 78)

print(
    manifest["benchmark_health_region"]
    .value_counts()
    .to_string()
)

print("\nOPERATION SUMMARY")
print("-" * 78)

summary = (
    manifest
    .groupby("operation_code")
    .agg(
        cycles=("cycle_index", "count"),
        min_index=("cycle_index", "min"),
        max_index=("cycle_index", "max"),
        training_healthy=(
            "benchmark_health_region",
            lambda x: (x == "training_healthy").sum(),
        ),
        test_healthy=(
            "benchmark_health_region",
            lambda x: (x == "test_healthy").sum(),
        ),
        intermediate=(
            "benchmark_health_region",
            lambda x: (x == "intermediate_unlabelled").sum(),
        ),
        faulty_evident=(
            "benchmark_health_region",
            lambda x: (x == "faulty_evident").sum(),
        ),
    )
)

print(summary.to_string())


# =============================================================================
# 11. SEQUENCE CONTINUITY CHECK
# =============================================================================

print("\nSEQUENCE CONTINUITY")
print("-" * 78)

for operation_code, group in manifest.groupby("operation_code"):

    indices = sorted(group["cycle_index"].tolist())

    expected = list(
        range(
            min(indices),
            max(indices) + 1,
        )
    )

    missing = sorted(set(expected) - set(indices))

    if not missing:
        print(
            f"{operation_code}: CONTINUOUS "
            f"({indices[0]}–{indices[-1]})"
        )
    else:
        print(
            f"{operation_code}: "
            f"{len(missing):,} missing indices"
        )


# =============================================================================
# 12. SAVE
# =============================================================================

manifest.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(f"\nManifest saved to:\n{OUTPUT_FILE}")

print("\nFIRST 10 ROWS")
print("-" * 78)
print(manifest.head(10).to_string(index=False))

print("\nLAST 10 ROWS")
print("-" * 78)
print(manifest.tail(10).to_string(index=False))

print("\n" + "=" * 78)
print("MASTER CYCLE MANIFEST COMPLETE")
print("=" * 78)