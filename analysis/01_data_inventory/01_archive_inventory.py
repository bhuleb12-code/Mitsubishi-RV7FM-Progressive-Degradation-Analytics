from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
import csv


# =============================================================================
# 1. PROJECT PATHS
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "me_ad_archive_inventory.csv"


# =============================================================================
# 2. VALIDATE ARCHIVE
# =============================================================================

if not ARCHIVE_PATH.exists():
    raise FileNotFoundError(
        f"ME-AD archive not found:\n{ARCHIVE_PATH}"
    )

print("=" * 70)
print("ME-AD DATASET — ARCHIVE INVENTORY")
print("=" * 70)
print(f"Archive: {ARCHIVE_PATH}")
print()


# =============================================================================
# 3. INVENTORY CONTAINERS
# =============================================================================

extension_counts = Counter()

operation_stats = defaultdict(
    lambda: {
        "csv_files": 0,
        "csv_bytes": 0,
        "pkl_files": 0,
        "pkl_bytes": 0,
    }
)

total_entries = 0
total_files = 0
total_uncompressed_bytes = 0
total_compressed_bytes = 0


# =============================================================================
# 4. SCAN ZIP WITHOUT EXTRACTING
# =============================================================================

with ZipFile(ARCHIVE_PATH, "r") as archive:

    for entry in archive.infolist():

        total_entries += 1

        # Ignore directory entries
        if entry.is_dir():
            continue

        total_files += 1
        total_uncompressed_bytes += entry.file_size
        total_compressed_bytes += entry.compress_size

        path = Path(entry.filename)
        extension = path.suffix.lower()

        extension_counts[extension] += 1

        parts = entry.filename.split("/")

        # ---------------------------------------------------------
        # CSV structure:
        # ME-AD/CSV/1050_CSV/example.csv
        # ---------------------------------------------------------
        if (
            len(parts) >= 4
            and parts[0] == "ME-AD"
            and parts[1] == "CSV"
            and extension == ".csv"
        ):
            operation_code = parts[2].replace("_CSV", "")

            operation_stats[operation_code]["csv_files"] += 1
            operation_stats[operation_code]["csv_bytes"] += entry.file_size

        # ---------------------------------------------------------
        # Pandas structure:
        # ME-AD/Pandas/1050/example.pkl
        # ---------------------------------------------------------
        elif (
            len(parts) >= 4
            and parts[0] == "ME-AD"
            and parts[1] == "Pandas"
            and extension == ".pkl"
        ):
            operation_code = parts[2]

            operation_stats[operation_code]["pkl_files"] += 1
            operation_stats[operation_code]["pkl_bytes"] += entry.file_size


# =============================================================================
# 5. DATASET SUMMARY
# =============================================================================

print(f"Total ZIP entries       : {total_entries:,}")
print(f"Actual files            : {total_files:,}")
print(
    f"Uncompressed size       : "
    f"{total_uncompressed_bytes / (1024**3):,.2f} GiB"
)
print(
    f"Compressed file content : "
    f"{total_compressed_bytes / (1024**3):,.2f} GiB"
)

print()
print("FILE TYPES")
print("-" * 70)

for extension, count in extension_counts.most_common():
    label = extension if extension else "[no extension]"
    print(f"{label:<15} {count:>10,}")


# =============================================================================
# 6. OPERATION SUMMARY
# =============================================================================

print()
print("OPERATION INVENTORY")
print("-" * 70)

header = (
    f"{'Operation':<12}"
    f"{'CSV files':>12}"
    f"{'PKL files':>12}"
    f"{'CSV GiB':>12}"
    f"{'PKL GiB':>12}"
)

print(header)
print("-" * 60)

for operation_code in sorted(operation_stats):

    stats = operation_stats[operation_code]

    print(
        f"{operation_code:<12}"
        f"{stats['csv_files']:>12,}"
        f"{stats['pkl_files']:>12,}"
        f"{stats['csv_bytes'] / (1024**3):>12.3f}"
        f"{stats['pkl_bytes'] / (1024**3):>12.3f}"
    )


# =============================================================================
# 7. SAVE MACHINE-READABLE INVENTORY
# =============================================================================

with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "operation_code",
            "csv_file_count",
            "pkl_file_count",
            "csv_uncompressed_bytes",
            "pkl_uncompressed_bytes",
            "csv_size_gib",
            "pkl_size_gib",
        ]
    )

    for operation_code in sorted(operation_stats):

        stats = operation_stats[operation_code]

        writer.writerow(
            [
                operation_code,
                stats["csv_files"],
                stats["pkl_files"],
                stats["csv_bytes"],
                stats["pkl_bytes"],
                round(
                    stats["csv_bytes"] / (1024**3),
                    6,
                ),
                round(
                    stats["pkl_bytes"] / (1024**3),
                    6,
                ),
            ]
        )


# =============================================================================
# 8. VALIDATION CHECKS
# =============================================================================

csv_total = sum(
    stats["csv_files"]
    for stats in operation_stats.values()
)

pkl_total = sum(
    stats["pkl_files"]
    for stats in operation_stats.values()
)

print()
print("VALIDATION")
print("-" * 70)
print(f"CSV files identified : {csv_total:,}")
print(f"PKL files identified : {pkl_total:,}")
print(f"Operation codes       : {len(operation_stats):,}")

if csv_total == pkl_total:
    print("CSV/PKL count match   : YES")
else:
    print("CSV/PKL count match   : NO")

print()
print(f"Inventory saved to:\n{OUTPUT_FILE}")
print()
print("=" * 70)
print("ARCHIVE INVENTORY COMPLETE")
print("=" * 70)