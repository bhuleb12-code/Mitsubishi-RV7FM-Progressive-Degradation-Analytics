from pathlib import Path
from zipfile import ZipFile
import io
import pickle
import sys
import types

import pandas as pd


# =============================================================================
# PANDAS PICKLE BACKWARD COMPATIBILITY
# =============================================================================
# ME-AD pickle files were created with an older Pandas version that referenced
# pandas.core.indexes.numeric. Newer Pandas versions no longer expose that
# module. The aliases below allow the historical pickle to be deserialized
# without downgrading the project's Pandas installation.

from pandas import Index

numeric_module = types.ModuleType("pandas.core.indexes.numeric")
numeric_module.Int64Index = Index
numeric_module.UInt64Index = Index
numeric_module.Float64Index = Index

sys.modules["pandas.core.indexes.numeric"] = numeric_module


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

OPERATION = "1050"
PKL_MEMBER = f"ME-AD/Pandas/{OPERATION}/cleaned_dataset_0.pkl"


# =============================================================================
# 2. LOAD ONE PROCESSED CYCLE DIRECTLY FROM ZIP
# =============================================================================

print("=" * 72)
print("ME-AD DATASET — SAMPLE CYCLE INSPECTION")
print("=" * 72)

with ZipFile(ARCHIVE_PATH, "r") as archive:

    with archive.open(PKL_MEMBER) as file:
        data = pickle.load(file)


# =============================================================================
# 3. BASIC STRUCTURE
# =============================================================================

print(f"\nOperation code : {OPERATION}")
print(f"Archive member : {PKL_MEMBER}")
print(f"Python object  : {type(data)}")

if not isinstance(data, pd.DataFrame):
    raise TypeError(
        f"Expected pandas DataFrame, received {type(data)}"
    )

print(f"Rows           : {len(data):,}")
print(f"Columns        : {len(data.columns):,}")


# =============================================================================
# 4. COLUMN INVENTORY
# =============================================================================

print("\nCOLUMNS")
print("-" * 72)

for i, column in enumerate(data.columns, start=1):
    print(f"{i:>3}. {column}")


# =============================================================================
# 5. DATA TYPES
# =============================================================================

print("\nDATA TYPES")
print("-" * 72)
print(data.dtypes.to_string())


# =============================================================================
# 6. FIRST OBSERVATIONS
# =============================================================================

print("\nFIRST 5 ROWS")
print("-" * 72)
print(data.head().to_string())


# =============================================================================
# 7. QUALITY CHECKS
# =============================================================================

missing = data.isna().sum()
missing = missing[missing > 0]

print("\nQUALITY CHECK")
print("-" * 72)
print(f"Total missing values : {data.isna().sum().sum():,}")
print(f"Duplicate rows       : {data.duplicated().sum():,}")

if len(missing) > 0:
    print("\nColumns containing missing values:")
    print(missing.to_string())
else:
    print("Columns with missing values: NONE")


# =============================================================================
# 8. APPROXIMATE CYCLE DURATION
# =============================================================================

SAMPLING_INTERVAL_SECONDS = 0.003555
SAMPLING_RATE_HZ = 1 / SAMPLING_INTERVAL_SECONDS

duration_seconds = len(data) * SAMPLING_INTERVAL_SECONDS

print("\nSAMPLING")
print("-" * 72)
print(f"Sampling interval     : {SAMPLING_INTERVAL_SECONDS:.6f} s")
print(f"Approx. sampling rate : {SAMPLING_RATE_HZ:.2f} Hz")
print(f"Approx. cycle duration: {duration_seconds:.3f} s")


# =============================================================================
# 9. NUMERIC SUMMARY
# =============================================================================

print("\nNUMERIC SUMMARY")
print("-" * 72)
print(data.describe().T.to_string())


print("\n" + "=" * 72)
print("SAMPLE INSPECTION COMPLETE")
print("=" * 72)