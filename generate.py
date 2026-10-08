import time
from pathlib import Path

import pandas as pd

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

REAL_DATA_PATH = Path("data/real/employee.csv")
OUTPUT_PATH = Path("data/synthetic/hr_ctgan_synthetic.csv")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load real HR data
# ---------------------------------------------------------

print("Loading HR dataset...")

real_df = pd.read_csv(REAL_DATA_PATH)

print(f"Real data shape: {real_df.shape}")
print(f"Rows: {len(real_df):,}")
print(f"Columns: {len(real_df.columns)}")

print("\nColumns:")
print(list(real_df.columns))


# ---------------------------------------------------------
# Basic data information
# ---------------------------------------------------------

print("\nMissing values:")
print(real_df.isnull().sum().sum())

print("\nData types:")
print(real_df.dtypes.value_counts())


# ---------------------------------------------------------
# Convert dataframe to SynthCity loader
# ---------------------------------------------------------

loader = GenericDataLoader(
    real_df,
    target_column="Attrition"
)


# ---------------------------------------------------------
# Load CTGAN plugin
# ---------------------------------------------------------

print("\nLoading CTGAN plugin...")

plugin = Plugins().get(
    "ctgan",
    n_iter=300
)


# ---------------------------------------------------------
# Train CTGAN
# ---------------------------------------------------------

print("\nTraining CTGAN...")

start_time = time.perf_counter()

plugin.fit(loader)

training_time = time.perf_counter() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")


# ---------------------------------------------------------
# Generate synthetic HR data
# ---------------------------------------------------------

NUM_SYNTHETIC_ROWS = len(real_df)

print(f"\nGenerating {NUM_SYNTHETIC_ROWS:,} synthetic rows...")

generation_start = time.perf_counter()

synthetic_data = plugin.generate(
    count=NUM_SYNTHETIC_ROWS
)

generation_time = time.perf_counter() - generation_start


# ---------------------------------------------------------
# Convert generated data to dataframe
# ---------------------------------------------------------

synthetic_df = synthetic_data.dataframe()

print(f"\nGeneration completed in {generation_time:.2f} seconds")

print(f"Synthetic data shape: {synthetic_df.shape}")


# ---------------------------------------------------------
# Save synthetic dataset
# ---------------------------------------------------------

synthetic_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nSynthetic dataset saved to:")
print(OUTPUT_PATH)


# ---------------------------------------------------------
# Final comparison
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("REAL VS SYNTHETIC")
print("=" * 60)

print(f"Real rows:              {len(real_df):,}")
print(f"Synthetic rows:         {len(synthetic_df):,}")

print(f"Real columns:           {len(real_df.columns)}")
print(f"Synthetic columns:      {len(synthetic_df.columns)}")

print(f"Training time:           {training_time:.2f} seconds")
print(f"Generation time:         {generation_time:.2f} seconds")

print(
    f"Real missing values:     "
    f"{real_df.isnull().sum().sum():,}"
)

print(
    f"Synthetic missing values: "
    f"{synthetic_df.isnull().sum().sum():,}"
)

print("\nSynthetic data preview:")
print(synthetic_df.head())