import pandas as pd
from synthcity.plugins import Plugins


# Load the real dataset
real_data = pd.read_csv("data/real/customers_training.csv")

print("REAL DATA")
print(real_data.head())
print()
print("Shape:", real_data.shape)


# Create a SynthCity CTGAN plugin
plugin = Plugins().get(
    "ctgan",
    n_iter=100,
)


# Train the generator on the real data
print("\nTraining CTGAN...")
plugin.fit(real_data)

print("Training completed.")


# Generate synthetic data
print("\nGenerating synthetic data...")

synthetic_data = plugin.generate(
    count=1000
).dataframe()


# Display generated data
print("\nSYNTHETIC DATA")
print(synthetic_data.head())

print()
print("Synthetic shape:", synthetic_data.shape)


# Save synthetic data
synthetic_data.to_csv(
    "data/synthetic/synthetic_customers.csv",
    index=False,
)

print("\nSynthetic data saved successfully.")