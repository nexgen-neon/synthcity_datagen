# MOSTLY AI Synthetic Data SDK

> Generate realistic synthetic tabular data from real datasets using the open-source MOSTLY AI Synthetic Data SDK.

---

# 1. Project Overview

This project demonstrates how to use the **MOSTLY AI Synthetic Data SDK** to train a synthetic-data generator on an existing dataset and then generate new synthetic records from the learned patterns.

The project uses the SDK in **LOCAL mode**, meaning the training and synthetic-data generation happen on the local machine rather than sending the dataset to a remote SDK service.

The overall workflow is:

```text
Real Dataset
     │
     ▼
MOSTLY AI SDK
     │
     ▼
Train Synthetic Data Generator
     │
     ▼
Learn Patterns & Relationships
     │
     ▼
Generate Synthetic Records
     │
     ▼
Synthetic Dataset
     │
     ▼
CSV Output
```

---

# 2. High-Level Documentation

## 2.1 The Story — Why Does This Project Exist?

Imagine a company has a customer dataset containing information such as:

* Age
* Income
* City
* Purchase behavior

The company wants to build and test machine-learning systems using this data.

However, real-world datasets can contain sensitive or restricted information.

For example:

```text
Customer A
Age: 34
Income: ₹72,000
City: Kolkata
Purchased: Yes
```

Using real customer records everywhere can create privacy, security, compliance, and data-access challenges.

This is where **synthetic data** becomes useful.

Instead of distributing the original customer records, we can train a synthetic-data generator on the real dataset.

The generator learns the statistical patterns and relationships within the data and creates new records.

For example:

```text
Original:

34, 72000, Kolkata, 1


Synthetic:

31, 68000, Kolkata, 1
42, 91000, Delhi, 1
27, 43000, Mumbai, 0
```

The synthetic records are newly generated records rather than simply copying the original dataset.

---

# 3. What Is Synthetic Data?

Synthetic data is artificially generated data that is designed to preserve useful characteristics of real-world data.

For example, suppose the real data contains:

```text
Age → Income
Age → Purchase probability
City → Purchase behavior
```

A synthetic-data model attempts to learn these relationships.

It can then generate new records that follow similar patterns.

The goal is not simply:

```text
Make random fake rows
```

The goal is:

```text
Learn useful patterns from real data
             ↓
Generate new data
             ↓
Preserve important characteristics
```

---

# 4. Why Synthetic Data Is Useful

Synthetic data can be useful for:

### Development

Developers can work with realistic-looking data without requiring access to production records.

### Testing

Applications can be tested with large datasets without exposing real customer information.

### Machine Learning

Synthetic data can supplement training datasets or help with experimentation.

### Data Sharing

Organizations can share synthetic datasets when direct sharing of the original data is undesirable.

### Data Augmentation

Additional samples can be generated when certain segments of a dataset are underrepresented.

### Privacy-Aware Workflows

Synthetic-data systems can incorporate privacy mechanisms and quality checks.

The MOSTLY AI SDK provides capabilities including synthetic-data generation, conditional simulation, rebalancing, imputation, fairness controls, and quality/privacy evaluation.

---

# 5. What Is MOSTLY AI?

MOSTLY AI provides an open-source **Synthetic Data SDK** for generating high-fidelity synthetic data.

The SDK supports:

* Tabular data
* Sequential data
* Textual data
* Mixed data types
* Multi-table data
* Conditional generation
* Data rebalancing
* Imputation
* Differential privacy
* Quality evaluation

The SDK exposes three important concepts:

```text
Generator
Synthetic Dataset
Connector
```

### Generator

A generator is trained on existing data.

```text
Real Data
    ↓
Generator Training
    ↓
Trained Generator
```

### Synthetic Dataset

The trained generator is used to produce new records.

```text
Trained Generator
       ↓
Synthetic Dataset
```

### Connector

Connectors allow the SDK to work with external data sources.

The official SDK documentation describes generators, synthetic datasets, and connectors as its core resources.

---

# 6. What Happens in This Project?

This project follows:

```text
customers_training.csv
          │
          ▼
       pandas
          │
          ▼
    MOSTLY AI SDK
          │
          ▼
   Train Generator
          │
          ▼
   Learned Data Patterns
          │
          ▼
 Generate Synthetic Data
          │
          ▼
synthetic_customers.csv
```

The input dataset contains:

```text
age
income
city
purchased
```

The generator learns relationships between these columns and generates new records.

---

# 7. LOCAL Mode vs CLIENT Mode

MOSTLY AI SDK supports two important execution modes.

## LOCAL Mode

In LOCAL mode:

```text
Your Computer
     │
     ├── Python
     ├── MOSTLY AI SDK
     ├── Model
     └── Dataset
```

Training and generation happen on your own compute resources.

This is the mode used by this project.

The official SDK documentation identifies LOCAL mode as the default mode for local training and generation.

---

## CLIENT Mode

CLIENT mode is different.

Your Python application becomes a client that connects to a remote SDK endpoint.

```text
Your Computer
     │
     │ HTTP/API
     ▼
Remote MOSTLY AI SDK
     │
     ▼
Training / Generation
```

The local Python package in CLIENT mode is lightweight and connects to the remote SDK endpoint.

The SDK documentation describes:

```python
from mostlyai.sdk import MostlyAI

mostly = MostlyAI(
    base_url="http://remote-server:8080"
)
```

The remote server performs the SDK operations.

---

# 8. Technical Documentation

# 8.1 Prerequisites

Required:

* Windows
* Python 3.11+
* `uv`
* VS Code (recommended)
* Git (optional)

The current MOSTLY AI SDK documentation requires Python 3.11 or higher and recommends using a dedicated `uv` virtual environment.

---

# 9. Project Structure

Recommended structure:

```text
mostly-ai-demo/
│
├── .venv/
│
├── data/
│   ├── real/
│   │   └── customers_training.csv
│   │
│   └── synthetic/
│       └── synthetic_customers.csv
│
├── generate.py
├── pyproject.toml
├── uv.lock
└── README.md
```

### Directory explanation

```text
.venv/
```

Python virtual environment.

```text
data/real/
```

Contains the original training dataset.

```text
data/synthetic/
```

Contains generated synthetic datasets.

```text
generate.py
```

Main Python program.

```text
pyproject.toml
```

Project and dependency configuration.

```text
uv.lock
```

Locks dependency versions for reproducible environments.

```text
README.md
```

Project documentation.

---

# 10. Installation

## Step 1 — Create the project

Open PowerShell:

```powershell
mkdir mostly-ai-demo
cd mostly-ai-demo
```

---

## Step 2 — Create a Python environment

Create a Python 3.12 environment:

```powershell
uv venv --python 3.12
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
```

Expected:

```text
Python 3.12.x
```

---

# 11. Install MOSTLY AI

For LOCAL mode:

```powershell
uv pip install -U "mostlyai[local]"
```

The `[local]` extra installs the dependencies required for local training and generation, including the required ML stack.

Verify:

```powershell
python -c "from mostlyai.sdk import MostlyAI; print('MOSTLY AI installed successfully')"
```

Expected:

```text
MOSTLY AI installed successfully
```

---

# 12. Prepare the Dataset

Create the directories:

```powershell
mkdir data
mkdir data\real
mkdir data\synthetic
```

Place the training dataset here:

```text
data/real/customers_training.csv
```

Example:

```csv
age,income,city,purchased
25,35000,Kolkata,0
42,90000,Delhi,1
31,52000,Mumbai,1
55,120000,Pune,1
```

---

# 13. Generate Synthetic Data

Create:

```text
generate.py
```

Add:

```python
import pandas as pd

from mostlyai.sdk import MostlyAI


# Initialize MOSTLY AI SDK in LOCAL mode
mostly = MostlyAI()


# Load real training data
real_data = pd.read_csv(
    "data/real/customers_training.csv"
)

print("Real dataset:")
print(real_data.head())

print()
print(f"Rows: {len(real_data)}")
print(f"Columns: {list(real_data.columns)}")


# Train a synthetic data generator
print("\nTraining synthetic data generator...")

generator = mostly.train(
    name="Customer Synthetic Generator",
    data=real_data,
)

print("Training completed.")


# Generate synthetic data
print("\nGenerating synthetic data...")

synthetic_dataset = mostly.generate(
    generator
)


# Convert generated dataset to pandas DataFrame
synthetic_data = synthetic_dataset.data()

print("\nSynthetic dataset:")
print(synthetic_data.head())

print()
print(f"Rows: {len(synthetic_data)}")


# Save synthetic dataset
synthetic_data.to_csv(
    "data/synthetic/synthetic_customers.csv",
    index=False,
)

print(
    "\nSynthetic data saved to "
    "data/synthetic/synthetic_customers.csv"
)
```

The official SDK follows the same fundamental workflow:

```python
g = mostly.train(
    name="...",
    data=data,
)

sd = mostly.generate(g)

df = sd.data()
```

---

# 14. Run the Project

From the project directory:

```powershell
uv run python generate.py
```

Expected flow:

```text
Real dataset:
...

Rows: 1000
Columns: ['age', 'income', 'city', 'purchased']

Training synthetic data generator...

Training completed.

Generating synthetic data...

Synthetic dataset:
...

Rows: 1000

Synthetic data saved to data/synthetic/synthetic_customers.csv
```

---

# 15. Detailed Technical Flow

The implementation can be understood as five stages.

## Stage 1 — Load Data

```python
real_data = pd.read_csv(
    "data/real/customers_training.csv"
)
```

The CSV is loaded into a pandas DataFrame.

```text
CSV
 ↓
pandas
 ↓
DataFrame
```

---

## Stage 2 — Initialize SDK

```python
mostly = MostlyAI()
```

Since no remote `base_url` is provided, the SDK operates in its default LOCAL mode.

```text
MostlyAI()
    ↓
Local SDK
```

---

## Stage 3 — Train Generator

```python
generator = mostly.train(
    name="Customer Synthetic Generator",
    data=real_data,
)
```

The SDK analyzes the training data and trains a generator.

Conceptually:

```text
Real Data
   │
   ├── Numerical patterns
   ├── Categorical distributions
   ├── Relationships
   └── Dependencies
          │
          ▼
     Trained Generator
```

The SDK currently uses TabularARGN-based models for structured tabular data.

---

# 16. What Does the Model Learn?

Suppose:

```text
age,income,purchased

25,35000,0
30,50000,1
40,80000,1
50,110000,1
```

The model can learn patterns such as:

```text
Age distribution
Income distribution
Purchase distribution
Age ↔ Income relationship
Age ↔ Purchase relationship
Income ↔ Purchase relationship
```

It does **not** simply perform:

```text
random.choice(real_rows)
```

Instead, it learns a generative representation of the dataset and uses that representation to produce new samples.

---

# 17. Stage 4 — Generate Data

```python
synthetic_dataset = mostly.generate(
    generator
)
```

The trained generator is used to create synthetic records.

Conceptually:

```text
Trained Generator
       │
       ▼
Random sampling
       │
       ▼
Generated records
```

The number of records can be controlled according to the SDK generation configuration.

The SDK also supports probing a generator for representative samples.

---

# 18. Stage 5 — Convert to DataFrame

```python
synthetic_data = synthetic_dataset.data()
```

This converts the generated synthetic dataset into a pandas DataFrame.

Now normal pandas operations can be used:

```python
synthetic_data.head()
synthetic_data.describe()
synthetic_data.to_csv(...)
```

---

# 19. Stage 6 — Save the Result

```python
synthetic_data.to_csv(
    "data/synthetic/synthetic_customers.csv",
    index=False,
)
```

Final flow:

```text
customers_training.csv
        │
        ▼
   MOSTLY AI
        │
        ▼
   Generator
        │
        ▼
Synthetic Dataset
        │
        ▼
Pandas DataFrame
        │
        ▼
synthetic_customers.csv
```

---

# 20. Generator vs Synthetic Dataset

These two concepts should not be confused.

## Generator

The generator is the trained model.

```text
Real Dataset
     ↓
Training
     ↓
Generator
```

It can be reused.

---

## Synthetic Dataset

The synthetic dataset is generated from the generator.

```text
Generator
    ↓
Synthetic Dataset
```

For example:

```text
One Generator
      │
      ├── Generate 1,000 rows
      ├── Generate 10,000 rows
      ├── Generate 100,000 rows
      └── Generate 1,000,000 rows
```

The SDK documentation explicitly separates trained generators from generated synthetic datasets and allows a generator to be reused.

---

# 21. Quality Assurance

Synthetic data should not simply be generated and immediately trusted.

The important question is:

> How similar is the synthetic data to the useful characteristics of the real data?

MOSTLY AI provides quality-assurance capabilities and HTML reports for visual analysis.

A generator can be inspected with:

```python
generator.reports(
    display=True
)
```

The SDK also provides tutorials for:

* Train-Synthetic-Test-Real
* Quality metrics
* Privacy metrics
* Fairness
* Rebalancing
* Differential privacy

---

# 22. Synthetic Data Validation

A useful validation workflow is:

```text
                 REAL DATA
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
      Train Model        Keep Test Data
          │                   │
          ▼                   │
 Generate Synthetic           │
          │                   │
          ▼                   │
 Compare Synthetic            │
          │                   │
          └─────────┬─────────┘
                    ▼
             Quality Analysis
```

Important dimensions include:

### Statistical Fidelity

Do the distributions resemble the original data?

### Relationship Preservation

Are important relationships between columns preserved?

### Machine Learning Utility

Can models trained on synthetic data perform useful tasks?

### Privacy

Does the synthetic data reduce exposure of individual records?

### Fairness

Are important population groups represented appropriately?

---

# 23. Conditional Generation

The SDK also supports conditional generation.

Instead of:

```text
Generate any customer
```

you can request:

```text
Generate customers where:

city = Kolkata
```

Conceptually:

```text
Condition
   │
   ▼
Generator
   │
   ▼
Synthetic records satisfying condition
```

The SDK documentation provides conditional sampling examples using seed data.

---

# 24. Probing a Generator

For quickly inspecting generated samples:

```python
df = mostly.probe(
    generator,
    size=100
)
```

This is useful when you don't want to create a large synthetic dataset yet.

For example:

```text
Train Generator
      ↓
    Probe
      ↓
100 samples
      ↓
Inspect
```

The SDK supports probing trained generators for representative samples.

---

# 25. Local Execution Architecture

The current project uses:

```text
┌───────────────────────────────────────┐
│          Your Windows Machine         │
│                                       │
│  ┌─────────────────────────────────┐  │
│  │ Python 3.12                    │  │
│  │                                 │  │
│  │ pandas                          │  │
│  │ MOSTLY AI SDK                   │  │
│  │ ML dependencies                 │  │
│  │                                 │  │
│  │ ┌─────────────────────────────┐ │  │
│  │ │ Synthetic Data Generator    │ │  │
│  │ └─────────────────────────────┘ │  │
│  └─────────────────────────────────┘  │
│                                       │
│  Real Dataset                         │
│       ↓                               │
│  Synthetic Dataset                    │
│                                       │
└───────────────────────────────────────┘
```

No remote SDK endpoint is required for this LOCAL workflow.

---

# 26. CLIENT Architecture

CLIENT mode changes the architecture:

```text
┌──────────────────┐
│ Your Application │
│                  │
│ MostlyAI()       │
└────────┬─────────┘
         │
         │ HTTP/API
         ▼
┌──────────────────┐
│ Remote SDK       │
│                  │
│ Training         │
│ Generation       │
│ Models           │
└──────────────────┘
```

Example:

```python
from mostlyai.sdk import MostlyAI

mostly = MostlyAI(
    base_url="http://remote-server:8080"
)
```

The official SDK also documents a Docker-based SDK server that exposes port `8080`, which can then be accessed using `base_url`.

---

# 27. LOCAL vs CLIENT

| Feature                | LOCAL                  | CLIENT                                |
| ---------------------- | ---------------------- | ------------------------------------- |
| Model execution        | Local                  | Remote endpoint                       |
| Dataset processing     | Local                  | Remote SDK                            |
| Requires remote server | No                     | Yes                                   |
| Python SDK             | Yes                    | Yes                                   |
| Suitable for learning  | Yes                    | Useful for client/server architecture |
| Network dependency     | No SDK server required | Required                              |
| Compute                | Your machine           | Remote machine                        |
| Main initialization    | `MostlyAI()`           | `MostlyAI(base_url=...)`              |

---

# 28. Data Connectors

MOSTLY AI also supports connectors for external data sources.

Examples include:

```text
PostgreSQL
MySQL
Microsoft SQL Server
Oracle
Snowflake
Google BigQuery
```

Additional connector extras can be installed depending on the data source.

For example:

```powershell
uv pip install -U "mostlyai[local, postgres]"
```

Conceptually:

```text
Database
    │
    ▼
Connector
    │
    ▼
MOSTLY AI Generator
    │
    ▼
Synthetic Data
```

---

# 29. Text Generation

MOSTLY AI also supports synthetic text generation.

The SDK can use:

```text
LSTM models
```

or:

```text
Pre-trained Hugging Face language models
```

for suitable textual data.

For example:

```text
Real Reviews
     │
     ▼
Language Model
     │
     ▼
Synthetic Reviews
```

The official documentation notes that GPU resources are strongly recommended for language-model workloads.

---

# 30. Sequential Data

MOSTLY AI also supports sequential/tabular data.

Example:

```text
Customer
   │
   ├── Transaction 1
   ├── Transaction 2
   ├── Transaction 3
   └── Transaction 4
```

This is different from a simple independent table.

The SDK's underlying engine supports sequential tabular data through context relationships.

---

# 31. Differential Privacy

Synthetic data does not automatically mean that the data is private.

Privacy must be explicitly considered.

MOSTLY AI supports differential privacy as an optional capability.

Conceptually:

```text
Real Data
    │
    ▼
Privacy Mechanism
    │
    ▼
Synthetic Generator
    │
    ▼
Synthetic Data
```

Differential privacy introduces formal controls on the influence of individual records.

The SDK documents differential-privacy training and provides dedicated tutorials for it.

---

# 32. Troubleshooting

## Python version problem

Check:

```powershell
python --version
```

Current SDK documentation requires:

```text
Python >= 3.11
```

---

## SDK import problem

Run:

```powershell
python -c "from mostlyai.sdk import MostlyAI; print('OK')"
```

If it fails, verify the environment:

```powershell
where python
```

and:

```powershell
uv pip list
```

---

## Virtual environment not activated

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```powershell
python --version
```

---

## Missing local dependencies

For LOCAL mode:

```powershell
uv pip install -U "mostlyai[local]"
```

---

# 33. Recommended Development Workflow

Use a crawl → walk → run approach.

The official SDK documentation recommends starting with a smaller subset and limited training time before scaling to larger workloads.

### Crawl

```text
Small dataset
+
Short training
+
Small synthetic output
```

Validate that the pipeline works.

### Walk

```text
Larger dataset
+
More training
+
More synthetic records
```

Evaluate quality.

### Run

```text
Full dataset
+
Production configuration
+
Large synthetic dataset
```

Perform proper quality and privacy validation.

---

# 34. Complete Command Reference

## Create project

```powershell
mkdir mostly-ai-demo
cd mostly-ai-demo
```

## Create environment

```powershell
uv venv --python 3.12
```

## Activate

```powershell
.venv\Scripts\Activate.ps1
```

## Install SDK

```powershell
uv pip install -U "mostlyai[local]"
```

## Verify

```powershell
python -c "from mostlyai.sdk import MostlyAI; print('MOSTLY AI installed successfully')"
```

## Create directories

```powershell
mkdir data
mkdir data\real
mkdir data\synthetic
```

## Run generator

```powershell
uv run python generate.py
```

## Inspect generated file

```powershell
dir data\synthetic
```

---

# 35. End-to-End Summary

```text
                 REAL DATA
                     │
                     ▼
              ┌─────────────┐
              │   pandas    │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ MOSTLY AI   │
              │     SDK     │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   Train     │
              │  Generator  │
              └──────┬──────┘
                     │
                     ▼
              Learned Patterns
                     │
                     ▼
              ┌─────────────┐
              │   Generate  │
              │ Synthetic   │
              │    Data     │
              └──────┬──────┘
                     │
                     ▼
              pandas DataFrame
                     │
                     ▼
          synthetic_customers.csv
                     │
                     ▼
            Quality Evaluation
```

---

# 36. Key Takeaways

### High-Level

MOSTLY AI allows an organization to take a real dataset, train a synthetic-data generator, and produce new data that preserves useful characteristics of the original dataset.

The main story is:

```text
Real Data
   ↓
Learn Patterns
   ↓
Generate New Data
   ↓
Validate Quality
   ↓
Use Synthetic Data
```

### Technical

The core SDK workflow is:

```python
from mostlyai.sdk import MostlyAI

mostly = MostlyAI()

generator = mostly.train(
    name="Customer Synthetic Generator",
    data=real_data,
)

synthetic_dataset = mostly.generate(
    generator
)

synthetic_data = synthetic_dataset.data()
```

The generator can then be reused for additional synthetic-data workflows.

---

# 37. Project Outcome

After completing this project, the expected output is:

```text
data/
│
├── real/
│   └── customers_training.csv
│
└── synthetic/
    └── synthetic_customers.csv
```

The project demonstrates:

* Synthetic data generation
* Local ML execution
* MOSTLY AI SDK usage
* Generator training
* Synthetic dataset generation
* pandas integration
* CSV input/output
* Generator reuse
* Conditional generation concepts
* Quality evaluation concepts
* Privacy concepts
* LOCAL vs CLIENT architecture
* Data connector concepts
* Tabular and sequential synthetic-data workflows

---

# 38. Official Resources

* MOSTLY AI SDK repository: https://github.com/mostly-ai/mostlyai
* MOSTLY AI Engine repository: https://github.com/mostly-ai/mostlyai-engine
* MOSTLY AI tutorials: https://github.com/mostly-ai/mostlyai/blob/main/docs/tutorials.md
* MOSTLY AI documentation: https://mostly-ai.github.io/mostlyai/

---

# 39. Final Architecture

```text
                        MOSTLY AI
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
       LOCAL            CLIENT         CONNECTORS
          │                │                │
          ▼                ▼                ▼
      Local ML       Remote SDK       External Data
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                   Synthetic Generator
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
       Tabular          Text           Sequential
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    Synthetic Data
                           │
                           ▼
                 Quality / Privacy
                     Evaluation
```

The MOSTLY AI SDK is therefore not simply a CSV faker. It is a higher-level synthetic-data framework around trained generators, generation, connectors, evaluation, privacy, conditional sampling, and related data workflows.
