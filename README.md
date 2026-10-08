Synthetic Data Generation
Overview
This project explores multiple approaches for generating synthetic tabular data and evaluates the methods using the experiments that were actually performed.
The main goals are to understand:
- How synthetic data compares with real data
- How different synthetic-data generation methods work
- How well important patterns are preserved
- How runtime and environment affect generation
- What limitations appeared during experimentation
- The estimated pros and cons of each method
- When each method may be useful
The methods explored are:
1. CTGAN
2. GReaT
3. DDPM / TabDDPM
4. NVIDIA NeMo Data Designer
5. Google Simula
6. MOSTLY AI
Important: Measured experimental results and estimated method-level advantages/disadvantages are kept separate. Methods that were researched but not locally benchmarked are clearly identified.

1. CTGAN
What is CTGAN?
CTGAN stands for Conditional Tabular Generative Adversarial Network.
It is designed specifically for structured/tabular datasets containing numerical and categorical columns.
CTGAN mainly uses two components:
Generator
Creates artificial records that attempt to resemble the patterns found in the real dataset.
Discriminator
Attempts to distinguish real records from synthetic records.
During training, both components improve against each other until the generator can create new records that resemble the characteristics of the original data.
2. CTGAN Customer Dataset Experiment
An initial CTGAN experiment used a customer dataset.
Dataset Size
Real Dataset
- Rows: 1,000
- Columns: 4
Synthetic Dataset
- Rows: 1,000
- Columns: 4
Duplicate Comparison
- Real duplicates: 11
- Synthetic duplicates: 6
The synthetic dataset contained fewer duplicate records than the original dataset.
This indicates that the model generated new combinations rather than simply reproducing every original row. Duplicate count alone, however, is not enough to determine synthetic-data quality.
Missing Values
Synthetic missing values:
0
The generated dataset did not introduce additional missing values in this experiment.
3. Customer Dataset Fidelity
Synthetic-data generation success does not automatically mean the generated data accurately represents the original data.
Statistical comparisons were therefore performed.
Age
- KS statistic: 0.0280
- p-value: 0.8282
The real and synthetic Age distributions were reasonably similar in this experiment.
Income
- KS statistic: 0.0290
- p-value: 0.7947
The real and synthetic Income distributions also showed reasonable similarity.
City
- Chi-square statistic: 4.7276
- p-value: 0.5792
The categorical distribution of City showed reasonable similarity in this small experiment.
Purchased
- Chi-square statistic: 0.8809
- p-value: 0.3479
The distribution of Purchased was also reasonably preserved.
Overall Customer Experiment Observation
CTGAN successfully generated:
- The requested 1,000 rows
- The same 4 columns
- No synthetic missing values
- Reasonably similar numerical distributions
- Reasonably similar categorical distributions
The runtime for this experiment was not recorded and is therefore not estimated.
4. HR Attrition Synthetic Data Experiment
A more realistic tabular experiment used the IBM HR Analytics — Employee Attrition & Performance dataset.
Actual Dataset
- Rows: 1,470
- Columns: 35
- Missing values: 0
- Duplicate rows: 0
Synthetic Dataset
- Rows: 1,470
- Columns: 35
- Missing values: 0
- Duplicate rows: 0
The synthetic HR dataset was generated using CTGAN through SynthCity.
The dataset contains numerical and categorical employee information, including fields related to:
- Age
- Attrition
- Department
- Job role
- Income
- Business travel
- Education
- Experience
- Overtime
- Years at the company
5. Actual Data vs Synthetic Data
The real and synthetic HR datasets were directly compared.
The HTML report displays:
- 10 actual records
- 10 synthetic records
- 10 matching representative columns
The comparison uses a clear visual separation between actual and synthetic data.
The objective is not for synthetic rows to be identical to real rows.
The objective is for the synthetic dataset to preserve useful structural and statistical characteristics of the original dataset while producing new records.
6. HR Dataset Profiling
Both datasets were profiled using the same profiling workflow.
All 35 columns were included.
The profiling comparison includes measurements such as:
- Null percentage
- Unique percentage
- Minimum
- Maximum
- Mean
- Median
- Standard deviation
- Variance
- Quantiles
- IQR
- Outliers
- Most frequent categorical values
- Category frequencies
- Category percentages
The report uses:
- 🟢 MATCH
- 🟡 PARTIAL MISMATCH
- 🔴 MISMATCH
These indicators provide a simple visual interpretation of how closely the synthetic profile resembles the real profile.
They are visual aids and do not replace statistical testing.
7. HR Dataset Observations
The synthetic HR dataset preserved several characteristics of the real dataset reasonably well.
However, some columns showed larger differences.
Notable fields requiring attention included:
- Attrition
- JobRole
- MonthlyIncome
- YearsInCurrentRole
- YearsWithCurrManager
This demonstrates an important point:
A model can successfully generate a complete synthetic dataset while individual columns still differ from the original dataset.

Therefore, synthetic-data evaluation should be performed at both dataset level and column level.
8. Fidelity Evaluation
What is Fidelity?
Fidelity describes how closely synthetic data preserves useful characteristics of the original dataset.
Realistic-looking rows alone are not enough.
A synthetic dataset should also preserve important distributions and relationships.
Numerical Fidelity
Numerical fields can be compared using:
- Mean
- Standard deviation
- Minimum
- Maximum
- Distribution comparisons
- KS statistic
- p-value
A smaller KS statistic generally indicates closer numerical distributions.
Categorical Fidelity
Categorical fields can be evaluated using:
- Category frequencies
- Category percentages
- Chi-square statistics
- p-values
- Total Variation Distance where applicable
Additional Checks
The project also considers:
- Missing values
- Unique values
- Duplicate records
- Correlations where meaningful
- Column-level distribution differences
9. GReaT
What is GReaT?
GReaT is a transformer-based approach for synthetic tabular-data generation.
Instead of treating the data only as traditional rows and columns, tabular records can be represented in a form that a language model can learn.
The model then generates new records based on the patterns it has learned.
Dataset Used
The experiment used the LAPD dataset.
Original dataset:
- Rows: 2,133,137
- Columns: 28
Smaller subsets were also explored:
- 20,000 × 16
- 5,000 × 9
Recorded Training Experiment
- Training records: 500
- Training steps: 63
- Training time: 656.47 seconds
- Training loss: 2.678913
Generation Test
- Generated records: 10
- Approximate generation time: 4 minutes 48 seconds
Observation
GReaT demonstrated that synthetic records could be generated in the tested environment.
However, the local experiment was slow and was affected by:
- Compilation/toolchain limitations
- Environment configuration
- Hardware limitations
- Lack of effective acceleration
These results describe the local experiment and should not be interpreted as a universal limitation of GReaT.
10. DDPM / TabDDPM
What is DDPM?
DDPM stands for Denoising Diffusion Probabilistic Model.
Diffusion models use a different approach from GANs.
A simplified workflow is:
1. Start with real data.
2. Gradually add noise.
3. Train the model to learn how to remove that noise.
4. Start generation from noise.
5. Repeatedly denoise.
6. Produce a synthetic sample.
TabDDPM adapts diffusion modelling for tabular datasets.
Local Experiment
The local experiment encountered:
- CPU-only execution
- Missing effective CUDA/GPU support
- KeOps C++ compiler limitations
- Missing OpenMP support
- Categorical encoding issues
- Class-value preprocessing problems
One observed error indicated that class values needed to be non-negative.
Because the experiment did not complete as a clean standardized benchmark, no comparable fidelity result is reported.
This should not be interpreted as proof that TabDDPM cannot work. The experiment was constrained by the local environment, dependencies, hardware and preprocessing setup.
11. NVIDIA NeMo Data Designer
NVIDIA NeMo Data Designer was explored using a product-review generation workflow.
The input field was:
product_category
The generated field was:
review
Recorded Results
- Generated records: 10
- Columns: 2
- Successful records: 10 / 10
- Completion: 100%
- Runtime: 127.8 seconds
Token Usage
- Input tokens: 298
- Output tokens: 1,896
- Total tokens: 2,194
Configuration
- Model: nvidia/nemotron-3.5-lightning-30b-a3b
- Maximum parallel requests: 4
- Temperature: 1.00
- Top-p: 0.95
- Thinking: Disabled
Observation
All requested records were generated successfully.
However, this was a small 10-record experiment, so it does not establish large-scale performance or million-row scalability.
12. Google Simula
Google Simula was explored as part of the research into modern synthetic-data generation approaches.
It was not locally benchmarked during this project.
Therefore, this project does not report:
- Local runtime
- Local fidelity measurements
- Local scalability measurements
- Local resource measurements
It is included as a research/evaluation method rather than a measured local benchmark.
13. MOSTLY AI
MOSTLY AI was also researched as a synthetic-data platform.
It was not locally benchmarked during this project.
Therefore, no project-specific:
- Runtime
- Fidelity result
- Generation benchmark
- Scalability measurement
is reported.
It is included as part of the broader investigation into available synthetic-data approaches.
14. Side-by-Side Method Comparison
Method	Real / Input Data	Synthetic Output	Runtime	Fidelity Tested?	Main Observation
CTGAN – Customer	1,000 × 4	1,000 × 4	Not recorded	Yes	Successfully generated data with measurable similarity
CTGAN – HR Attrition	1,470 × 35	1,470 × 35	Not recorded	Yes	Detailed profiling showed strong matches and some column-level differences
GReaT – SynthCity	LAPD dataset, up to 2,133,137 × 28	10-row generation test	656.47 sec training + ~4m48s generation	Limited	Worked locally but generation was slow
DDPM / TabDDPM	Tabular datasets	Experimental	CPU/local environment	No standardized result	Environment, compilation and preprocessing issues
NVIDIA NeMo Data Designer	Product-review setup	10 × 2	127.8 sec	No full fidelity benchmark	10/10 records generated
Google Simula	Not locally benchmarked	Not locally benchmarked	—	—	Research/evaluation only
MOSTLY AI	Not locally benchmarked	Not locally benchmarked	—	—	Research/evaluation only


15. Estimated Pros and Cons of Each Method
Important: The following pros and cons describe estimated/general method-level characteristics. They are intentionally separated from the measured experimental results above. Runtime, memory and hardware requirements are treated as a common consideration rather than being repeated as disadvantages of individual methods.

CTGAN — Estimated Pros
- Designed specifically for tabular data.
- Supports numerical and categorical columns.
- Suitable for datasets such as HR, customer and business data.
- Can learn relationships between multiple columns.
- Generated output can be evaluated using standard profiling and statistical tests.
- Worked successfully in the measured CTGAN experiments in this project.
CTGAN — Estimated Cons
- Highly imbalanced categorical columns can be difficult to reproduce accurately.
- Rare categories or minority patterns may be underrepresented or missed.
- GAN training can be unstable and results may vary between training runs.
- Mode collapse can reduce the diversity of generated records.
- Good overall distributions do not guarantee that every individual column is reproduced equally well.
- Complex relationships involving several columns may be harder to preserve than individual distributions.
- Hyperparameters and training configuration can noticeably affect output quality.
- Successful generation does not itself provide a privacy guarantee; privacy and memorization risk require separate evaluation.
GReaT — Estimated Pros
- Uses transformer-based modelling.
- Provides an alternative to traditional GAN-based tabular generation.
- Can represent complex relationships between fields through a language-model-style representation.
- Useful for investigating contextual dependencies between attributes.
- Synthetic records were successfully produced in the local experiment.
GReaT — Estimated Cons
- Converting structured rows into text adds another modelling layer between the original table and generated data.
- Output quality can depend strongly on how columns and values are represented as text.
- Generated records may require post-processing and validation before being treated as valid structured data.
- Strict numerical relationships and numerical precision can be harder to control through language-model generation.
- Invalid or logically inconsistent combinations may be generated unless constraints or validation are applied.
- Results depend partly on the behaviour and limitations of the underlying pretrained language model.
- Rare categories and unusual combinations may still be difficult to reproduce reliably.
- Plausible-looking rows do not guarantee preservation of the original statistical distributions, so fidelity analysis is still necessary.
DDPM / TabDDPM — Estimated Pros
- Provides a modern diffusion-based alternative to GANs.
- TabDDPM is specifically intended for tabular data.
- Can model complex tabular distributions when numerical and categorical variables are represented appropriately.
- Useful for comparing diffusion approaches with GAN and transformer approaches.
- Does not rely on the traditional generator-versus-discriminator GAN structure.
DDPM / TabDDPM — Estimated Cons
- Tabular datasets contain heterogeneous feature types, making joint modelling of continuous and discrete variables challenging.
- Categorical and continuous variables require different treatment and careful preprocessing.
- Discrete variables remain a challenging part of tabular diffusion modelling.
- Mixed-type features can be particularly difficult to represent correctly.
- Output quality can depend on diffusion configuration and the selected noise/denoising process.
- Rare classes and unusual regions of the original distribution may require additional attention.
- Generated records still need validation for business rules and logical consistency.
- Preprocessing or encoding mistakes can prevent training or produce misleading synthetic output.
NVIDIA NeMo Data Designer — Estimated Pros
- Provides a configuration-driven synthetic-data workflow.
- Useful for LLM-assisted and text-rich synthetic-data generation.
- Supports dependent-field generation.
- Supports composable stages, column dependencies, validators and quality checks.
- Supports multi-model workflows and model routing.
- The project achieved 10/10 successful generations in the measured experiment.
NVIDIA NeMo Data Designer — Estimated Cons
- Data Designer orchestrates generation but does not host the underlying models, so output quality depends partly on the configured model endpoint/provider.
- Complex datasets may require careful design of prompts, column dependencies, constraints and validators.
- Overloading a single generation call with schema, diversity, content and quality requirements can reduce output quality.
- Generated text can still contain incorrect, hallucinated or logically inconsistent information.
- Schema compliance does not automatically mean the generated dataset preserves the desired statistical distribution.
- Provider availability, API behaviour and rate limits can affect generation workflows.
- Different underlying models can produce noticeably different datasets from the same high-level configuration.
- Validators and quality checks must still be designed for the intended use case.
Google Simula — Estimated Pros
- Explores reasoning-first synthetic dataset design rather than only learning a distribution from an existing table.
- Useful when dataset coverage, complexity and quality need to be deliberately designed.
- Provides a different research direction from conventional tabular GAN and diffusion generation.
- Can be useful for constructing training or evaluation scenarios from first-principles mechanisms.
Google Simula — Estimated Cons
- Dataset quality depends strongly on the reasoning capability of the underlying model.
- Reasoning errors can propagate into the generated dataset design.
- A carefully designed mechanism can still encode incorrect assumptions about the target domain.
- Because the approach can be seedless, its generated distribution is not simply learned from a real reference table.
- Domain-specific generation requires careful specification of desired coverage, complexity and quality.
- Poorly defined target characteristics can produce a well-structured dataset that represents the wrong problem distribution.
- Generated examples still require downstream validation; reasoning-based generation does not automatically guarantee real-world validity.
- Direct comparison with CTGAN/TabDDPM-style real-to-synthetic fidelity testing is less straightforward because the generation objective is different.
- In this project, Simula was research-only, so no local benchmark was available to validate these characteristics against the project datasets.
MOSTLY AI — Estimated Pros
- Provides a dedicated synthetic-data workflow for structured/tabular data.
- Can reduce the amount of custom model engineering required.
- Includes privacy-oriented mechanisms and model quality reporting.
- Platform-oriented tooling can simplify end-to-end synthetic-data workflows.
MOSTLY AI — Estimated Cons
- Rare-category protection can replace uncommon values with _RARE_, intentionally removing some original detail.
- Sampled replacement of rare categories can alter the original category distribution.
- Privacy protection and statistical accuracy can therefore involve a trade-off.
- High-cardinality categorical columns and noisy categories require careful preparation and encoding choices.
- Incorrect encoding can generate invalid synthetic values, such as treating categorical numeric codes as continuous measurements.
- Incorrectly formatted datetime fields may be interpreted as categorical values.
- Bivariate relationships between columns can become weakened or lost.
- Extreme-value privacy protection can intentionally remove unusual numerical or datetime cases before training, so some edge cases may not appear in the generated data.
- Quality therefore depends partly on selecting appropriate encoding and privacy settings.
- Quality reports are still needed to identify distribution and relationship problems.
Common Consideration Across All Methods
Runtime, memory usage and hardware requirements depend on dataset size, model configuration, implementation and deployment environment. Because these factors are not unique to one synthetic-data method, they are not treated as method-specific disadvantages in this comparison.

The measured runtime values from this project remain valid experimental observations and are reported separately in the relevant experiment sections.
16. Major Observations
Generation Success Does Not Equal Data Quality
Successfully generating the requested number of rows only proves that generation completed.
It does not prove that the generated rows accurately represent the original dataset.
Fidelity must be evaluated separately.
Fidelity Must Be Measured
Synthetic data should be compared against real data using:
- Numerical distributions
- Categorical distributions
- Missing values
- Unique values
- Duplicate behaviour
- Correlations
- Column-level profiling
Runtime Matters
Recorded examples include:
GReaT
- Training: 656.47 seconds
- Generation: approximately 4 minutes 48 seconds for 10 rows
NVIDIA NeMo Data Designer
- Generation: 127.8 seconds for 10 records
CTGAN
- Runtime: not recorded
Missing runtime values should not be estimated or invented.
17. Small Experiments Cannot Prove Scalability
A method working on:
- 10 rows
- 500 rows
- 1,000 rows
does not automatically prove that it will perform equally well on:
- 100,000 rows
- 1,000,000 rows
- 10,000,000 rows
Scalability must be tested separately.
18. Hardware and Environment Matter
The experiments showed that environment configuration can strongly affect synthetic-data workflows.
Important factors include:
- CPU
- GPU
- CUDA availability
- RAM
- Compiler availability
- OpenMP support
- Package compatibility
- Model dependencies
A method failing or running slowly in one local environment does not automatically mean the method itself is poor.
19. Data Preparation Matters
Correct preprocessing is especially important for tabular synthetic-data models.
Important areas include:
- Numerical columns
- Categorical columns
- Missing values
- Class labels
- Encoding
- Data types
The TabDDPM experiment particularly demonstrated how preprocessing and environment problems can prevent a clean experiment.
20. What a Fair Benchmark Should Look Like
The current experiments provide useful practical observations, but they are not a perfectly controlled benchmark.
Different methods were tested using different:
- Datasets
- Dataset sizes
- Column counts
- Output sizes
- Configurations
A fair benchmark should use the same conditions.
Same Dataset
Every model should receive exactly the same real dataset.
Same Number of Training Rows
Every model should train using the same number of records.
Same Columns
Every model should receive the same features.
Same Synthetic Output Size
For example:
- Real dataset: 100,000 rows
- CTGAN output: 100,000 rows
- GReaT output: 100,000 rows
- TabDDPM output: 100,000 rows
Same Hardware
Every experiment should run on the same:
- CPU
- GPU
- RAM
- Operating environment
Same Evaluation Measurements
Every model should be compared using:
- Training time
- Generation time
- Memory usage
- CPU usage
- GPU usage
- Numerical fidelity
- Categorical fidelity
- Correlation preservation
- Missing values
- Duplicate behaviour
- Scalability
21. Suggested Scalability Benchmark
A future controlled experiment could test every model using:
Test 1
1,000 rows
Test 2
10,000 rows
Test 3
100,000 rows
Test 4
1,000,000 rows
This would make it possible to compare how runtime, resource usage and fidelity change as dataset size increases.
22. Overall Deductions
Synthetic-data generation should not be evaluated using only one measurement.
Three areas are especially important.
1. Data Quality
Does the synthetic data preserve important patterns from the original data?
This requires fidelity analysis.
2. Performance
How much time and computational resources are required?
This includes:
- Training time
- Generation time
- CPU usage
- GPU usage
- Memory requirements
3. Scalability
Can the approach continue working when the dataset becomes much larger?
A successful small experiment does not automatically prove large-scale performance.
23. Key Lessons
Lesson 1
Synthetic-data generation and synthetic-data quality are different things.
Lesson 2
Every generated dataset should be profiled.
Lesson 3
Fidelity should be measured rather than assumed.
Lesson 4
Runtime should be recorded for meaningful comparisons.
Lesson 5
Runtime, memory, hardware and environment configuration can affect synthetic-data workflows across methods.
Lesson 6
Categorical preprocessing is especially important for tabular diffusion approaches.
Lesson 7
Small experiments test functionality but do not prove production scalability.
Lesson 8
Different methods should only be directly compared when tested under equivalent conditions.
Lesson 9
Estimated pros and cons should not be presented as measured project results.
24. When to Use Each Approach
CTGAN
Useful when:
- Working with structured/tabular datasets
- Data contains numerical and categorical columns
- A GAN-based approach is suitable
- Statistical comparison against real data is required
- A dedicated tabular synthetic-data generator is desired
GReaT
Useful when:
- Exploring transformer-based tabular generation
- Testing language-model-style approaches
- Relationships between attributes may benefit from transformer modelling
- Sufficient computational resources are available
- Slower generation is acceptable
DDPM / TabDDPM
Useful when:
- Exploring diffusion-based synthetic-data generation
- Researching tabular diffusion models
- Correct numerical and categorical preprocessing is available
- Compilation dependencies are correctly configured
- Suitable computational resources are available
NVIDIA NeMo Data Designer
Useful when:
- Generating structured synthetic records
- Generating text-rich synthetic data
- Creating data using controlled prompts
- Generating one field based on another
- Using an LLM-based generation workflow
Google Simula
Useful to investigate when:
- Researching newer synthetic-data approaches
- Comparing modern generation techniques
- A suitable environment is available for proper benchmarking
No local benchmark was completed in this project.
MOSTLY AI
Useful to investigate when:
- Looking for a platform-oriented synthetic-data workflow
- Evaluating managed synthetic-data generation
- Privacy and data utility are important considerations
No local benchmark was completed in this project.
25. Final Summary
This project explored multiple synthetic-data generation approaches:
- CTGAN
- GReaT
- DDPM / TabDDPM
- NVIDIA NeMo Data Designer
- Google Simula
- MOSTLY AI
CTGAN provided the strongest direct real-vs-synthetic fidelity analysis in the experiments.
The HR Attrition CTGAN experiment generated 1,470 synthetic rows across 35 columns, matching the dimensions of the real dataset. Profiling demonstrated that several characteristics were preserved well while other columns showed noticeable differences.
GReaT successfully produced synthetic records in the recorded local experiment but showed relatively slow generation and was affected by local environment and toolchain limitations.
DDPM / TabDDPM experimentation was constrained by preprocessing, compilation, dependency and hardware limitations, preventing a clean standardized benchmark.
NVIDIA NeMo Data Designer successfully generated 10/10 requested records in the product-review experiment, with a recorded runtime of 127.8 seconds.
Google Simula and MOSTLY AI were explored at the research/evaluation level and were not locally benchmarked.
Estimated pros and cons were also considered for each method, but these are intentionally kept separate from the project's measured results.
The central conclusion is:
Synthetic data should not be judged only by whether a model successfully generates rows. It should also be evaluated for fidelity, performance, scalability and preservation of important characteristics from the original dataset.

Because the methods were tested with different datasets, sizes and configurations, the current experiments should not be used to declare a universal winner.
A controlled benchmark using the same dataset, hardware, row count, columns, output size and evaluation metrics would be required for a true direct comparison.