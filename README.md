# Hard Instance for Assortment Optimization under MMNL and NL Choice Models

This repository provides a framework for testing the hard assortment optimization problems under two popular discrete choice models: the **Mixed Multinomial Logit (MMNL)** and **Nested Logit (NL)** choice models.  

This project provides the hard instances we obtained in the paper, and to test the performance of a given method. The code is designed for **reproducibility, extensibility, and comparability**.

---

## 📂 Project Structure
The repository is organized into several key directories:

```
root/
│── generator/                  # Synthetic data generators
│    ├── constraint.py          # Capacity and structural constraints
│    ├── mmnl_data_generator.py # Data generator for MMNL instances
│    ├── nl_data_generator.py   # Data generator for NL instances
│    ├── utils.py               # Load the data from the json file
│
│── method/                     # Optimization algorithms
│    ├── general_method.py      # General optimization methods
│    ├── mmnl_method.py         # Heuristic algorithms for MMNL
│    ├── nl_method.py           # Heuristic algorithms for NL
│
│── models/                     # Functions for evaluating performance
│    ├── mmnl_functions.py      # MMNL-specific functions
│    ├── nl_functions.py        # NL-specific functions
│
│── src/                        # Example notebooks
│    ├── mmnl_cardinality_example.ipynb  # MMNL with cardinality constraint
│    ├── mmnl_unconstrained_example.ipynb # MMNL unconstrained problem
│    ├── nl_cardinality_example.ipynb    # NL with cardinality constraint
│    ├── nl_unconstrained_example.ipynb  # NL unconstrained problem
│
│── hard_data/                  # Pre-generated hard instances (JSON files)
│    ├── mmnl_card_RS2_data.json         # MMNL cardinality - RS2 revenue curve
│    ├── mmnl_card_RS4_data.json         # MMNL cardinality - RS4 revenue curve
│    ├── mmnl_unconstrained_RS2_data.json # MMNL unconstrained - RS2 revenue curve
│    ├── mmnl_unconstrained_RS4_data.json # MMNL unconstrained - RS4 revenue curve
│    ├── nl_card_01_data.json            # NL cardinality - vi0 ~ Uniform(0,1)
│    ├── nl_card_34_data.json            # NL cardinality - vi0 ~ Uniform(3,4)
│    ├── nl_unconstrained_01_data.json   # NL unconstrained - vi0 ~ Uniform(0,1)
│    └── nl_unconstrained_34_data.json   # NL unconstrained - vi0 ~ Uniform(3,4)
│
│── requirements.txt            # Python dependencies
│── README.md                   # Project documentation

```

---

## ⚙️ Installation

To get started with this project, follow these steps.

```bash
git clone https://github.com/wch444/Assortment-Benchmark.git
```

For install dependencies, the project requires the following:
- Python (>=3.11.13)
- NumPy (>=2.3.2)
- Pandas (>=12.0.3)
- Gurobi (>=12.0.3)

You can use the requirements.txt files with pip to install a fully predetermined and working environment.
```bash
pip install -r requirements.txt
```
> **Note:** Gurobi must be installed separately. You can download it from [Gurobi's official website](https://www.gurobi.com/) and obtain a valid license for activation.

---

## Hard Data Files

The `hard_data/` folder contains pre-generated challenging instances for benchmarking assortment optimization algorithms. All instances are stored in JSON format and can be loaded using the utilities in `generator/utils.py`.

### The Generation of the Instances
- For MMNL model, the data is generated from the function `generator/mmnl_data_v0_lognorm.py`
- For NL model, the data is generated from the function `generator/'nl_data_vi0_uniform01.py` and `generator/'nl_data_vi0_uniform01.py`

### The Selection of the Instances

To ensure that the provided instances are genuinely challenging and representative of difficult cases, we followed a systematic selection process:

1. **Initial Generation**: For each parameter combination (e.g., specific values of m, n, and cap_rate), we generated 100 candidate instances by controlling the random seed (seeds 0-99).

2. **Multi-Method Evaluation**: We evaluated each candidate instance using multiple state-of-the-art algorithms, including:
   - Revenue-ordered heuristic [[Talluri et al. (2004)](#Talluri2004)]
   - ADXOpt algorithm [[Gallego et al. (2024b)](#Gallego2024b)]
   - AlphaPhi heuristic [[Gallego et al. (2024a)](#Gallego2024a)]
   - Our proposed neural network-based policy

3. **Hard Instance Identification**: For each algorithm, we identified the 5 instances where it exhibited the largest optimality gap (i.e., the instances on which it performed worst).

4. **Union of Challenging Cases**: We took the union of all identified hard instances across all tested methods. This ensures that the final dataset includes instances that are challenging for at least one (and often multiple) algorithms.

5. **Final Dataset Composition**: The resulting hard instances in the `hard_data/` folder represent cases where existing methods struggle, making them ideal benchmarks for evaluating new algorithms.


**Optimal Solution Calculation**:
- **MMNL instances**: The optimal revenue is computed by solving the mixed-integer conic program formulation proposed by [Şen et al. (2018)](#Şen2018) using Gurobi. In cases where Gurobi fails to find the exact optimal solution within a reasonable time limit, we use the best assortment found across all compared methods as the benchmark.
- **NL instances**: Due to the computational complexity of finding exact optimal solutions for large-scale NL problems, all methods are evaluated against the theoretical upper bound developed by [Kunnumkal (2023)](#Kunnumkal2023). This upper bound provides a performance guarantee for assessing solution quality.


**Key Statistics**:
- Each parameter combination typically contains 3-10 hard instances
- Instances are selected to maximize algorithmic difficulty rather than random sampling

This selection methodology ensures that researchers can:
- Test their algorithms on genuinely difficult problem instances
- Compare performance across multiple challenging scenarios
- Identify algorithmic weaknesses and opportunities for improvement


### MMNL (Mixed Multinomial Logit) Instances

**File naming convention**: `mmnl_{constraint}_{revenue_curve}_data.json`

- **Constraint types**:
  - `unconstrained`: No capacity constraints
  - `card`: Cardinality constraint (limited number of products)

- **Revenue curves**:
  - `RS2`: Revenue curve type 2
  - `RS4`: Revenue curve type 4

- **Instance parameters**:
  - Number of products (n): {50, 100, 200}
  - Number of customer segments (m): {5, 10, 25}
  - Cardinality rates (for constrained): {0.1, 0.3, 0.5} × n
  - Each (m, n, cap_rate) combination contains multiple instances with different random seeds

### NL (Nested Logit) Instances

**File naming convention**: `nl_{constraint}_{vi0_method}_data.json`

- **Constraint types**:
  - `unconstrained`: No capacity constraints
  - `card`: Cardinality constraint (limited number of products)

- **vi0 distribution methods**:
  - `01`: vi0 ~ Uniform(0, 1) - Low outside-nest utility
  - `34`: vi0 ~ Uniform(3, 4) - High outside-nest utility

- **Instance parameters**:
  - Number of nests (m): {5, 10, 15}
  - Number of products per nest (n): {25, 50, 75}
  - Cardinality rates (for constrained): {0.1, 0.3, 0.5} × (m × n)
  - Each (m, n, cap_rate) combination contains multiple instances with different random seeds

### Loading Instances

```python
from generator.utils import load_MNL_instances, load_NL_instances

# Load MMNL instances
mmnl_instances = load_MNL_instances("hard_data/mmnl_card_RS2_data.json")

# Load NL instances
nl_instances = load_NL_instances("hard_data/nl_unconstrained_01_data.json")
```

### Instance Data Structure

Each instance contains:
- **Problem parameters**: m, n, cap_rate (if applicable)
- **Utility/preference data**: u (MMNL) or v (NL), v0, vi0 (NL), gamma (NL)
- **Price data**: price vector
- **Segment weights**: omega (MMNL)
- **Optimal solution**: max_rev (optimal revenue), optimal_assortment
- **Random seed**: For reproducibility


## 🚀 How to Use?

The easiest way to get started is to run the example Jupyter notebooks located in the `src/` directory. Each notebook demonstrates how to load hard instances, implement your own algorithm, and evaluate its performance.

### 1. Mixed Multinomial Logit (MMNL) Model

#### Unconstrained Problem
[`src/mmnl_unconstrained_example.ipynb`](src/mmnl_unconstrained_example.ipynb)

This notebook demonstrates:
- Loading pre-generated hard instances from `hard_data/`
- Understanding the instance structure and data distribution
- Implementing your own optimization algorithm
- Evaluating performance against optimal solutions
- Comparing results across different revenue curves (RS2 vs RS4)
- Analyzing optimality gaps across problem sizes (m, n combinations)

**Instance Parameters**:
- Number of products (n): {50, 100, 200}
- Number of customer segments (m): {5, 10, 25}
- Revenue curves: RS2 and RS4

#### Cardinality-Constrained Problem
[`src/mmnl_cardinality_example.ipynb`](src/mmnl_cardinality_example.ipynb)

This notebook covers:
- Loading cardinality-constrained instances
- Visualizing instance distribution across different dimensions
- Implementing algorithms that respect capacity constraints

**Additional Constraints**:
- Cardinality rates: {0.1, 0.3, 0.5} × n
- Your algorithm must satisfy: `sum(assortment) <= cap_rate * n`

---

### 2. Nested Logit (NL) Model

#### Unconstrained Problem
[`src/nl_unconstrained_example.ipynb`](src/nl_unconstrained_example.ipynb)

This notebook demonstrates:
- Loading and exploring NL instances with detailed visualizations
- Understanding nest structures, utilities, and dissimilarity parameters
- Implementing your algorithm for the NL choice model
- Evaluating against theoretical upper bounds
- Comparing performance across different vi0 distributions (uniform01 vs uniform34)
- Analyzing how problem size affects algorithm performance

**Instance Parameters**:
- Number of nests (m): {5, 10, 15}
- Number of products per nest (n): {25, 50, 75}
- vi0 distributions: Uniform(0,1) and Uniform(3,4)

**Key Visualizations Included**:
- Utility matrix heatmaps
- Price distributions
- Dissimilarity parameters by nest
- Within-nest no-purchase utilities
- Comprehensive statistical summaries

#### Cardinality-Constrained Problem
[`src/nl_cardinality_example.ipynb`](src/nl_cardinality_example.ipynb)

This notebook covers:
- Loading cardinality-constrained NL instances
- Understanding constraint structures across nests
- Implementing algorithms with nested cardinality constraints

**Additional Constraints**:
- Cardinality rates: {0.1, 0.3, 0.5} × (m × n)
- Your algorithm must return a binary matrix of shape (m, n)
- Constraint: `sum(assortment) <= cap_rate * m * n`

---

### General Workflow for All Notebooks

Each notebook follows a consistent structure:

1. **Import Required Modules**: Load necessary libraries and utility functions
2. **Load Hard Instances**: Read pre-generated challenging instances from JSON files
3. **Explore Instance Structure**: Visualize data distributions and problem characteristics
4. **Implement Your Algorithm**: 
   ```python
   # TODO: Replace this section with your method
   assortment = your_algorithm(data.m, data.n, ...)
   ```
5. **Evaluate Performance**: Calculate revenue and optimality gaps
6. **Analyze Results**: Generate comprehensive statistics and visualizations
7. **Save Results**: Export detailed performance metrics to Excel

### Quick Start Example

```python
# Load instances
from generator.utils import load_MNL_instances, load_NL_instances

# For MMNL
instances = load_MNL_instances("hard_data/mmnl_unconstrained_RS2_data.json")

# For NL
instances = load_NL_instances("hard_data/nl_card_01_data.json")

# Access instance data
data = instances[0]
print(f"Problem size: m={data.m}, n={data.n}")
print(f"Optimal revenue: {data.max_rev:.4f}")

# Implement your method
assortment = your_algorithm(data)

# Evaluate
revenue_fn = get_revenue_function_mmnl(data)  # or get_revenue_function_nl
revenue = revenue_fn(assortment)[0]
gap = (data.max_rev - revenue) / data.max_rev * 100
print(f"Your gap: {gap:.2f}%")
```

### Output and Analysis

Notebooks generate:
- **Detailed statistics tables**: Mean, std, min, max gaps by problem size
- **Visualizations**: Box plots, bar charts, distribution analyses
- **Excel reports**: Comprehensive results saved to `results/` folder
- **Performance comparisons**: Side-by-side analysis across methods and parameters

---

## 🔱 What's in there ?

### 1. Data Generation
Here summarizes the available data generation methods provided in  
`generator/`. 

The following is the data generation method of the MMNL model, you can import all generators as:
```bash
from generator.mmnl_data_generator import *
```

| Function| Description |
| ------------- | ----------------------------------------------- |
| `mmnl_data_v0_lognorm` | **Our constructed** data generator designed to capture continuous heterogeneity among customer segments. A few segments exhibit exceptionall high no-purchase utilities `v0`, following a log-normal distribution. And there are various product revenue curves.|
| `mmnl_data_random`     | Randomly generates related data, where the no-purchase utility takes values 1 or 5 for half if the segments.  |
| `mmnl_data_easy`       | Following the setting used in [Şen et al.(2018)](#Şen2018). Produces uniformly random utilities and prices, with equal segment weights and no-purchase utility. | 
| `mmnl_data_hard`       | Following the setting used in [Şen et al.(2018)](#Şen2018). Generates a sparse utility matrix, where each customer type includes only k products with utility greater than 0 and the number of customer types equals the number of products.| 

The following is the data generation method of the NL model, you can import all generators as:
```bash
from generator.nl_data_generator import *
```

| Function| Description |
| ------------- | ---------------------------------------------- |
| `nested_data_complex`   | Following the setting used in [Davis et al.(2014)](#Davis2014). Generates complex nested data with nonlinear interactions.|
| `nested_data_random`    | Following the setting used in [Gallego et al.(2024)](#Gallego2024). Generates random prices and utilities within user-defined ranges.|
| `nested_data_NewBounds` | Following the setting used in [Kunnumkal (2023)](#Kunnumkal2023). Creates complex nested structure data with smooth relationships between price and utility.|
| `nl_data_vi0_uniform01` | **Our extension** of `nested_data_NewBounds` introducing low within-nest no-purchase utility `vi0 ∼ U(0, 1)`. |
| `nl_data_vi0_uniform34` | **Our extension** of `nested_data_NewBounds` introducing high within-nest no-purchase utility `vi0 ∼ U(3, 4)`.  |
| `nl_data_vi0_lognormal` | **Our extension** of `nested_data_NewBounds` introducing long-tail distribution no-purchase utility `vi0 ∼ LogNormal(μ=1, σ=0.5)` clipped to [1, 5]. |

### 2. Constraint Generation
Here summarizes commonly used constraint generators for assortment optimization. Each function returns a pair `(A, B)` representing linear constraints of the form $Ax \leq B$, where $x$ is the binary assortment vector.

You can import them as:
```bash
from generator.constraints import *
```

| Function | Description |
| -------- | ----------- |
| `cardinality`| Generates a cardinality constraint ensuring that at most `cap` products can be selected. |
| `card_nested_logit` | Only applicable to NL models. It restricts the maximum number `cap` of products within each nest. | 
| `cons_capacity` | Generates capacity constraints with different randomized structures. |

### 3. Heuristic Optimization Method
Here summarizes the optimization algorithms implemented in our framework.  

The following method is used to solve the MMNL model, you can import all heuristics as:
```bash
from heuristic.general_heuristic import *
from heuristic.mmnl_heuristic import *
```
| Function| Description |
| ------------- | ---------------------------------------------- |
| `conic_mmnl_warm_start` | An exact method to find the globally optimal assortment by formulating the problem as the conic integer formulation. [Şen et al.(2018)](#Şen2018)|
| `revenue_order` | A simple heuristic that sorts products by price and selects all products with revenue greater than a threshold. [Talluri et al.(2004)](#Talluri2004)|

The following method is used to solve the NL model, you can import all heuristics as:
```bash
from heuristic.general_heuristic import *
from heuristic.nl_heuristic import *
```
| Function| Description |
| ------------- | ---------------------------------------------- |
| `revenue_order_nl` | A algorithm based on linear programming, where each nest $i$ contains the $k_i$ highest-revenue products. [Davis et al.(2014)](#Davis2014)| 

---


---

## 🧩 Key Features

- Supports **two choice models**: NL and MMNL
- Provides **exact solver** and **heuristic baselines**
- Provides **multiple data generation methods**
- Ensures **reproducibility** via fixed random seeds

---

## 🛠️ Extending

- Add new data generation in `generator/`
- Implement new heuristics in `heuristic/`
- Extend to new choice models by creating functions in the corresponding files

---

## 📄 License

This project is released under the MIT License.

---

## 🙌 Acknowledgments

This codebase is part of ongoing research on **Solving Hard Assortment Optimization Problems with Instance-Specific Neural Optimizer: A Theoretical and Computational Analysis**.  

If you use this repository, please cite it in your work.

## 📖 References
<a id="Rogosinski2024"></a> [1] Rogosinski S, Müller S, Reyes-Rubiano L. Distribution-specific approximation guarantees for the random-parameters logit assortment problem[J]. 2024.  
<a id="Şen2018"></a> [2] Şen A, Atamtürk A, Kaminsky P. A conic integer optimization approach to the constrained assortment problem under the mixed multinomial logit model[J]. Operations Research, 2018, 66(4): 994-1003.  
<a id="Kunnumkal2023"></a> [3] Kunnumkal S. New bounds for cardinality-constrained assortment optimization under the nested logit model[J]. Operations Research, 2023, 71(4): 1112-1119.  
<a id="Gallego2024"></a> [4] Gallego G, Jagabathula S, Lu W. Efficient Local-Search Heuristics for Online and Offline Assortment Optimization[J]. Available at SSRN 4828069, 2024.  
<a id="Davis2014"></a> [5] Davis J M, Gallego G, Topaloglu H. Assortment optimization under variants of the nested logit model[J]. Operations Research, 2014, 62(2): 250-273.  
<a id="Talluri2004"></a> [6] Talluri K, Van Ryzin G. Revenue management under a general discrete choice model of consumer behavior[J]. Management Science, 2004, 50(1): 15-33.  