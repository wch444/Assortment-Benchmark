# Hard Instances for Assortment Optimization under MMNL and NL Choice Models

This repository provides a dataset for the assortment optimization problems under two popular discrete choice models: the **Mixed Multinomial Logit (MMNL)** and **Nested Logit (NL)** choice models.  

This dataset provides hard instances we generated using a systematic approach (see our paper), and an accessible interface to test the performance of an algorithm designed by you. The code is designed for **reproducibility, extensibility, and comparability**.

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
│    ├── plot.py                         # Functions for data analysis and visualization
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
- Python (=3.11.13)
- NumPy (=2.3.2)
- Pandas (=12.0.3)
- Gurobi (=12.0.3)

You can use the requirements.txt files with pip to install a fully predetermined and working environment.
```bash
pip install -r requirements.txt
```
> **Note:** Gurobi must be installed separately. You can download it from [Gurobi's official website](https://www.gurobi.com/) and obtain a valid license for activation.

---

## 📝 Hard Instance Data
This `hard_data/` folder provides **pre-generated challenging instances** for benchmarking assortment optimization algorithms under both **Mixed Multinomial Logit (MMNL)** and **Nested Logit (NL)** choice models.

All instances is stored in JSON format and can be loaded directly using utility functions in `generator/utils.py`.

Below we describe how we generate and select these hard instances.

---

### 1. Instance Generation
- For MMNL model, the data is generated from the function `mmnl_data_v0_lognorm`
- For NL model, the data is generated from the function `nl_data_vi0_uniform01` and `'nl_data_vi0_uniform01`

---

### 2. Instance Selection

To ensure that the provided instances are genuinely **challenging and representative of difficult cases**, we followed a systematic selection process:

- **Initial Generation**: For each parameter combination (e.g., specific values of m, n, and cap_rate), we generated 100 candidate instances by controlling the random seed (seeds 1-100).

-  **Multi-Method Evaluation**: Each candidate was evaluated using multiple state-of-the-art algorithms, including:
   - Revenue-ordered heuristic [[Talluri et al. (2004)](#Talluri2004)、[Davis et al. (2014)](#Davis2014)]
   - ADXOpt algorithm [[Gallego et al. (2024b)](#Gallego2024b)]
   - AlphaPhi heuristic [[Gallego et al. (2024a)](#Gallego2024a)]
   - LP-based policy: [[Kunnumkal (2023)](#Kunnumkal2023)]
   - Our proposed neural network-based policy

- **Hard Instance Identification**: For each algorithm, we identified the five instances with  the largest optimality gap (greater than $10^{-4}$), representing the most challenging instances where the algorithms performed worst.

- **Union of Challenging Cases**: The union of all identified hard instances across all tested methods, ensuring that each instance is difficult for at least one (and often multiple)  method.

- **Final Dataset Composition**: The resulting hard instances in the `hard_data/` folder represent cases where existing methods struggle, making them ideal benchmarks for evaluating new algorithms.


**Optimal Solution Calculation**:
- **MMNL instances**: The optimal revenue is computed by solving the mixed-integer conic program formulation proposed by [Şen et al. (2018)](#Şen2018) using Gurobi. When Gurobi fails to find the exact optimal solution within a reasonable time limit, we use the best assortment found across all compared methods as the benchmark.
- **NL instances**: Due to the computational complexity of finding exact optimal solutions for large-scale NL problems, all methods are evaluated against the theoretical upper bound developed by [Kunnumkal (2023)](#Kunnumkal2023). This upper bound provides a performance guarantee for assessing solution quality.


**Key Statistics**:
- Each parameter combination typically contains 3-10 hard instances
- Instances are selected to maximize algorithmic difficulty rather than random sampling

**This selection methodology ensures that researchers can**:
- Test their algorithms on genuinely difficult problem instances
- Compare performance across multiple challenging scenarios
- Identify algorithmic weaknesses and opportunities for improvement

---

### 3. Data Overview
This section details the structure and configuration of the generated MMNL and NL instances used in experiments.

All datasets are stored in JSON format for easy parsing and reproducibility.

#### 1) MMNL (Mixed Multinomial Logit)

**File naming convention**: `mmnl_{constraint}_{revenue_curve}_data.json`
- Constraint types: `unconstrained`, `card` (cardinality)
- Revenue curves: `RS2`, `RS4`
 
**Instance parameters configuration**:
  - Number of products (n): {50, 100, 200}
  - Number of customer segments (m): {5, 10, 25}
  - Cardinality rates of constraints (cap_rate): {0.1, 0.3, 0.5}
  - Each (m, n, cap_rate`(if applicable)`) combination contains multiple instances with different random seeds

#### 2) NL (Nested Logit) 

**File naming convention**: `nl_{constraint}_{vi0_method}_data.json`
- Constraint types: `unconstrained`, `card` (cardinality per nest)
- vi0 distribution methods: `01` (vi0 ~ U(0, 1), low within-nest utility), `34` (vi0 ~ U(3, 4), high within-nest utility)

**Instance parameters configuration**:
  - Number of nests (m): {5, 10, 20}
  - Number of products per nest (n): {25, 50}
  - Cardinality rates of constraints in each nest (cap_rate): {0.1, 0.3, 0.5} 
  - Each (m, n, cap_rate`(if applicable)`) combination contains multiple instances with different random seeds

---

### 4. Loading Instances

```python
from generator.utils import load_MNL_instances, load_NL_instances

# Load MMNL instances
mmnl_instances = load_MNL_instances("hard_data/mmnl_card_RS2_data.json")

# Load NL instances
nl_instances = load_NL_instances("hard_data/nl_unconstrained_01_data.json")
```

---

### 5. Instance Data Structure

Each instance contains:
- **Problem parameters**: m, n, cap_rate (if applicable)
- **Random seed**: For reproducibility
- **Optimal revenue**: Corresponding optimal revenue (max_rev)
- **Related data**: u, price, v0, omega (for MMNL); price, v, gamma, v0, vi0 (for NL)

---

## 🚀 User Guide

The easiest way to get started is to run the example Jupyter notebooks located in the `src/` directory. Each notebook demonstrates how to load hard instances, implement your own algorithm, and evaluate its performance.

---

### 1. Example Notebooks for MMNL and NL Models

#### MMNL Model

- Unconstrained problem: [`src/mmnl_unconstrained_example.ipynb`](src/mmnl_unconstrained_example.ipynb)

- Cardinality constrained problem: [`src/mmnl_cardinality_example.ipynb`](src/mmnl_cardinality_example.ipynb). Implement algorithms that respect cardinality constraints, your algorithm must satisfy: `sum(assortment) <= cap_rate * n`

#### NL Model

- Unconstrained Problem:[`src/nl_unconstrained_example.ipynb`](src/nl_unconstrained_example.ipynb)

- Cardinality-Constrained Problem:[`src/nl_cardinality_example.ipynb`](src/nl_cardinality_example.ipynb). Implement algorithms with nested cardinality constraints `sum(assortment_i) <= cap_rate * n` for each nest `i`

---

### 2. General Workflow for All Notebooks

Each notebook follows a consistent structure:

- **Import Required Modules**: Load necessary libraries and utility functions
-  **Load Hard Instances**: Load pre-generated hard instances from JSON files `hard_data/`
-   **Explore Instance Structure**: Visualize data distributions and problem characteristics
-    **Implement Your Algorithm**: 
  ```python
  # TODO: Replace this section with your method
  assortment = your_algorithm(data.m, data.n, ...)
  ```
- **Evaluate Performance**: Calculate revenue and optimality gaps
- **Save Results**: Export detailed performance metrics to Excel
- **Analyze Results**: Generate comprehensive statistics and visualizations


### 3. Quick Start Example

```python
from generator.utils import load_MMNL_instances, load_NL_instances
from models.mmnl_functions import get_revenue_function_mmnl
from models.nl_functions import get_revenue_function_nl

# Load instances
# For MMNL
instances = load_MMNL_instances("hard_data/mmnl_unconstrained_RS2_data.json")

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

### 4. Output and Analysis

Notebooks generate:
- **Detailed statistics tables**: Mean, std, min, max gaps by problem size
- **Visualizations**: Box plots, bar charts, distribution analyses
- **Excel reports**: Comprehensive results saved to `results/{model}_summary_statistics.xlsx` folder
- **Performance comparisons**: Side-by-side analysis across methods and parameters

---

## 📚 Module Documentation

This section provides detailed documentation for the main modules in this repository. These modules support data generation, constraint modeling, optimization algorithms, and performance evaluation.

---

### 1. Data Generators (`generator/`)

The `generator/` directory contains functions for creating synthetic problem instances. These generators are used to create both the hard instances in `hard_data/` and allow you to generate custom instances.

- #### MMNL Data Generators

The following is the data generation method of the MMNL model, you can import all generators as:
```python
from generator.mmnl_data_generator import *
```

| Function | Description |
| -------- | ----------- |
| `mmnl_data_v0_lognorm` | **Used for hard instances**. Captures continuous heterogeneity among customer segments with log-normal no-purchase utilities `v0`. Supports multiple product revenue curves (RS2, RS4). |
| `mmnl_data_random` | Randomly generates utilities and prices, with no-purchase utility taking values 1 or 5 for half of the segments. |
| `mmnl_data_easy` | Following [Şen et al. (2018)](#Şen2018). Generates uniformly random utilities and prices with equal segment weights and no-purchase utility. |
| `mmnl_data_hard` | Following [Şen et al. (2018)](#Şen2018). Creates sparse utility matrix where each customer type has only k products with positive utility. |

- #### NL Data Generators

The following is the data generation method of the NL model, you can import all generators as:
```python
from generator.nl_data_generator import *
```

| Function | Description |
| -------- | ----------- |
| `nl_data_vi0_uniform01` | **Used for hard instances**. Extension of `nested_data_NewBounds` with low within-nest no-purchase utility `vi0 ~ U(0, 1)`. |
| `nl_data_vi0_uniform34` | **Used for hard instances**. Extension of `nested_data_NewBounds` with high within-nest no-purchase utility `vi0 ~ U(3, 4)`. |
| `nl_data_vi0_lognormal` | Extension of `nested_data_NewBounds` with long-tail distribution `vi0 ~ LogNormal(μ=1, σ=0.5)` clipped to [1, 5]. |
| `nested_data_NewBounds` | Following [Kunnumkal (2023)](#Kunnumkal2023). Creates nested structure with smooth price-utility relationships. |
| `nested_data_random` | Following [Gallego et al. (2024)](#Gallego2024). Generates random prices and utilities within user-defined ranges. |
| `nested_data_complex` | Following [Davis et al. (2014)](#Davis2014). Generates complex nested data with nonlinear interactions. |

---

### 2. Constraint Generators (`generator/constraint.py`)

These functions generate various constraints for assortment optimization. Each returns `(A, B)` representing linear constraints $Ax \leq B$, where $x$ is the binary assortment vector.

You can import them as:
```python
from generator.constraint import *
```

| Function | Description |
| -------- | ----------- |
| `cardinality` | Generates cardinality constraint: at most `cap` products can be selected. Returns constraint ensuring `sum(x) <= cap`. |
| `card_nested_logit` | NL-specific, which restricts the maximum number of products within each nest. Returns `m` separate constraints, one per nest. |
| `cons_capacity` | Generates capacity constraints with different randomized structures for more complex scenarios. |

---

### 3. Optimization Methods (`method/`)

These modules implement various optimization algorithms for solving assortment problems.

- #### General Methods

You can import general methods:
```python
from method.general_method import *
```

| Function | Description |
| -------- | ----------- |
| `revenue_order` | Revenue-ordered heuristic [[Talluri et al. (2004)](#Talluri2004)]. Sorts products by revenue and selects high-revenue items. Works for both MMNL and NL models. |

- #### MMNL-Specific Methods

You can import methods to solve the MMNL model:
```python
from method.mmnl_method import *
```

| Function | Description |
| -------- | ----------- |
| `conic_mmnl_warm_start` | Exact method using conic integer programming formulation [[Şen et al. (2018)](#Şen2018)]. Finds globally optimal assortment using Gurobi solver. |

- #### NL-Specific Methods

You can import methods to solve the NL model:
```python
from method.nl_method import *
```

| Function | Description |
| -------- | ----------- |
| `revenue_order_nl` | LP-based algorithm [[Davis et al. (2014)](#Davis2014)] where each nest contains the $k_i$ highest-revenue products. |

---

### 4. Evaluation Functions (`models/`)

These modules provide functions to evaluate assortment performance and calculate revenues.

- #### MMNL Evaluation Functions

Import MMNL functions:
```python
from models.mmnl_functions import *
```

| Function | Description |
| -------- | ----------- |
| `get_revenue_function_mmnl` | Returns a revenue function for a given MMNL instance. The returned function takes an assortment and computes expected revenue. |

- #### NL Evaluation Functions

Import NL functions:
```python
from models.nl_functions import *
```

| Function | Description |
| -------- | ----------- |
| `get_revenue_function_nl` | Returns a revenue function for a given NL instance. The returned function takes an assortment matrix and computes expected revenue. |

---

### 5. Utility Functions (`generator/utils.py`)
The module provides helper functions for loading hard instances used in assortment optimization experiments.


You can import them as:
```python
from generator.utils import *
```

| Function | Description |
| -------- | ----------- |
| `load_MNL_instances` | Loads MMNL instances from JSON file. Returns list of instance objects with all problem parameters and optimal solutions. |
| `load_NL_instances` | Loads NL instances from JSON file. Returns list of instance objects with nest structures and optimal solutions. |

---

## 🧩 Key Features

- **Two choice models supported**: Mixed Multinomial Logit (MMNL) and Nested Logit (NL)
- **Multiple data generators**: Create custom instances or use pre-generated hard instances
- **Flexible constraints**: Cardinality, capacity, and nest-specific constraints
- **Exact and heuristic solvers**: Gurobi-based exact methods and fast heuristics
- **Comprehensive evaluation**: Built-in functions for computing revenues and optimality gaps
- **Reproducibility**: All instances include random seeds for exact replication

---

## 🛠️ Extending the Framework

This codebase is designed to be easily extensible:

- **Add new data generators**: Create new functions in `generator/mmnl_data_generator.py` or `generator/nl_data_generator.py`
- **Implement new algorithms**: Add methods to `method/mmnl_method.py` or `method/nl_method.py`
- **Define custom constraints**: Extend `generator/constraint.py` with new constraint types
- **Support new choice models**: Create new modules following the structure of existing `models/` files

### Example: Adding a New Method

```python
# In method/mmnl_method.py
def my_new_heuristic(m, n, u, price, v0, omega, constraint=None):
    """
    Your algorithm description here
    
    Args:
        m: number of customer segments
        n: number of products
        u: utility matrix (m x n)
        price: product prices (n,)
        v0: no-purchase utilities (m,)
        omega: segment weights (m,)
        constraint: optional linear constraint (A, B) where A @ x <= B
    
    Returns:
        assortment: binary vector of length n
    """
    # Your implementation here
    assortment = ...
    return assortment
```

---

## 📄 License

This project is released under the MIT License.

---

## 🙌 Acknowledgments

This repository accompanies the ongoing work [link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5671592)

If you use this repository, please cite it in your work.


## 📖 References
<a id="Rogosinski2024"></a> [1] Rogosinski S, Müller S, Reyes-Rubiano L. Distribution-specific approximation guarantees for the random-parameters logit assortment problem[J]. 2024.  
<a id="Şen2018"></a> [2] Şen A, Atamtürk A, Kaminsky P. A conic integer optimization approach to the constrained assortment problem under the mixed multinomial logit model[J]. Operations Research, 2018, 66(4): 994-1003.  
<a id="Kunnumkal2023"></a> [3] Kunnumkal S. New bounds for cardinality-constrained assortment optimization under the nested logit model[J]. Operations Research, 2023, 71(4): 1112-1119.  
<a id="Gallego2024a"></a> [4] Gallego G, Gao P, Wang S, Berbeglia G (2024a) Assortment optimization with downward feasibility: Efficient
heuristics based on independent demands. Available at SSRN 5021867, 2024.
<a id="Gallego2024b"></a> [5] Gallego G, Jagabathula S, Lu W. Efficient Local-Search Heuristics for Online and Offline Assortment Optimization[J]. Available at SSRN 4828069, 2024.  
<a id="Davis2014"></a> [6] Davis J M, Gallego G, Topaloglu H. Assortment optimization under variants of the nested logit model[J]. Operations Research, 2014, 62(2): 250-273.  
<a id="Talluri2004"></a> [7] Talluri K, Van Ryzin G. Revenue management under a general discrete choice model of consumer behavior[J]. Management Science, 2004, 50(1): 15-33.  
