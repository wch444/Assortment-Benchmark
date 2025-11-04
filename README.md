# Hard Instances for Assortment Optimization under MMNL and NL Choice Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-SSRN-red.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5671592)

This repository provides a benchmark dataset for assortment optimization problems under two popular discrete choice models: the **Mixed Multinomial Logit (MMNL)** and **Nested Logit (NL)** choice models.  

This benchmark provides hard instances we generated using a systematic approach (see our paper), and an accessible interface to test the performance of algorithms designed by you. The code is designed for **reproducibility, extensibility, and comparability**.

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Hard Instance Data](#-hard-instance-data)
- [User Guide](#-user-guide)
- [Extending the Framework](#️-extending-the-framework)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [References](#-references)

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
│── setup_env.sh                # Environment setup script (cross-platform)
│── README.md                   # Project documentation

```

---

## 🚀 Quick Start

### Prerequisites
- **Python**: Version 3.9 or higher (supports 3.9, 3.10, 3.11, and 3.12)
- **Shell**: Unix-like shell (bash/zsh) for macOS/Linux/WSL users

### Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/wch444/Assortment-Benchmark.git
cd Assortment-Benchmark
```

#### Step 2: Set Up Environment

Choose one of the following methods based on your preference:

##### Option 1: Automated Setup (Recommended)

Use our setup script for hassle-free installation:

```bash
# Grant execute permission (first time only)
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

**What the script does:**
1. ✅ Checks and installs `uv` (ultra-fast Python package installer) if needed
2. ✅ Prompts you to select Python version (3.9-3.12 or system default)
3. ✅ Creates an isolated virtual environment
4. ✅ Installs all required dependencies from `requirements.txt`
5. ✅ Verifies installation and displays installed packages

**Platform Support:**
- ✅ macOS
- ✅ Linux  
- ✅ Windows WSL (Windows Subsystem for Linux)
- ⚠️  Native Windows (requires manual uv installation via PowerShell)

---

##### Option 2: Manual Setup with uv

For users who prefer manual control:

```bash
# Install uv (if not already installed)
# macOS/Linux/WSL:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell as Administrator):
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create a Python 3.11 virtual environment
uv venv --python 3.11 .venv-py311

# Install dependencies
uv pip install --python .venv-py311 -r requirements.txt

# Activate the environment
source .venv-py311/bin/activate  # macOS/Linux/WSL
# .venv-py311\Scripts\activate    # Windows
```

---

##### Option 3: Traditional pip Installation

If you prefer the standard Python toolchain:

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv .venv

# Activate the environment
source .venv/bin/activate          # macOS/Linux/WSL
# .venv\Scripts\activate            # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### Running the Examples

Once the environment is set up, you can run the example notebooks:

```bash
# Activate the virtual environment (if not already activated)
source .venv-py311/bin/activate  # Use the appropriate environment name

# Launch Jupyter Notebook
jupyter notebook
```

**Using VS Code:**
1. Open any `.ipynb` file in the `src/` directory
2. Click the **kernel selector** in the top-right corner
3. Select your virtual environment (e.g., `.venv-py311`)
4. Run the notebook cells

---

### Dependencies

**Core Dependencies** (automatically installed):
- Python ≥ 3.9, ≤ 3.12
- numpy ≥ 1.20, < 3.0
- pandas ≥ 1.3, < 3.0
- matplotlib ≥ 3.0, < 4.0
- seaborn ≥ 0.11, < 1.0
- openpyxl ≥ 3.0, < 4.0
- ipykernel ≥ 6.0, < 7.0

**Optional: Gurobi Solver**

For exact optimization methods (e.g., `conic_mmnl_warm_start`):
- gurobipy ≥ 11.0, < 13.0

> **📌 Note**: Gurobi requires separate installation and a valid license. Download from [Gurobi's official website](https://www.gurobi.com/). **Academic licenses are free** for qualifying users. All heuristic methods work without Gurobi.

---

## 📝 Hard Instance Data

The `hard_data/` folder provides **pre-generated challenging instances** for benchmarking assortment optimization algorithms under both **Mixed Multinomial Logit (MMNL)** and **Nested Logit (NL)** choice models.

All instances are stored in JSON format and can be loaded directly using utility functions in `generator/utils.py`.

Below we describe how we generate and select these hard instances.

---

### 1. Instance Generation
- For MMNL model, the data is generated from the function `mmnl_data_v0_lognorm`
- For NL model, the data is generated from the functions `nl_data_vi0_uniform01` and `nl_data_vi0_uniform34`

---

### 2. Instance Selection

To ensure that the provided instances are genuinely **challenging and representative of difficult cases**, we followed a systematic selection process:

- **Initial Generation**: For each parameter combination (e.g., specific values of m, n, and cap_rate), we generated 100 candidate instances by controlling the random seed (seeds 1-100).

-  **Multi-Method Evaluation**: Each candidate was evaluated using multiple state-of-the-art algorithms, including:
   - Revenue-ordered heuristic [[Talluri et al. (2004)](#Talluri2004), [Davis et al. (2014)](#Davis2014)]
   - ADXOpt algorithm [[Gallego et al. (2024b)](#Gallego2024b)]
   - AlphaPhi heuristic [[Gallego et al. (2024a)](#Gallego2024a)]
   - LP-based policy [[Kunnumkal (2023)](#Kunnumkal2023)]
   - Our proposed neural network-based policy

- **Hard Instance Identification**: For each algorithm, we identified the five instances with  the largest optimality gap (greater than $10^{-4}$), representing the most challenging instances where the algorithms performed worst.

- **Union of Challenging Cases**: We take the union of all identified hard instances across all tested methods, ensuring that each instance is difficult for at least one (and often multiple) method(s).

- **Final Dataset Composition**: The resulting hard instances in the `hard_data/` folder represent cases where existing methods struggle, making them ideal benchmarks for evaluating new algorithms.


**Optimal Solution Calculation:**
- **MMNL instances**: The optimal revenue is computed by solving the mixed-integer conic program formulation proposed by [Şen et al. (2018)](#Şen2018) using Gurobi. When Gurobi fails to find the exact optimal solution within a reasonable time limit, we use the best assortment found across all compared methods as the benchmark.
- **NL instances**: Due to the computational complexity of finding exact optimal solutions for large-scale NL problems, all methods are evaluated against the theoretical upper bound developed by [Kunnumkal (2023)](#Kunnumkal2023). This upper bound provides a performance guarantee for assessing solution quality.


**Key Statistics:**
- Each parameter combination typically contains 3–10 hard instances
- Instances are selected to maximize algorithmic difficulty rather than random sampling

**This selection methodology ensures that researchers can:**
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
 
**Instance parameters configuration:**
  - Number of products ($n$): {50, 100, 200}
  - Number of customer segments ($m$): {5, 10, 25}
  - Cardinality rates of constraints (`cap_rate`): {0.1, 0.3, 0.5}
  - Each ($m$, $n$, `cap_rate` (if applicable)) combination contains multiple instances with different random seeds

#### 2) NL (Nested Logit) 

**File naming convention**: `nl_{constraint}_{vi0_method}_data.json`
- Constraint types: `unconstrained`, `card` (cardinality per nest)
- vi0 distribution methods: `01` (vi0 ~ U(0, 1), low within-nest utility), `34` (vi0 ~ U(3, 4), high within-nest utility)

**Instance parameters configuration:**
  - Number of nests ($m$): {5, 10, 20}
  - Number of products per nest ($n$): {25, 50}
  - Cardinality rates of constraints in each nest (`cap_rate`): {0.1, 0.3, 0.5} 
  - Each ($m$, $n$, `cap_rate` (if applicable)) combination contains multiple instances with different random seeds

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
- **Problem parameters**: $m$, $n$, `cap_rate` (if applicable)
- **Random seed**: For reproducibility
- **Optimal revenue**: Optimal revenue (`max_rev` for MMNL) or upper bound (`upper_bound` for NL)
- **Best-found revenues**: Maximum revenue obtained across all evaluations (`max_rev` of MMNL, `best_rev` for NL)
- **Corresponding assortment**: Corresponding assortment (`best_ass`)
- **Related data**: `u`, `price`, `v0`, `omega` (for MMNL); `price`, `v`, `gamma`, `v0`, `vi0` (for NL)

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
opt_rev = data.max_rev # For MMNL
opt_rev = data.upper_bound # For NL
print(f"Optimal revenue: {opt_rev:.4f}") # For MMNL
print(f"Optimal revenue: {opt_rev:.4f}") # For NL

# Implement your method
assortment = your_algorithm(data)

# Evaluate
revenue_fn = get_revenue_function_mmnl(data)  # or get_revenue_function_nl
revenue = revenue_fn(assortment)[0]
gap = (opt_rev - revenue) / opt_rev * 100
print(f"Your gap: {gap:.2f}%")
```

### 4. Output and Analysis

Notebooks generate:
- **Detailed statistics tables**: Mean, std, min, max gaps by problem size
- **Visualizations**: Box plots, bar charts, distribution analyses
- **Excel reports**: Comprehensive results saved to `results/{model}_summary_statistics.xlsx` folder
- **Performance comparisons**: Side-by-side analysis across methods and parameters

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

This repository accompanies the ongoing work  **Solving Assortment Optimization with First-Order Methods and Neural Networks: A Computational Framework and Public Benchmark** [link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5671592)

If you use this repository, please cite it in your work.

### Citation

Use the following BibTeX:
```bibtex
@misc{GuoLagziWangEtAl2025,
  title = {Solving Assortment Optimization with First-Order Methods and Neural Networks: A Computational Framework and Public Benchmark},
  author = {Guo, Qing and Lagzi, Saman and Wang, Chenhao and Chen, Ningyuan and Gallego, Guillermo and Kunnumkal, Sumit and Wang, Yao and Yu, Li},
  year = {2025},
  howpublished = {SSRN Electronic Journal},
  url = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5671592},
  note = {Available at SSRN: \url{https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5671592}}
}
```


## 📖 References

<a id="Rogosinski2024"></a>[1] Rogosinski S, Müller S, Reyes-Rubiano L. Distribution-specific approximation guarantees for the random-parameters logit assortment problem. 2024.

<a id="Şen2018"></a>[2] Şen A, Atamtürk A, Kaminsky P. A conic integer optimization approach to the constrained assortment problem under the mixed multinomial logit model. *Operations Research*, 2018, 66(4): 994-1003.

<a id="Kunnumkal2023"></a>[3] Kunnumkal S. New bounds for cardinality-constrained assortment optimization under the nested logit model. *Operations Research*, 2023, 71(4): 1112-1119.

<a id="Gallego2024a"></a>[4] Gallego G, Gao P, Wang S, Berbeglia G. Assortment optimization with downward feasibility: Efficient heuristics based on independent demands. Available at SSRN 5021867, 2024.

<a id="Gallego2024b"></a>[5] Gallego G, Jagabathula S, Lu W. Efficient local-search heuristics for online and offline assortment optimization. Available at SSRN 4828069, 2024.

<a id="Davis2014"></a>[6] Davis JM, Gallego G, Topaloglu H. Assortment optimization under variants of the nested logit model. *Operations Research*, 2014, 62(2): 250-273.

<a id="Talluri2004"></a>[7] Talluri K, Van Ryzin G. Revenue management under a general discrete choice model of consumer behavior. *Management Science*, 2004, 50(1): 15-33.