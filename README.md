# Assortment Optimization for MMNL and NL Choice Models

This repository provides a Python framework for solving assortment optimization problems under two popular discrete choice models: the **Mixed Multinomial Logit (MMNL)** and **Nested Logit (NL)** choice models.  

The goal of assortment optimization is to select the subset of products to offer to customers to maximize expected revenue. This project provides tools to generate synthetic data, compute revenue, and find optimal assortments using both heuristic and exact optimization algorithms. The code is designed for **reproducibility, extensibility, and comparability**.

---

## 📂 Project Structure
The repository is organized into several key directories:

```
root/
│── generator/                  # Synthetic data generators
│    ├── constraint.py          # Capacity and structural constraints
│    ├── mmnl_data_generator.py # Data generator for MMNL instances
│    ├── nl_data_generator.py   # Data generator for NL instances
│
│── heuristic/                  # Heuristic baselines
│    ├── general_heuristic.py   # General heuristic solvers
│    ├── mmnl_heuristic.py      # Heuristic algorithms for MMNL
│    ├── nl_heuristic.py        # Heuristic algorithms for NL
│    ├── utils.py               # Utility functions for heuristics
│
│── models/                     # Functions for evaluating performance
│    ├── mmnl_functions.py      # MMNL-specific functions
│    ├── nl_functions.py        # NL-specific functions
│
│── src/                        # Entry points and core scripts
│    ├── mmnl_example.ipynb     # Jupyter notebook demonstrating MMNL assortment optimization.
│    ├── nl_example.ipynb      # Jupyter notebook demonstrating NL assortment optimization.
│
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation

```

---

## ⚙️ Installation

To get started with this project, follow these steps.

```bash
git clone https://github.com/wch444/Benchmark.git
cd Benchmark
```

For install dependencies, the project requires the following:
- Python (>=3.11.13)
- NumPy (>=2.3.2)
- Pandas (>=12.0.3)
- Torch (>=2.8.0)
- Gurobi (>=12.0.3)

You can use the requirements.txt files with pip to install a fully predetermined and working environment.
```bash
pip install -r requirements.txt
```
> **Note:** Gurobi must be installed separately. You can download it from [Gurobi's official website](https://www.gurobi.com/) and obtain a valid license for activation.

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
| `Conic_mmnl_warm_start` | An exact method to find the globally optimal assortment by formulating the problem as the conic integer formulation. [Şen et al.(2018)](#Şen2018)|
| `revenue_order` | A simple heuristic that sorts products by price and selects all products with revenue greater than a threshold. [Talluri et al.(2004)](#Talluri2004)|

The following method is used to solve the NL model, you can import all heuristics as:
```bash
from heuristic.general_heuristic import *
from heuristic.nl_heuristic import *
```
| Function| Description |
| ------------- | ---------------------------------------------- |
| `revenue_order_nl` | A algorithm based on linear programming, where each nest $i$ contains the $k_i$ highest-revenue products.[Davis et al.(2014)](#Davis2014)| 

---

## 🚀 How to use ?
The easiest way to get started is to run the example Jupyter notebooks located in the `src/` directory.

### 1. Mixed Multinomial Logit (MMNL) Model
There is a short example for assortment optimization under the MMNL, which is shown in [`src/mmnl_example.ipynb`](src/mmnl_example.ipynb)

This notebook will guide you through:
- Generating synthetic data for an MMNL model.
- Solving the unconstrained assortment problem using both the revenue-order heuristic and the exact conic method.
- Consider the cardinality constraint and resolve the problem.

---
### 2. Nested Logit (NL) Model
There is a short example for assortment optimization under the NL, which is shown in [`src/nl_example.ipynb`](src/nl_example.ipynb)

This notebook demonstrates assortment optimization for the NL model. It covers:
- Generating data for a multiple product "nests".
- Solving the unconstrained problem using the specialized revenue-ordered heuristic.
- Display the obtained assortment and corresponding revenue

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