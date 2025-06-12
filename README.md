# Bayesian Analysis of the Antikythera Mechanism

This project applies a Bayesian Hamiltonian Monte Carlo analysis to infer the original structure of the fragmented Antikythera calendar ring, providing statistically robust estimates of its intended configuration.

---

## Calendar Analysis Pipeline Outline

Provides a flexible and modular framework for modelling the fragmented Antikythera calendar ring using Bayesian inference.

**1. Data Filtering**

* **Multiple Filtering Levels:** Includes `None`, `Basic`, and `Full` modes to accommodate varying degrees data filtering.
* **Automated Unreliable Section Removal:** Identifies and discards sections with few holes, to ensure data that will greater constrain global parameters.

**2. Error Model Selection**

* **Gaussian Error Models:** Allows switching between `isotropic` (uniform uncertainty) and `anisotropic` (direction-dependent uncertainty).
* **Isotropic Model**
* **Anisotropic Mode:** Separates uncertainty into `radial` (distance from center) and `tangential` (along the ring) components.


**3. Maximum Likelihood Estimation (MLE)**

* **Optimisation Algorithms:** Employs Stochastic Gradient Descent (`SGD`), Adam optimisation and Scipy minimisation with adjustable learning rates.
* **Multiple Initialisations:** Uses several starting points for parameter optimisation to avoid local minima.
* **Invalid Result Filtering:** Automatically discards parameter sets that yield non-physical or invalid results.

**4. Bayesian Inference via MCMC (NUTS)**

* **Sampling Algorithm:** Implements the No-U-Turn Sampler (`NUTS`), a form of Hamiltonian Monte Carlo.
* **Automatic Thinning:** Automatically prefroms thinning by fatoring in autocorrelation in the sampled parameter values by automatically discarding correlated samples.
* **Hyperparameter Optimisation:** Includes tools for tuning the hyperparameters of the MCMC sampler.
* **Convergence Diagnostics:** Provides metrics and visualisations to assess whether the MCMC chains have converged to the true posterior distribution.

**5. Model Comparison with Savage-Dickey Ratio**

* **Calculation:** Computes the Savage-Dickey ratio to estimate Bayes factors for comparing the nested models.
* **Density Estimation:** Utilises Kernel Density Estimation (`KDE`) to approximate the prior and posterior densities of the constrained parameter.

**6. Custom Nested Sampling**

* **Complete Nested Sampler:** Offers a full implementation of the Nested Sampling algorithm for robust Bayesian model comparison and parameter estimation.
* **Stable Evidence Calculation:** Computes Bayesian evidence in log-space to maintain numerical stability.
* **Prior Sampling:** Efficiently handles sampling from constrained prior distributions, ensuring physically relevant parameter exploration. The `ns_prior_transform` function maps unit cube samples to various prior types for robust use with for both scalar and section-based parameters.

---

## Documentation for the project

[Documentation on Read the Docs](https://antikythera-bayesian-analysis.readthedocs.io/en/latest/)

For this project I have produced documentation for the pipeline to make it more accessible and easier to follow. Throughout the notebooks hyperlinks are provided to the relevent functions.

---


## Installation Instructions

To run the notebooks, please follow these steps:

### 1. Clone the Repository

Clone the repository from the remote repository (GitLab) to your local machine.
Or your 
```bash
git clone https://github.com/JacobTutt/antikythera_bayesian_analysis.git
```

### 2. Create a Fresh Virtual Environment
Use a clean virtual environment to avoid dependency conflicts.
```bash
python -m venv env
source env/bin/activate   # For macOS/Linux
env\Scripts\activate      # For Windows
```

### 3. Install the Package and Dependencies
Navigate to the repository’s root directory and install the package along with its dependencies:
```bash
cd jlt67
pip install -e .
```

### 4. Set Up a Jupyter Notebook Kernel
To ensure the virtual environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Python (Antikythera)"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel (Python (Antikythera)) to run the code.

## Report

A report for this project can be found under the Report directory of the repository
