# **Reverse Engineering the Antikythera Calendar: A Bayesian Perspective**

## Author: Jacob Tutt, Department of Physics, University of Cambridge

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/antikythera-bayesian-analysis/badge/?version=latest)](https://antikythera-bayesian-analysis.readthedocs.io/en/latest/?badge=latest)

## Description

This project applies a Bayesian Hamiltonian Monte Carlo analysis to infer the original structure of the fragmented Antikythera calendar ring, providing statistically robust estimates of its intended configuration.

## Table of Contents
- [Pipeline Functionalities](#calendar-analysis-pipeline-outline)
- [Notebooks](#notebooks)
- [Results](#results)
- [Documentation](#documentation)
- [Installation](#installation-and-usage)
- [License](#license)
- [Support](#support)
- [Author](#author)

## Calendar Analysis Pipeline Outline

This work present and self consistent [pipeline](calender_analysis/analysis.py) a flexible and modular framework for modelling the fragmented Antikythera calendar ring using Bayesian inference.

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

## Notebooks

The [notebooks](notebooks) in this repository serve as walkthroughs for the analysis performed. They include derivations of the mathematical implementations, explanations of key choices made, and present the main results. Five notebooks are provided:

| Notebook | Description |
|----------|-------------|
| [Notebook 1: Exploration](notebooks/Notebook_0_Exploration.ipynb) | Performs prelimary analysis on the Antikythera calendar images to provided more informed  constraint on the parameters priors. |
| [Notebook 2: Anisotropic Model](notebooks/Notebook_1_Anisotropic.ipynb) | Employs a rigurous bayesian analysis on the Antikythera Calender using an anisotropic model ($\sigma_r,\sigma_t$), while also providing a walkthrough of the pipeline and its functionalities. |
| [Notebook 3: Isotropic Model](notebooks/Notebook_2_Isotropic.ipynb) | An similiar analysis to that of notebook 2, however using an istropic model ($\sigma = \sigma_r = \sigma_t$) to compare results.|
| [Notebook 4: Comparison](notebooks/Notebook_3_Model_Comparison.ipynb) | Uses the bayesian evidence ratio to preform model comparison and evaluate what model (Isotropic or Anistropic) provides the best statistical fit to the data. This was achieved using both the Savage Dickey Ratio and a  custom implementation of Nested Sampling. |
| [Notebook 5: Extra Results](notebooks/Notebook_4_Extra.ipynb) | Explores the impact of different data filtering schemes on parameter inference. Tests over six combinations of model and dataset configurations using the established pipeline. |

## Results

To provide flexibility for the user, intermediate results for all six model–dataset configurations — including MCMC sampling outputs and model configuration optimisations — are precomputed and stored in the [Results](stored_results) directory. Using these can be easily toggled on and off using `rerun_comp_expensive_analysis` at the start of each.


## Documentation

For this project I have produced [documentation](https://antikythera-bayesian-analysis.readthedocs.io/en/latest/) for the pipeline to make it more accessible and easier to follow. Throughout the notebooks hyperlinks are provided to the relevent functions.

## Installation and Usage

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
Open the notebooks and select the created kernel `Python (Antikythera)` to run the code.

## For Assessment
- The associated project report can be found under [Project Report](report/Report.pdf). 

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the [LICENSE](LICENSE.txt) file for details.

## Support
If you have any questions, run into issues, or just want to discuss the project, feel free to:
- Open an issue on the [GitHub Issues](https://github.com/JacobTutt/antikythera_bayesian_analysis/issues) page.  
- Reach out to me directly via [email](mailto:jacobtutt@icloud.com).

