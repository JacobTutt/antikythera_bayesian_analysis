# S2 Coursework Repository
This repository contains the package, its documentation, and implementation required for the coursework.
---

## Documentation for the project

[Documentation on Read the Docs](https://coursework-s2.readthedocs.io/en/latest/Calender_Analysis/index.html)

The coursework uses a modular, inherited class-based structure, which is explained below, to make it adaptable to different probability distributions. As a result documentation has been created for easier understanding of each functions methods and implementation


## Outline of Implementation

# mention filtering, model types and all implements in class automatically. 


## Installation Instructions

To run the notebooks, please follow these steps:

### 1. Clone the Repository

Clone the repository from the remote repository (GitLab) to your local machine.
Or your 
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/s2_coursework/jlt67.git
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
python -m ipykernel install --user --name=env --display-name "Python (S2 Coursework)"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel (Python (S2 Coursework)) to run the code.

## Report for the Coursework

The coureworks PDF report can be found under the **Report** directory of the repository


## Declaration of Use of Autogeneration Tools

This project made use of Large Language Models (LLMs), primarily ChatGPT and Co-Pilot, to assist in the development of the statistical analysis pipeline. These tools were utilized for:

- Generating detailed docstrings for the repository’s documentation.
- Formatting plots to enhance presentation quality.
- Performing iterative changes to already defined code.
- Debugging code and identifying issues in implementation.
- Assisting with LaTeX formatting for the report.
