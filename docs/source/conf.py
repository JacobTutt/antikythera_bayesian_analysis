# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Include your package directory

project = 'S2 Coursework'
copyright = '2025, Jacob Tutt'
author = 'Jacob Tutt'
release = '1.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax', 
]

autodoc_default_options = {
    'members': True,             # Include all public members
    'undoc-members': True,       # Include members without docstrings
    'private-members': True,     # Include members starting with _
    'special-members': '__init__',  # Include special methods (like __init__)
    'show-inheritance': True,    # Show class inheritance
    'alphabetical': False,       # To maintain source order (optional)
    'member-order': 'bysource',  # To maintain source order (optional)
}
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "seaborn",
    "iminuit",
    "sweights",
    "tqdm",
    "tabulate",
    "jax",
    "jaxlib",
    "numpyro",
    "optax",
    "arviz",
    "corner",
]
html_theme = 'sphinx_rtd_theme'

import sys
print("Python Path:", sys.path)
try:
    import calender_analysis
    print("calender_analysis imported successfully")
except ImportError:
    print("Failed to import alender_analysis")