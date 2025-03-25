# Standard library imports
import os
import sys
import math
import time
import pickle
import logging
import itertools
import tracemalloc
import multiprocessing


# Scientific computing & data handling
import numpy as np
import pandas as pd
import scipy.stats
from iminuit import Minuit

# JAX
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)

# NumPyro
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist

# Optimization libraries
import optax

# Visualization
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import arviz as az
import corner

# Progress & display utilities
from tqdm.notebook import tqdm  
from IPython.display import display

 # Set Logging level - INFO and above
logging.basicConfig(level=logging.INFO,  format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


class Calender_Analysis:
    def __init__(self, data, model_type="anisotropic", filtering = 'None', priors=None, num_cores = 4):
        r"""
        Initializes the Bayesian model for analyzing hole positions on a fragmented calendar ring.

        This class models the hole positions using a probabilistic framework, supporting different
        error models, prior distributions, and dataset filtering options. It also enables parallel
        MCMC sampling using JAX across multiple CPU cores.

        Parameters
        ----------
        data : str, pd.DataFrame, or np.ndarray
            The observed hole position data. It can be provided as:
            - A **file path** (CSV, TXT, or DAT) containing the data.
            - A **Pandas DataFrame**.
            - A **NumPy array** (which is internally converted to a DataFrame).
        model_type : str, optional, default="anisotropic"
            Specifies the error model used:
            - `"isotropic"`: Uses a single `sigma` parameter for errors.
            - `"anisotropic"`: Uses separate `sigma_r` and `sigma_t` for radial and tangential errors.
        filtering : str, optional, default="None"
            Defines the filtering applied to the dataset:
            - `"None"`: No filtering.
            - `"Basic"`: Removes sections with only one hole.
            - `"Full"`: Removes sections with fewer than three holes and discards the first and last hole in each section.
        priors : dict, optional, default=None
            A dictionary specifying prior distributions for the model parameters. If `None`, default priors are used.
            The dictionary can include:
            - `"N"` (Total number of holes, default: `Uniform(340, 370)`)
            - `"r"` (Ring radius, default: `Uniform(65, 85)`)
            - `"x0", "y0"` (Section-specific x and y offsets, default: `Normal(80, 5)`, `Normal(135, 5)`)
            - `"alpha"` (Angular offsets, default: `Normal(-2.5, 1)`)
            - **Isotropic model:**
              - `"sigma"` (Shared error term, default: `Uniform(0, 5)`)
            - **Anisotropic model:**
              - `"sigma_r"` (Radial error term, default: `Uniform(0, 5)`)
              - `"sigma_t"` (Tangential error term, default: `Uniform(0, 5)`)
            If any user-defined priors are provided, they override the defaults.
        num_cores : int, optional, default=4
            The number of CPU cores to use for parallel processing with JAX during MCMC sampling.

        Raises
        ------
        ValueError
            If `model_type` or `filtering` contains an invalid value.
        RuntimeError
            If JAX fails to configure the requested number of CPU cores.
        """

        # Try to set JAX to use multiple devices for parallel chains for faster execution
        # If not possible, fall back to single-device execution
        try:
            num_devices = multiprocessing.cpu_count()
            logging.info(f"JAX Detected Devices: {num_devices}") 

            if num_devices < num_cores:
                numpyro.set_host_device_count(num_devices)
                logging.info(f"Only {num_devices} device(s) available. Running {num_devices} cores instead.")
            else:
                numpyro.set_host_device_count(num_cores)
                logging.info(f"JAX configured to use {num_cores} devices for parallel chains.")

        except Exception as e:
            logging.error(f"Error configuring JAX for parallel execution: {str(e)}")
            logging.info("Unable to set the number of host devices. Falling back to default single-device execution.")

        # Validate filtering type
        if filtering not in ["None", "Basic", "Full"]:
            raise ValueError("Invalid filtering type. Choose either 'None', 'Basic', or 'Full'.")
        # Validate model type
        if model_type not in ["anisotropic", "isotropic"]:
            raise ValueError("Invalid model_type. Choose either 'anisotropic' or 'isotropic'.")
        self.model_type = model_type

        # Import data and store with metadata
        self.data = self._load_data(data, filtering)
        self.model_type = model_type
        self.num_sections = len(self.data["Section ID"].unique())
        self.n_holes = len(self.data["Hole"].unique())

        #Import relevent data as class atributes store as jax arrays for efficient computation later
        self.section_ids_obs = jnp.array(self.data["Section ID"].values)
        self.hole_nos_obs = jnp.array(self.data["Hole"].values)
        self.x_obs = jnp.array(self.data["Mean(X)"].values)
        self.y_obs = jnp.array(self.data["Mean(Y)"].values)

        # Default priors if none provided - these can be individually overridden
        if model_type == "isotropic":
            default_priors = {
                "N": dist.Uniform(330, 370),
                "r": dist.Uniform(65, 90),
                "x0": dist.Uniform(60, 100),
                "y0": dist.Uniform(120, 160),
                "alpha": dist.Uniform(-2.8, -2.1),
                "sigma": dist.LogUniform(1e-5, 5),
            }
        else: # Anisotropic model - seperate sigma_r and sigma_t
            default_priors = {
                "N": dist.Uniform(330, 370),
                "r": dist.Uniform(65, 90),
                "x0": dist.Uniform(60, 100),
                "y0": dist.Uniform(120, 160),
                "alpha": dist.Uniform(-2.8, -2.1),
                "sigma_r": dist.LogUniform(1e-5, 5),
                "sigma_t": dist.LogUniform(1e-5, 5),
            }

        # Use user-defined priors if provided, otherwise use default priors
        self.priors = default_priors if priors is None else {**default_priors, **priors}


    
    def _load_data(self, data, filtering=False):
        r"""
        Loads, validates, and optionally filters the hole position data.

        This method ensures the data is correctly formatted and sorted by hole number.
        It supports multiple input formats (file, DataFrame, NumPy array) and applies optional
        filtering to clean the dataset.

        Parameters
        ----------
        data : str, pd.DataFrame, or np.ndarray
            The input dataset. It can be:
            - A file path (CSV, TXT, or DAT).
            - A Pandas DataFrame.
            - A NumPy array (converted to a DataFrame).
        filtering : str, optional, default="None"
            Determines how data is filtered:
            - "None": No filtering applied.
            - "Basic": Removes sections with only one hole.
            - "Full": Removes sections with fewer than three holes and discards the first and last hole in each section.

        Returns
        -------
        pd.DataFrame
            A cleaned and sorted DataFrame containing the hole position data.

        Raises
        ------
        ValueError
            If the required columns are missing or the file format is unsupported.
        TypeError
            If the input data format is not a string, DataFrame, or NumPy array.
        RuntimeError
            If an error occurs while loading the file.
        """
        # Required columns for the data and analysis
        required_columns = {"Section ID", "Hole", "Inter-hole Distance", "Mean(X)", "Mean(Y)"}
        
        # If the data input is already a DataFrame, simply accept
        if isinstance(data, pd.DataFrame):
            df = data
        
        # If the data input is a NumPy array, convert to DataFrame
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        
        # If the `data` input is a file path, try to load the data from the respective file type
        elif isinstance(data, str):
            # Determine the file type from the extension (.csv, .txt, .dat)
            file_ext = os.path.splitext(data.lower())[1]
            # Load the data from the file with corresponding file type
            try:
                if file_ext == ".csv":
                    df = pd.read_csv(data)
                elif file_ext in [".txt", ".dat"]:
                    df = pd.read_csv(data, delim_whitespace=True, header=0)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}. Supported types: .csv, .txt, .dat.")
            
            # Error incase the file is not found or cannot be loaded
            except Exception as e:
                raise RuntimeError(f"Error loading data: {e}")
        
        # Error in case the data format is not supported
        else:
            raise TypeError("Unsupported data format. Use a file path (CSV, TXT), Pandas DataFrame, or NumPy array.")
        
        # Ensure that the required columns (defined above) are present in the DataFrame
        if not required_columns.issubset(df.columns):
            raise ValueError(f"File must contain columns: {required_columns}")
        
        # Sort data by hole number for proper line connections
        df = df.sort_values(by="Hole").reset_index(drop=True)

        # -------------------- Apply Filtering (If Enabled) --------------------
        if filtering in ["Basic", "Full"]:
            # Count the number of holes per section
            section_counts = df["Section ID"].value_counts()

            # Identify valid sections (Basic: >1 holes, Full: >3 holes)
            min_holes_required = 1 if filtering == "Basic" else 3
            valid_sections = section_counts[section_counts > min_holes_required].index

            # Filter out sections that do not meet the hole count requirement
            df = df[df["Section ID"].isin(valid_sections)]

            # If using 'Full' filtering, remove first and last hole of each valid section
            if filtering == "Full":
                # Identify first and last hole per section
                first_last_holes = df.groupby("Section ID")["Hole"].agg(["min", "max"]).reset_index()

                # Remove first and last holes of each section using Masking
                mask = ~df.apply(lambda row: row["Hole"] in first_last_holes.loc[first_last_holes["Section ID"] == row["Section ID"], ["min", "max"]].values, axis=1)
                df = df[mask]

            # Renumber sections sequentially - this removes unecessary gaps and stops constant calculation of unpopulated sections
            section_mapping = {old: new + 1 for new, old in enumerate(sorted(valid_sections))}
            df["Section ID"] = df["Section ID"].map(section_mapping)

        # Sort data by hole number for proper ordering after filtering
        df = df.sort_values(by=["Section ID", "Hole"]).reset_index(drop=True)

        # -------------------- Print summary of imported data --------------------
        # In a data frame display hole range for each section
        section_summary = df.groupby("Section ID")["Hole"].agg(["min", "max"])
        section_summary["Hole Range"] = section_summary["min"].astype(str) + " - " + section_summary["max"].astype(str)
        section_summary_str = "\n".join(
            f"      {section:<7}|   {row['Hole Range']}" 
            for section, row in section_summary.iterrows())

        logging.info(f" \n===============================" + 
                     "\n        DATA SUMMARY"+
                     "\n===============================" +
                     f"\nTotal Sections   : {df['Section ID'].nunique()}" +
                     f"\nTotal Holes      : {df['Hole'].nunique()}" +
                     "\n---------------------------------" +
                     "\nSection ID   |   Hole Range " +
                     "\n--------------------------------- " +
                     f"\n{section_summary_str} " +
                     "\n--------------------------------- ")

        return df
        
    
    def plot_hole_locations(self, figsize=(10, 8)):
        r"""
        Plots the measured hole locations in the x-y plane, color-coded by section ID.

        This function provides a visualization of hole positions (`Mean(X)`, `Mean(Y)`) from the dataset.
        It uses color-coded markers for different sections, adds annotations for hole numbers at regular
        intervals, and marks section transitions with perpendicular bisecting lines.

        Features
        --------
        - **Color-coded markers**: Each section ID is assigned a unique color.
        - **Different marker styles**: Cycles through circle, square, and triangle markers.
        - **Annotations**: Labels every third hole with its hole number.
        - **Section transition markers**: Red dashed perpendicular bisecting lines indicate section changes.
        - **Multiple legends**:
          - Section ID legend.
          - Bisector legend listing splits with corresponding hole numbers.

        Parameters
        ----------
        figsize : tuple, optional, default=(10, 8)
            The figure size in inches.

        Returns
        -------
        None
            Displays the plot.
        """
        # Initialise the plot with configuration
        plt.figure(figsize=figsize)
        sns.set_context("talk")
        sns.set_style("whitegrid")
        plt.rcParams["font.family"] = "serif"

        # Define a color mapping mapping each unique sections to a colour
        unique_sections = self.data["Section ID"].unique()
        # Seaborns high contrast colors
        palette = sns.color_palette("Dark2", len(unique_sections))
        color_map = {section: palette[i] for i, section in enumerate(unique_sections)}

        # Define three marker styles (circle, square, triangle)
        marker_styles = ['o', 's', '^'] 

        # Plot each hole with color and marker style based on section
        for i, section in enumerate(unique_sections):
            subset = self.data[self.data["Section ID"] == section]
            plt.scatter(subset["Mean(X)"], subset["Mean(Y)"], 
                        color=color_map[section], label=f"Section {section}", 
                        s=80, edgecolors='black', alpha=0.6, marker=marker_styles[i % len(marker_styles)])
        
        # Annotate every 3rd hole starting from hole 1, adjusting position to optimise visibility
        for i in range(self.n_holes):
            hole_num_int = int(self.data.iloc[i]["Hole"])
            if (hole_num_int - 1) % 3 == 0:
                x, y = self.data.iloc[i]["Mean(X)"], self.data.iloc[i]["Mean(Y)"]
                # Define offsets for hole number labels based on hole number range
                offsets = {
                    hole_num_int < 29: (-3, 0, 'right', 'center'),
                    28 < hole_num_int < 46: (-4, 0, 'center', 'top'),
                    45 < hole_num_int < 69: (0, -1, 'center', 'top'),
                    hole_num_int >= 69: (5, 0, 'left', 'center')
                }
                # Apply the offsets using tuple unpacking
                dx, dy, ha, va = offsets[True]
                plt.text(x + dx, y + dy, str(hole_num_int), fontsize=12, ha=ha, va=va,
                         color='black')

        # Draw and label perpendicular bisecting lines at section transitions
        bisector_legend_handles = []
        bisector_index = 1
        for i in range(1, self.n_holes):
            prev_section = self.data.iloc[i - 1]["Section ID"]
            curr_section = self.data.iloc[i]["Section ID"]
            if prev_section != curr_section:
                x1, y1 = self.data.iloc[i - 1]["Mean(X)"], self.data.iloc[i - 1]["Mean(Y)"]
                x2, y2 = self.data.iloc[i]["Mean(X)"], self.data.iloc[i]["Mean(Y)"]
                
                # Midpoint between transition points
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Compute perpendicular bisector vector 
                # Scale for visualization
                scale = 4
                # First - determine normlaised direction vector between points
                dx, dy = x2 - x1, y2 - y1
                norm = np.sqrt(dx**2 + dy**2)
                dx, dy = dx / norm, dy / norm
                # Second - Rotate 90 degrees to get perpendicular vector
                px, py = -dy, dx

                plt.plot([mx - px * scale, mx + px * scale], [my - py * scale, my + py * scale],
                         color='red', linewidth=2, linestyle='--')
                
                # Label bisector with index number - adjust position for visibility
                if bisector_index == 4:
                    x_pos_label, y_pos_label = mx + px * scale-0.5, my + py * scale
                elif bisector_index == 5:
                    x_pos_label, y_pos_label = mx + px * scale+0.5, my + py * scale
                else:
                    x_pos_label, y_pos_label = mx + px * scale, my + py * scale
                
                plt.text(x_pos_label, y_pos_label, str(bisector_index), fontsize=12, ha='center', va='bottom', color='red')
                bisector_legend_handles.append(mlines.Line2D([], [], color='red', linestyle='--', linewidth=2, label=f"Split {bisector_index}: Holes {self.data.iloc[i-1]['Hole']} - {self.data.iloc[i]['Hole']}"))
                bisector_index += 1
        
        # Plot two legends - one for section ID and one for bisector labels
        # Legend 1 - Section ID (maybe removed later)
        legend = plt.legend(title="Section ID", loc="upper right", fontsize=12, frameon=True)
        legend.get_frame().set_alpha(0.8)
        # Legend 2 - Bisector Labels
        plt.legend(handles=bisector_legend_handles, loc="upper center", fontsize=10, frameon=True, title="Split Locations")
        plt.gca().add_artist(legend)
        
        # Plot configuration
        x_min, x_max = self.data["Mean(X)"].min(), self.data["Mean(X)"].max()
        y_min, y_max = self.data["Mean(Y)"].min(), self.data["Mean(Y)"].max()
        plt.xlim(x_min - 5, x_max + 5.5)
        plt.ylim(y_min - 4, y_max + 4)
        plt.xlabel("Mean X Position (mm)", fontsize=14)
        plt.ylabel("Mean Y Position (mm)", fontsize=14)
        plt.title("Measured Hole Locations in the x-y Plane", fontsize=16)
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
        plt.minorticks_on()

        plt.show()

    def hole_positions(self, N, r, x0, y0, alpha, section_ids=None, hole_nos=None):
        r"""
        Computes the expected positions of holes on the calendar ring based on model parameters.

        This function calculates the predicted x and y coordinates of holes under the assumption that:
        - The ring is circular with radius `r`.
        - The ring originally had `N` evenly spaced holes before fragmentation.
        - Each section has a relative center `(x0, y0)` and the 0th hole occours at a rotation `alpha`.

        The function can process a single hole or multiple holes at once.

        Parameters
        ----------
        N : float
            The total number of holes in the full calendar ring before fragmentation.
        r : float
            The estimated radius of the ring.
        x0 : jnp.ndarray
            The x-offsets for each section.
        y0 : jnp.ndarray
            The y-offsets for each section.
        alpha : jnp.ndarray
            The angular offsets for each section.
        section_ids : int, list, or jnp.ndarray, optional
            The section IDs corresponding to each hole being modeled.
            If `None`, it defaults to the observed dataset values.
        hole_nos : int, list, or jnp.ndarray, optional
            The hole numbers corresponding to each hole being modeled.
            If `None`, it defaults to the observed dataset values.

        Returns
        -------
        jnp.ndarray
            A 2D array of shape `(N_holes, 2)`, where each row contains the expected (x, y) position
            of a hole based on the model parameters.

        Raises
        ------
        ValueError
            - If `section_ids` and `hole_nos` do not have the same length.
            - If `N` is not a positive number.
            - If `r` is not a positive number.
            - If `x0`, `y0`, and `alpha` do not match the expected number of sections.
        """

        # ---- Input Validation ---- #

        # If section_ids or hole_nos are not provided, use the stored dataset values
        if section_ids is None:
            section_ids = self.section_ids_obs
        section_ids = jnp.atleast_1d(jnp.array(section_ids))  # Convert to JAX array, ensure 1D

        if hole_nos is None:
            hole_nos = self.hole_nos_obs
        hole_nos = jnp.atleast_1d(jnp.array(hole_nos))  # Convert to JAX array, ensure 1D

        # Ensure section_ids and hole_nos have matching shapes
        if section_ids.shape != hole_nos.shape:
            raise ValueError(f"section_ids and hole_nos must have the same shape, got {section_ids.shape} and {hole_nos.shape}.")

        # Validate N is a positive integer
        if not isinstance(N, (float, int, np.floating, jnp.floating)) or N <= 0:
            raise ValueError(f"N must be a positive value, got {type(N)}.")
       

        # Validate r is a positive float
        if not isinstance(r, (float, int, np.floating, jnp.floating)) or r <= 0:

            raise ValueError(f"r must be a positive float, got {type(r)}.")

        # Ensure x0, y0, and alpha match the number of sections
        if x0.shape[0] != self.num_sections or y0.shape[0] != self.num_sections or alpha.shape[0] != self.num_sections:
            raise ValueError(f"x0, y0, and alpha must have length {self.num_sections}, got {x0.shape[0]}, {y0.shape[0]}, and {alpha.shape[0]}.")

        # ---- Compute Expected Positions ---- #
        # The phi value of each hole is calculated based on the hole number and the section it belongs to
        # Hole value - angular position in the ring - from fraction of 2 pi based on hole number/ N 
        # Section value - using alpha gives relative rotation of each section/ angular offset
        # List of phi values for each hole
        phi = (2 * jnp.pi * (hole_nos - 1) / N) + alpha[section_ids - 1]

        # Expected x and y positions of the hole based on the model - using r, phi, x0, y0
        # List of modeled x and y values for each hole
        x_model = r * jnp.cos(phi) + x0[section_ids - 1]
        y_model = r * jnp.sin(phi) + y0[section_ids - 1]

        # Stack x and y values to get a 2D array of hole positions
        hole_posn_model = jnp.column_stack((x_model, y_model))

        return hole_posn_model

    def likelihood(self, N, r, x0, y0, alpha, sigma, log=False, neg = False, data = None):
        r"""
        Computes the likelihood (or log-likelihood) of the observed hole positions given model parameters.

        This function evaluates how well the observed hole positions match the expected positions
        under a Gaussian error model. It supports two types of likelihoods:

        - **Isotropic Gaussian**: A single standard deviation `sigma` applies to both x and y errors.
        - **Anisotropic Gaussian**: Separate standard deviations `sigma_r` and `sigma_t` for radial
          and tangential errors.

        The likelihood is calculated as:

        - **Isotropic Model**:

        .. math::
            \log L = -\frac{1}{2} \sum_i \left(\frac{(e_{i,x})^2 + (e_{i,y})^2}{\sigma^2} \right)- n \log(2\pi\sigma)

        - **Anisotropic Model**:

        .. math::
            \log L = -\frac{1}{2} \sum_i \left( \frac{(e_{i,r})^2}{\sigma_r^2} + \frac{(e_{i,t})^2}{\sigma_t^2} \right) - n \log(2\pi\sigma_r\sigma_t)

        where:

        - :math:`e_{i,x} = x_{\text{obs},i} - x_{\text{model},i}` (x-coordinate error)
        - :math:`e_{i,y} = y_{\text{obs},i} - y_{\text{model},i}` (y-coordinate error)
        - :math:`e_{i,r}` and :math:`e_{i,t}` are the radial and tangential errors.

        If a subset of data (`data`) is provided, the likelihood is computed over only that subset.
        This enables efficient **stochastic gradient descent (SGD)** or **mini-batch optimization**.

        Parameters
        ----------
        N : float
            The total number of holes in the original (pre-fragmented) calendar ring.
        r : float
            The estimated radius of the ring.
        x0 : jnp.ndarray
            Array of **x-offsets** for each section.
        y0 : jnp.ndarray
            Array of **y-offsets** for each section.
        alpha : jnp.ndarray
            Array of **angular offsets** (in radians) for each section.
        sigma : float or jnp.ndarray
            - If **isotropic**, a single float `sigma` is used for both x and y errors.
            - If **anisotropic**, a JAX array `[sigma_r, sigma_t]` is used for radial and tangential errors.
        log : bool, optional, default=False
            - If `True`, returns the **log-likelihood**.
            - If `False`, returns the **likelihood**.
        neg : bool, optional, default=False
            - If `True`, returns the **negative log-likelihood** (useful for optimization).
            - Only valid when `log=True`.
        data : pd.DataFrame, optional, default=None
            A subset of the dataset to compute likelihood on. If `None`, the full dataset is used.
            This enables **stochastic optimization** by computing likelihood using **mini-batches**.

        Returns
        -------
        jnp.ndarray
            - If `log=True, neg=False`: Returns the **log-likelihood**.
            - If `log=True, neg=True`: Returns the **negative log-likelihood**.
            - If `log=False`: Returns the **likelihood** (exponentiated log-likelihood).

        Raises
        ------
        ValueError
            - If `neg=True` but `log=False`, as negative likelihood is only defined for log space.
        """

        # This implements a similiar analysis to `hole_positions` 
        # Goes on to evaluate discrepancy between model and observed data and calculate log likelihood


        # Compute the expected hole positions based on the model parameters
        # It does no not use the `hole_positions` function but instead calculates the modelled hole positions directly 
        # Stops type checking inputs from slowing preformance when many calls are made

        if neg and not log:
            raise ValueError("Negative is only valid when log is True - ie negetive log likelihood")

        # ------------------------- Import data or use full dataset -------------------------
        if data is None:
            section_ids = self.section_ids_obs
            hole_nos = self.hole_nos_obs
            x_pos = self.x_obs
            y_pos = self.y_obs
        else:
            section_ids = data["Section ID"].values
            hole_nos = data["Hole"].values
            x_pos = data["Mean(X)"].values
            y_pos = data["Mean(Y)"].values
            

        # ------------------------- Repeat of code from `hole_positions` -------------------------
        
        # Compute expected positions of model from parameters for each hole/ data point
        # The phi value of each hole is calculated based on the hole number anf the section it belongs to
        # Hole value - anguluar position in the ring - from fraction of 2 pi based on hole number/ N 
        # Section value - using alpha gives relative rotation of each section/ anguluar offset
        # List of phi values for each hole
        phi = (2 * jnp.pi * (hole_nos - 1) / N) + alpha[section_ids - 1]
        # Expected x and y positions of the hole based on the model - using r, phi, x0, y0
        # List of modelled x and y values of for each hole
        x_model = r * jnp.cos(phi) + x0[section_ids - 1]
        y_model = r * jnp.sin(phi) + y0[section_ids - 1]

        # -----------------------------------------------------------------------------------------

        # Compute error vectors in cartesian coordinates
        # List of error vectors for each hole - difference between observed and modelled x and y values
        error_x = x_pos - x_model
        error_y = y_pos - y_model

        # # Store deterministic transformed errors (optional)
        # error_vectors = jnp.stack([error_r, error_t], axis=1)
        # numpyro.deterministic("error_vectors", error_vectors)


        # Define likelihood based on error model
        # Plate can be used as independent random variables
        # Allows the likelihood to be calculated in parallel for each hole as they are independent
        # Each normal likelihood is conditioned on the error_r and error_t values and total likelihood is product of all likelihoods

        ### --- Isotropic Model --- ###
        if self.model_type == "isotropic":
            sigma_val = sigma
            # No need for translation to radial and tangential coordinates as isotropic model - reduce computation

            log_likelihoods_variable = (-0.5 * (error_x**2 + error_y**2) / sigma_val**2)
            
            log_likelihoods_constant =  log_likelihoods_variable.size * jnp.log(2 * jnp.pi * sigma_val**2)

        ### --- Anisotropic Model --- ###
        elif self.model_type == "anisotropic":
             # Translate error vectors to radial and tangential coordinates
            # For each hole, determine unit radial and tangential vectors - ie transform error vectors x and y to radial and tangential components
            unit_r = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=1)
            unit_t = jnp.stack([jnp.sin(phi), -jnp.cos(phi)], axis=1)

            # List of radial and tangential error values for each hole
            error_r = error_x * unit_r[:, 0] + error_y * unit_r[:, 1]
            error_t  = error_x * unit_t[:, 0] + error_y * unit_t[:, 1]

            # Compute anisotropic Gaussian likelihood for each hole
            log_likelihoods_variable = (-0.5 * ((error_r**2 / sigma[0]**2) + (error_t**2 / sigma[1]**2)))

            # Compute constant term once for efficiency - no. holes * log(const)
            log_likelihoods_constant = log_likelihoods_variable.size * jnp.log(2 * jnp.pi * sigma[0] * sigma[1])
        
        # If Log = True - Return log-likelihood 
        if log:
            if neg: 
                return - jnp.sum(log_likelihoods_variable) + log_likelihoods_constant
            else:
                return jnp.sum(log_likelihoods_variable) - log_likelihoods_constant
            
        
        # If Log = False - Return likelihood
        else: 
            return jnp.exp(jnp.sum(log_likelihoods_variable) - log_likelihoods_constant)
        
        
    def grad_likelihood(self, N, r, x0, y0, alpha, sigma, log=True, neg = False, data = None):
        r"""
        Computes the gradients of the likelihood or log-likelihood using automatic differentiation.

        This function leverages **JAX's automatic differentiation (`jax.grad`)** to compute the gradients
        of the **likelihood** or **log-likelihood** with respect to all model parameters.

        - If `log=True`, computes gradients of **log-likelihood**:

          .. math::
             \frac{\partial \log L}{\partial \theta}

        - If `log=False`, computes gradients of **likelihood**:

          .. math::
             \frac{\partial L}{\partial \theta}

        - If `neg=True`, returns **negative gradients** for optimization (minimization of negative log-likelihood).

        Parameters
        ----------
        N : float
            Total number of holes in the original (pre-fragmented) calendar ring.
        r : float
            Estimated radius of the ring.
        x0 : jnp.ndarray
            Array of **x-offsets** for each section (shape: `num_sections`).
        y0 : jnp.ndarray
            Array of **y-offsets** for each section (shape: `num_sections`).
        alpha : jnp.ndarray
            Array of **angular offsets** (in radians) for each section (shape: `num_sections`).
        sigma : float or jnp.ndarray
            - If **isotropic**, a single float `sigma` (shared for x and y errors).
            - If **anisotropic**, a JAX array `[sigma_r, sigma_t]` for radial and tangential errors.
        log : bool, optional, default=True
            - If `True`, computes gradients of the **log-likelihood**.
            - If `False`, computes gradients of the **likelihood**.
        neg : bool, optional, default=False
            - If `True`, computes gradients of the **negative log-likelihood** (useful for optimization).
        data : pd.DataFrame, optional, default=None
            A subset of the dataset for computing gradients. If `None`, uses the full dataset.

        Returns
        -------
        dict
            A dictionary containing gradients of the likelihood/log-likelihood with respect to each parameter:
            - `"N"` : Gradient w.r.t. total number of holes.
            - `"r"` : Gradient w.r.t. radius of the ring.
            - `"x0"` : Gradients w.r.t. x-offsets (array of shape `num_sections`).
            - `"y0"` : Gradients w.r.t. y-offsets (array of shape `num_sections`).
            - `"alpha"` : Gradients w.r.t. angular offsets (array of shape `num_sections`).
            - `"sigma"` : Gradient w.r.t. sigma (scalar for isotropic, array for anisotropic).

        Notes
        -----
        - Uses **JAX's automatic differentiation** (`jax.grad`) for gradient computation.
        - More efficient than finite-difference approximations but may be **less stable** than analytical gradients.
        - If `neg=True`, returns **negative gradients** for optimization-based approaches (e.g., gradient descent).
        """
        if neg and not log:
            raise ValueError("Negative is only valid when log is True - ie negetive log likelihood")

        # Compute gradients using JAX's automatic differentiation tool
        loss_grad_fn = jax.grad(self.likelihood, argnums=(0, 1, 2, 3, 4, 5))
        gradients = loss_grad_fn(N, r, x0, y0, alpha, sigma, log, neg, data)

        # Create a dictionary of gradients
        grad_dict = {
            "N": gradients[0],
            "r": gradients[1],
            "x0": gradients[2],  # Vector of gradients for each section
            "y0": gradients[3],  # Vector of gradients for each section
            "alpha": gradients[4],  # Vector of gradients for each section
            "sigma": gradients[5]  # Scalar (isotropic) or tuple (anisotropic)
        }

        return grad_dict
    
    def analytic_grad_loglikelihood(self, N, r, x0, y0, alpha, sigma, neg = False, data = None):
        r"""
        Computes the **analytical gradients** of the log-likelihood function.

        This function explicitly derives the **partial derivatives** of the **log-likelihood** function
        with respect to each model parameter using the **chain rule**. It is more stable and efficient
        than automatic differentiation (e.g., `jax.grad`).

        **Log-likelihood expressions**

        - **Isotropic Gaussian Model** (single `sigma` for x and y errors):

          .. math::

             \log L = -\frac{1}{2\sigma^2} \sum_{i=1}^{n} \left( e_{x,i}^2 + e_{y,i}^2 \right) - n \log(2\pi \sigma)

        - **Anisotropic Gaussian Model** (separate `sigma_r`, `sigma_t` for radial and tangential errors):

          .. math::

             \log L = -\frac{1}{2} \sum_{i=1}^{n} \left( \frac{e_{r,i}^2}{\sigma_r^2} + \frac{e_{t,i}^2}{\sigma_t^2} \right) - n \log(2\pi \sigma_r \sigma_t)

        Parameters
        ----------
        N : float
            Total number of holes in the original (pre-fragmented) circular ring.
        r : float
            Estimated radius of the ring.
        x0 : jnp.ndarray
            x-offsets for each section (shape: `num_sections`).
        y0 : jnp.ndarray
            y-offsets for each section (shape: `num_sections`).
        alpha : jnp.ndarray
            Angular offsets for each section (shape: `num_sections`).
        sigma : float or jnp.ndarray
            - If **isotropic**, a single float `sigma` is used for both x and y errors.
            - If **anisotropic**, a JAX array `[sigma_r, sigma_t]` is used for radial and tangential errors.
        neg : bool, optional, default=False
            - If `True`, returns **negative gradients** for optimization (gradient ascent).
        data : pd.DataFrame, optional, default=None
            A subset of the dataset to compute gradients on. If `None`, uses the full dataset.

        Returns
        -------
        dict
            A dictionary containing **analytical gradients** of the log-likelihood:
            - `"N"` : Gradient w.r.t. total number of holes.
            - `"r"` : Gradient w.r.t. radius of the ring.
            - `"x0"` : Gradients w.r.t. x-offsets (shape: `num_sections`).
            - `"y0"` : Gradients w.r.t. y-offsets (shape: `num_sections`).
            - `"alpha"` : Gradients w.r.t. angular offsets (shape: `num_sections`).
            - `"sigma"` : Gradient w.r.t. sigma (scalar for isotropic, array for anisotropic).

        Notes
        -----
        - Uses **explicit derivatives** instead of automatic differentiation.
        - More stable than `jax.grad()`.
        - If `neg=True`, returns **negative gradients** (useful for optimization problems).
        """
        # Uses the chain rule to determine the analytical gradients of the log likelihood
        # For each item of the graphical model we word out gradients of next dependant variable
        # Total derivatives will be product of all partial derivatives

        # ------------------------- Import data or use full dataset -------------------------
        if data is None:
            section_ids = self.section_ids_obs
            hole_nos = self.hole_nos_obs
            x_pos = self.x_obs
            y_pos = self.y_obs

        else:
            section_ids = data["Section ID"].values
            hole_nos = data["Hole"].values
            x_pos = data["Mean(X)"].values
            y_pos = data["Mean(Y)"].values


        ## ------------------ Compute intermediate values ------------------

        # phi: functions on N, alpha
        phi = (2 * jnp.pi * (hole_nos - 1) / N) + alpha[section_ids - 1] # - (n_holes)

        # x_model: function of r, phi, x0
        x_model = r * jnp.cos(phi) + x0[section_ids - 1] # - (n_holes)

        # y_model: function of r, phi, y0
        y_model = r * jnp.sin(phi) + y0[section_ids - 1] # - (n_holes)

        # Error terms: function of modeled positions (x_model, y_model)
        error_x = x_pos - x_model  # - (n_holes)
        error_y = y_pos - y_model  # - (n_holes)

        ## ------------------ Compute derivatives using chain rule ------------------

        # Partial derivatives of phi
        grad_phi_N = -2 * jnp.pi * (hole_nos - 1) / N**2  # dphi/dN - (n_holes)
        grad_phi_alpha = jnp.eye(self.num_sections)[section_ids - 1].T # dphi/dalpha - one-hot encoded matrix of shape (num_sections, n_holes) -- identity vector where 1 at section index, 0 elsewhere

        # Partial derivatives of x_model
        grad_x_model_r = jnp.cos(phi)  # dx_model/dr - (n_holes)
        grad_x_model_x0 = jnp.eye(self.num_sections)[section_ids - 1].T  # dx_model/dx0 - (num_sections, n_holes) 
        # grad_x_model_y0 = 0 
        grad_x_model_phi = -r * jnp.sin(phi)  # dx_model/dphi - (n_holes)
        grad_x_model_N = grad_x_model_phi * grad_phi_N  # dx_model/dN - (n_holes)
        grad_x_model_alpha = grad_phi_alpha * grad_x_model_phi  # dx_model/dalpha - (num_sections, n_holes) 

        # Partial derivatives of y_model
        grad_y_model_r = jnp.sin(phi)  # dy_model/dr
        # grad_y_model_x0 = 0
        grad_y_model_y0 = jnp.eye(self.num_sections)[section_ids - 1].T  # dy_model/dy0 - (num_sections, n_holes) 
        grad_y_model_phi = r * jnp.cos(phi)  # dy_model/dphi - (n_holes)
        grad_y_model_N = grad_y_model_phi * grad_phi_N  # dy_model/dN - (n_holes)
        grad_y_model_alpha = grad_y_model_phi * grad_phi_alpha  # dy_model/dalpha  - (num_sections, n_holes) 

        # Partial derivatives of error terms
        grad_error_x_x_model = -1  # de_x/dx_model - (1)
        grad_error_y_y_model = -1  # de_y/dy_model - (1)

        grad_error_x_r = grad_error_x_x_model * grad_x_model_r  # de_x/dr - (n_holes)
        grad_error_x_x0 = grad_error_x_x_model * grad_x_model_x0   # de_x/dx0 - (num_sections, n_holes) 
        # grad_error_x_y0 = 0  # de_x/dy0 - (num_sections, n_holes)
        grad_error_x_N = grad_error_x_x_model * grad_x_model_N # de_x/dN - (n_holes)
        grad_error_x_alpha = grad_error_x_x_model * grad_x_model_alpha # de_x/dalpha - (num_sections, n_holes) 

        grad_error_y_r = grad_error_y_y_model * grad_y_model_r  # de_y/dr
        # grad_error_y_x0 = 0  # de_y/dx0 - (num_sections, n_holes)
        grad_error_y_y0 = grad_error_y_y_model * grad_y_model_y0  # de_y/dy0 - (num_sections, n_holes) 
        grad_error_y_N = grad_error_y_y_model * grad_y_model_N  # de_y/dN - (n_holes)
        grad_error_y_alpha = grad_error_y_y_model * grad_y_model_alpha  # de_y/dalpha - (num_sections, n_holes) 


        ## ------------------ Compute log-likelihood gradients ------------------

        if self.model_type == "isotropic":
            # Isotropic: single sigma for x and y errors
            sigma_val = sigma  # Since sigma is a single float

            grad_N = - jnp.sum((error_x * grad_error_x_N + error_y * grad_error_y_N) / sigma_val**2)
            grad_r = - jnp.sum((error_x * grad_error_x_r + error_y * grad_error_y_r) / sigma_val**2)
            grad_sigma = + jnp.sum((error_x**2 + error_y**2) / sigma_val**3) - 2*len(error_x) / sigma_val

            # Sum of errors for each section - does the following but seperately for each section - as all data points outside of this section go to zero
            grad_x0 = - jnp.sum((error_x * grad_error_x_x0 / sigma_val**2), axis=1) # (num_sections)
            grad_y0 = - jnp.sum((error_y * grad_error_y_y0 / sigma_val**2), axis=1)  # (num_sections) 
            grad_alpha = -jnp.sum((error_x * grad_error_x_alpha + error_y * grad_error_y_alpha) / sigma_val**2, axis = 1)  # (num_sections)
            
            grad_dict = {
                "N": grad_N,
                "r": grad_r,
                "x0": grad_x0,
                "y0": grad_y0,
                "alpha": grad_alpha,
                "sigma": grad_sigma
            }

            # If neg = True, return negative gradients for optimisation
            if neg:
                return jtu.tree_map(lambda x: -x, grad_dict)

            return grad_dict

        elif self.model_type == "anisotropic":
            # Extract sigma values: radial and tangential
            sigma_r, sigma_t = sigma[0], sigma[1]

            ## ------------------ Further intermediate values for anisotropic models ------------------

            # Compute unit radial and tangential vectors
            unit_r = jnp.array([jnp.cos(phi), jnp.sin(phi)])  # (2, n_holes)
            unit_t = jnp.array([jnp.sin(phi), -jnp.cos(phi)])  # (2, n_holes)

            # Compute radial and tangential errors
            error_r = error_x * unit_r[0] + error_y * unit_r[1]
            error_t = error_x * unit_t[0] + error_y * unit_t[1]

            ## ------------------ Further derivatives using chain rule ------------------

            # Compute derivatives of unit vectors
            grad_unit_r_phi = - unit_t  # du_r / dphi = u_t - (n_holes, 2)
            grad_unit_t_phi = unit_r  # du_t / dphi = -u_r - (n_holes, 2)

            grad_unit_r0_alpha = grad_unit_r_phi[0] * grad_phi_alpha  # du_r / dalpha -  (num_sections, n_holes)
            grad_unit_r1_alpha = grad_unit_r_phi[1] * grad_phi_alpha  # du_t / dalpha -  (num_sections, n_holes)

            grad_unit_t0_alpha = grad_unit_t_phi[0] * grad_phi_alpha  # du_t / dalpha -  (num_sections, n_holes)
            grad_unit_t1_alpha = grad_unit_t_phi[1] * grad_phi_alpha  # du_t / dalpha -  (num_sections, n_holes)

            grad_unit_r0_N = grad_unit_r_phi[0] * grad_phi_N  # du_r / dN -  (n_holes)
            grad_unit_r1_N = grad_unit_r_phi[1] * grad_phi_N  # du_t / dN -  (n_holes)

            grad_unit_t0_N = grad_unit_t_phi[0] * grad_phi_N  # du_t / dN -  (n_holes)
            grad_unit_t1_N = grad_unit_t_phi[1] * grad_phi_N  # du_t / dN -  (n_holes)

            grad_error_r_N = grad_error_x_N * unit_r[0] + error_x * grad_unit_r0_N + grad_error_y_N * unit_r[1] + error_y * grad_unit_r1_N  # de_r / dN -  (n_holes)
            grad_error_r_alpha = grad_error_x_alpha * unit_r[0] + error_x * grad_unit_r0_alpha + grad_error_y_alpha * unit_r[1] + error_y * grad_unit_r1_alpha  # de_r / dalpha -  (num_sections, n_holes)
            grad_error_r_x0 = grad_error_x_x0 * unit_r[0] # (+ grad_error_y_x0 * unit_r[1] = 0)  # de_r / dx0 -  (num_sections, n_holes)
            grad_error_r_y0 = grad_error_y_y0 * unit_r[1] # (+ grad_error_x_y0 * unit_r[0] = 0)# de_r / dy0 -  (num_sections, n_holes)
            grad_error_r_r = grad_error_x_r * unit_r[0] + grad_error_y_r * unit_r[1]  # de_r / dr -  (n_holes)

            grad_error_t_N = grad_error_x_N * unit_t[0] + error_x * grad_unit_t0_N + grad_error_y_N * unit_t[1] + error_y * grad_unit_t1_N  # de_t / dN -  (n_holes)
            grad_error_t_alpha = grad_error_x_alpha * unit_t[0] + error_x * grad_unit_t0_alpha + grad_error_y_alpha * unit_t[1] + error_y * grad_unit_t1_alpha  # de_t / dalpha -  (num_sections, n_holes)
            grad_error_t_x0 = grad_error_x_x0 * unit_t[0] # (+ grad_error_y_x0 * unit_t[1] = 0)  # de_t / dx0 -  (num_sections, n_holes)
            grad_error_t_y0 = grad_error_y_y0 * unit_t[1] # (+ grad_error_x_y0 * unit_t[0] = 0)  # de_t / dy0 -  (num_sections, n_holes)
            grad_error_t_r = grad_error_x_r * unit_t[0] + grad_error_y_r * unit_t[1]  # de_t / dr -  (n_holes)


            ## ------------------ Compute overall log-likelihood gradients ------------------
            grad_N = - jnp.sum((error_r * grad_error_r_N) / sigma_r**2 + (error_t * grad_error_t_N) / sigma_t**2) # dL/dN
            grad_r = - jnp.sum((error_r * grad_error_r_r) / sigma_r**2 + (error_t * grad_error_t_r) / sigma_t**2) # dL/dr
            grad_x0 = - jnp.sum((error_r * grad_error_r_x0) / sigma_r**2 + (error_t * grad_error_t_x0) / sigma_t**2, axis=1) # dL/dx0
            grad_y0 = - jnp.sum((error_r * grad_error_r_y0) / sigma_r**2 + (error_t * grad_error_t_y0) / sigma_t**2, axis=1) # dL/dy0
            grad_alpha = - jnp.sum((error_r * grad_error_r_alpha)/ sigma_r**2 + (error_t * grad_error_t_alpha) / sigma_t**2, axis=1) # dL/dalpha

            # THINK THIS SHOULD BE A NEGETIVE
            grad_sigma_r = jnp.sum(error_r**2 / sigma_r**3) - len(error_r) / sigma_r # dL/dsigma_r
            grad_sigma_t = jnp.sum(error_t**2 / sigma_t**3) - len(error_t) / sigma_t # dL/dsigma_t
            grad_sigma = jnp.array([grad_sigma_r, grad_sigma_t]) # dL/dsigma
            
            grad_dict = {
                "N": grad_N,
                "r": grad_r,
                "x0": grad_x0,
                "y0": grad_y0,
                "alpha": grad_alpha,
                "sigma": grad_sigma
            }

            # If neg = True, return negative gradients for optimisation
            if neg:
                return jtu.tree_map(lambda x: -x, grad_dict)
            
            return grad_dict

    
    def compare_performance_grad(self, N, r, x0, y0, alpha, sigma, neg = False, tolerance=1e-2, num_runs=100, subset_size = 40, return_results=False):
        r"""
        Compare performance and numerical accuracy between automatic and analytical gradient computations.

        This function benchmarks the execution time, memory usage, and numerical agreement between:
        - **Automatic Differentiation (JAX Auto-Diff)**: `grad_likelihood`
        - **Manually Derived Gradients**: `analytic_grad_loglikelihood`

        Each gradient computation is performed on **random mini-batches** of data (subset size controlled by `subset_size`),
        simulating **stochastic gradient descent (SGD)** scenarios. This process is repeated `num_runs` times, and
        the **average execution time, peak memory usage, and gradient agreement** are reported.

        **Key Metrics Measured**

        - **Execution time** (seconds)
        - **Peak memory usage** (kilobytes)
        - **Numerical accuracy** (maximum absolute deviation between gradients)
        - **Gradient agreement** (boolean check if both methods agree within `tolerance`)

        Parameters
        ----------
        N : float
            Total number of holes in the original calendar ring before fragmentation.
        r : float
            Estimated radius of the ring.
        x0 : jnp.ndarray
            X-offsets for each section (shape: `num_sections`).
        y0 : jnp.ndarray
            Y-offsets for each section (shape: `num_sections`).
        alpha : jnp.ndarray
            Angular offsets (radians) for each section (shape: `num_sections`).
        sigma : float or tuple
            - If **isotropic**, a single float `sigma`.
            - If **anisotropic**, a tuple `(sigma_r, sigma_t)`.
        neg : bool, optional, default=False
            - If `True`, computes and compares the **negative log-likelihood gradients**.
            - If `False`, compares the standard log-likelihood gradients.
        tolerance : float, optional, default=1e-3
            The numerical precision threshold for gradient agreement. If absolute differences
            exceed this, a mismatch is reported.
        num_runs : int, optional, default=100
            The number of iterations used to compute averaged performance metrics.
        subset_size : int, optional, default=40
            Number of randomly selected data points per iteration, simulating **mini-batch training**.
            - Must be ≤ total dataset size.
        return_results : bool, optional, default=False
            - If `True`, returns a dictionary of performance metrics.
            - If `False`, logs the results and returns `None`.

        Returns
        -------
        dict or None
            If `return_results=True`, returns a dictionary containing:
            - `"Auto-Diff"` : Average execution time and memory usage for JAX automatic differentiation.
            - `"Manual-Diff"` : Average execution time and memory usage for manually derived gradients.
            - `"Agreement"` : `True` if gradients agree within `tolerance`, otherwise `False`.
            - `"Max Deviation"` : Maximum absolute difference between gradients across all parameters.
            - `"Deviations"` : Dictionary of parameters where numerical mismatches were found.

            If `return_results=False`, logs the results and returns `None`.

        Notes
        -----
        - Ensure that both `grad_likelihood` (automatic differentiation) and
          `analytic_grad_loglikelihood` (analytical gradients) functions are defined
          and accessible in the current scope.
        - The `subset_size` should be chosen carefully to balance computational efficiency
          and representativeness of the full dataset.
        """

        # Input validation
        if not isinstance(N, float):
            raise TypeError("N must be a float.")
        if not isinstance(r, float):
            raise TypeError("r must be a float.")
        if not isinstance(x0, jnp.ndarray):
            raise TypeError("x0 must be a JAX array.")
        if not isinstance(y0, jnp.ndarray):
            raise TypeError("y0 must be a JAX array.")
        if not isinstance(alpha, jnp.ndarray):
            raise TypeError("alpha must be a JAX array.")
        if not isinstance(sigma, (float, jnp.ndarray)):
            raise TypeError("sigma must be a float or JAX array.")
        
        # Enure data subset size is valid
        if subset_size > self.data.shape[0]:
            raise ValueError(f"Subset size cannot exceed the total number of data points: {self.data.shape[0]}.")


        # ------------------ Initialize Storage for Statistics ------------------
        auto_times, auto_memory = [], []
        analytic_times, analytic_memory = [], []
        
        all_agree = True
        max_deviation = 0.0
        deviation_details = {}

        # ------------------ Run Performance Test Over Multiple Iterations ------------------
        for _ in range(num_runs):
            # Take a random batch of 40 data points 
            data_subset = self.data.sample(subset_size) 

            # ------------------ Measure analytic_grad_loglikelihood (Manual) ------------------
            # Start memory and time tracing
            tracemalloc.start()
            start_time = time.perf_counter()

            grad_analytic = self.analytic_grad_loglikelihood(N, r, x0, y0, alpha, sigma, neg, data_subset)

            end_time = time.perf_counter()
            _, analytic_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate difference in time and memory
            analytic_times.append(end_time - start_time)
            # Convert to KB
            analytic_memory.append(analytic_peak / 1024) 

            # ------------------ Measure grad_likelihood (Auto-diff) ------------------
            # Start memory and time tracing
            tracemalloc.start()
            start_time = time.perf_counter()

            grad_auto = self.grad_likelihood(N, r, x0, y0, alpha, sigma, log = True, neg = neg, data = data_subset)

            end_time = time.perf_counter()
            _, auto_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate difference in time and memory
            auto_times.append(end_time - start_time)
            # Convert to KB
            auto_memory.append(auto_peak / 1024)  

            # ------------------ Compare Gradients ------------------
            # Run through each output - for derivative wrt each parameter 
            for key in grad_auto.keys():
                auto_grad = jnp.asarray(grad_auto[key])
                analytic_grad = jnp.asarray(grad_analytic[key])
                # Difference between results achieved for both methods
                abs_diff = jnp.abs(auto_grad - analytic_grad)
                max_diff = jnp.max(abs_diff) if abs_diff.size > 0 else 0.0

                # Ensure that the gradients are close within the tolerance
                if not jnp.all(abs_diff < tolerance):
                    all_agree = False
                    # Ensure that the max diff and where it occoured is stored for reporting
                    if key not in deviation_details or deviation_details[key] < max_diff:
                        deviation_details[key] = max_diff

                # Update/ store the maximum deviation if it exceeds the current maximum
                max_deviation = max(max_deviation, max_diff)

        # ------------------ Compute Means and Std Dev ------------------
        avg_auto_time = np.mean(auto_times)
        std_auto_time = np.std(auto_times)
        avg_auto_memory = np.mean(auto_memory)
        std_auto_memory = np.std(auto_memory)

        avg_analytic_time = np.mean(analytic_times)
        std_analytic_time = np.std(analytic_times)
        avg_analytic_memory = np.mean(analytic_memory)
        std_analytic_memory = np.std(analytic_memory)

        # ------------------ Logging and Output ------------------
        agreement_status = " MATCH" if all_agree else " MISMATCH"

        log_message = (
            f"\nPerformance & Accuracy Comparison ({num_runs} runs):\n"
            f"{'-'*60}\n"
            f"Method:                 Auto-Diff              Manual-Diff\n"
            f"Avg Execution Time (s): {avg_auto_time:.6f} ± {std_auto_time:.6f}   {avg_analytic_time:.6f} ± {std_analytic_time:.6f}\n"
            f"Avg Peak Memory (KB):   {avg_auto_memory:.2f} ± {std_auto_memory:.2f}        {avg_analytic_memory:.2f} ± {std_analytic_memory:.2f}\n"
            f"Gradient Agreement:     {agreement_status}\n"
            f"Max Deviation:          {max_deviation:.3e}\n"
        )

        if not all_agree:
            log_message += f"Deviations found in: {deviation_details}\n"

        log_message += f"{'-'*60}"
        logging.info(log_message)

        # ------------------ Return Results if Requested ------------------
        if return_results:
            return {
                "Auto-Diff": {"Time (s)": avg_auto_time, "Peak Memory (KB)": avg_auto_memory},
                "Manual-Diff": {"Time (s)": avg_analytic_time, "Peak Memory (KB)": avg_analytic_memory},
                "Agreement": all_agree,
                "Max Deviation": max_deviation,
                "Deviations": deviation_details if not all_agree else "None"
            }

        return None
    
    # ------------------ Independent Sampling from Priors ------------------

    def sample_from_priors(self, key, num_samples=100):
        r"""
        Generate samples from the prior distributions of model parameters.

        This function uses **NumPyro's sampling utilities** to draw `num_samples` realisations from 
        the prior distributions defined during model initialisation. These samples can be used for:

        It supports **both isotropic and anisotropic models**:
        - **Isotropic model**: A single standard deviation `sigma` is sampled.
        - **Anisotropic model**: Independent `sigma_r` and `sigma_t` values are sampled.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for JAX-based sampling, ensuring reproducibility.
        num_samples : int, optional, default=100
            Number of parameter samples to generate.

        Returns
        -------
        dict
            A dictionary where:
            - Each **key** represents a parameter name (`"N"`, `"r"`, `"x0"`, `"y0"`, `"alpha"`, `"sigma"`).
            - Each **value** is a JAX array of shape `(num_samples, ...)`, where:
              - Scalar parameters (`"N"`, `"r"`, `"sigma"`) have shape `(num_samples,)`.
              - Section-dependent parameters (`"x0"`, `"y0"`, `"alpha"`) have shape `(num_samples, num_sections)`.
              - **For anisotropic models**, `"sigma"` is of shape `(num_samples, 2)`, with `[sigma_r, sigma_t]` per row.
        """
        # Generate subkeys for each parameter from overall key
        keys = jax.random.split(key, len(self.priors) + 1)

        # Use NumPyro to sample from the prior distributions defined during model initialisation
        # For each it will sample num_samples times and return the array of samples for each parameter in overall dictionary
        sampled_params = {
            "N": numpyro.sample("N", self.priors["N"], rng_key=keys[0], sample_shape=(num_samples,)),
            "r": numpyro.sample("r", self.priors["r"], rng_key=keys[1], sample_shape=(num_samples,)),
            "x0": numpyro.sample("x0", self.priors["x0"].expand([self.num_sections]), 
                                rng_key=keys[2], sample_shape=(num_samples,)),
            "y0": numpyro.sample("y0", self.priors["y0"].expand([self.num_sections]), 
                                rng_key=keys[3], sample_shape=(num_samples,)),
            "alpha": numpyro.sample("alpha", self.priors["alpha"].expand([self.num_sections]), 
                                    rng_key=keys[4], sample_shape=(num_samples,)),
        }

        # Sample sigma values based on model type
        if self.model_type == "isotropic":
            sampled_params["sigma"] = numpyro.sample("sigma", self.priors["sigma"], 
                                                    rng_key=keys[5], sample_shape=(num_samples,))
            
        # If anisotropic model, sample sigma_r and sigma_t separately and then stack them in pairs - (num_samples, 2)
        elif self.model_type == "anisotropic":
            sampled_params["sigma"] = jnp.stack([
                numpyro.sample("sigma_r", self.priors["sigma_r"], rng_key=keys[6], sample_shape=(num_samples,)),
                numpyro.sample("sigma_t", self.priors["sigma_t"], rng_key=keys[7], sample_shape=(num_samples,))
            ], axis=1)  

        return sampled_params

    # ------------------ Maximum Likelihood Estimation (MLE) ------------------

    def max_likelihood_est(self, sampling_type, num_samples=1000, num_iterations=500, learning_rate=0.01, batch_size=None, key=None, derivative='analytic', plot_history=False, summary_table = None, save_path = None):
        r"""
        Perform Maximum Likelihood Estimation (MLE) using stochastic optimization methods.

        Supported methods:
        - Stochastic Gradient Descent (SGD)
        - Adam Optimizer
        - BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm)

        **Recommended Hyperparameters for Each Method:**

        - **SGD (Stochastic Gradient Descent)**
          - `learning_rate`: **0.01 - 0.1** (Recommended: **0.01**)
          - `num_iterations`: **500 - 5000** (Recommended: **1000**)
          - `batch_size`: **20 - 100** (If `None`, full batch is used)
          - `derivative`: **'analytic'** (Recommended) or `'auto'`

        - **Adam Optimizer**
          - `learning_rate`: **0.001 - 0.01** (Recommended: **0.01**)
          - `num_iterations`: **500 - 5000** (Recommended: **1000**)
          - `batch_size`: **20 - 100** (If `None`, full batch is used)
          - `derivative`: **'analytic'** (Recommended) or `'auto'`

        - **BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm)**
          - `None` (This is a gradient-based optimization algorithm that typically determines its own step size.)
          - `num_iterations`: The optimization will run until convergence or the maximum number of function evaluations/iterations is reached (controlled by SciPy's `minimize` function).
          - `batch_size`: Not directly applicable as BFGS is generally a full-batch optimization method.
          - `derivative`: **'analytic'** (Recommended for performance and stability) or `'auto'` (though analytical gradients are generally preferred for BFGS).

        Parameters
        ----------
        sampling_type : str
            Optimization method: 'SGD', 'Adam', or 'BFGS'.
        num_samples : int, optional
            Number of parameter initializations to optimize. Default is `1000`.
        num_iterations : int, optional
            Number of iterations for the optimization algorithm (for SGD and Adam). For BFGS, this influences the termination criteria of the `minimize` function. Default is `500`.
        learning_rate : float, optional
            Learning rate for gradient-based optimization (for SGD and Adam). Default is `0.01`.
        batch_size : int, optional
            Size of minibatches for stochastic gradient estimation (for SGD and Adam). Default is `None` (full-batch). Not directly applicable to BFGS.
        key : jax.random.PRNGKey, optional
            Random key for reproducibility. Default is a fixed seed.
        derivative : str, optional
            - `'analytic'`: Uses manually computed gradients.
            - `'auto'`: Uses automatic differentiation via `jax.grad()`.
        plot_history : bool, optional
            If `True`, plots log-likelihood history during optimization (only supported for single sample optimization with SGD and Adam).
        summary_table : int, optional
            If provided, displays a table with the top `summary_table` results ranked by log-likelihood.
        save_path : str, optional
            If provided, saves the optimisation results to a Pickle file.

        Returns
        -------
        dict
            Dictionary containing the best parameter set found.
        """
        ### MADE A CHANGE HERE: 

        # Ensure that the sampling type is valid
        # if sampling_type not in ['SGD',  'Adam']:
        #     raise ValueError("sampling_type must be one of 'SGD' or 'Adam'.")
        
        if sampling_type not in ['SGD',  'Adam', 'BFGS']:
            raise ValueError("sampling_type must be one of 'SGD', 'Adam', or 'BFGS'.")
        
        if summary_table is not None:
            if not isinstance(summary_table, int) or summary_table >= num_samples:
                raise ValueError("Summary table must be an integer smaller than the number of samples.")
        
        # Ensure that if plotting history it is simply a single optimisation
        if plot_history and num_samples > 1:
            raise ValueError("Plot of log-likelihood history is only supported for single sample optimisation as an overview of the training process.")
        
        # Allow for either stochastic or full batch gradient computation
        # If batch size is provided, ensure it is valid - ie less than total number of data points
        if batch_size is not None and batch_size > self.data.shape[0]:
                raise ValueError(f"Batch size cannot exceed the total number of data points: {self.data.shape[0]}.")

        # Allow user to choose between automatic and analytical gradients - analytical is default as shown to be faster and more memory efficient
        if derivative not in ['analytic', 'auto']:
            raise ValueError("Derivative method must be either 'analytic' or 'auto'.")
        
        # Ensure a valid key is available to the function
        if key is None:
            key = jax.random.PRNGKey(42)
        else:
            key = jax.random.PRNGKey(key)
        
        # Initialize a list to store the trajectory of parameter estimates and log-likelihoods - allows them to be analysed later
        all_results = [] 

        # As this is only every used for single sample optimisation can be initialised globally here
        if plot_history:
            all_results_log_likelihood = []

            def _plot_log_likelihood_history(all_results_log_likelihood):
                """Plot the log-likelihood history, removing the first 10% of iterations for clarity."""

                # Remove first 20% of iterations
                num_iterations = len(all_results_log_likelihood)
                cutoff = max(1, int(0.2 * num_iterations)) 
                plot_data = all_results_log_likelihood[cutoff:]

                # Set high-quality plot settings
                plt.figure(figsize=(6, 4), dpi=300) 
                sns.set_context("talk")  
                sns.set_style("whitegrid") 

                # Plot log-likelihood history
                plt.plot(plot_data, linestyle='-', color='darkblue', linewidth=2, alpha=0.8, label="Negative Log-Likelihood - Gradient Descent")

                # Labels and formatting
                plt.xlabel("Iterations", fontsize=12, fontweight='bold')
                plt.ylabel("Negative Log-Likelihood", fontsize=12, fontweight='bold')
                plt.title("Log-Likelihood Convergence", fontsize=12, fontweight='bold', pad=15)
                plt.yscale("log")
                plt.grid(which="both", linestyle="--", linewidth=0.7, alpha=0.6)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=18)
                plt.tight_layout()
                plt.legend(fontsize=10)
                plt.show()

        # Define the loss function to be minimized - negative log-likelihood
        def loss_fn(params, data_subset):
            """Negative log-likelihood function that will be minimised."""
            return self.likelihood(params["N"], params["r"], params["x0"], params["y0"], params["alpha"], params["sigma"], 
                                    log=True, neg=True, data=data_subset)
        
        # Define the gradient function based on user choice for relevant methods
        if sampling_type in ['SGD', 'Adam']:
            if derivative == 'auto':
                grad_fn = jax.grad(loss_fn)
            else:
                def grad_fn(params, data_subset):
                    """Derivative Negative log-likelihood function that will be minimised."""
                    return self.analytic_grad_loglikelihood(params["N"], params["r"], params["x0"], params["y0"], params["alpha"], 
                                                            params["sigma"], neg=True, data=data_subset)
        

        # Sample from the priors simply as initialisation position for MLE optimisation
        # All initial samples are retrieved in one go
        prior_samples = self.sample_from_priors(key, num_samples=num_samples)


        # ------------------ BFGS Optimizer ------------------

        if sampling_type == 'BFGS':

            # Check if sigma is anisotropic (vector per section) or isotropic (scalar)
            sigma_size = 2 if self.model_type == "anisotropic" else 1

            def flatten_params(params_dict):
                return np.concatenate([
                    [params_dict["N"]],
                    [params_dict["r"]],
                    np.array(params_dict["x0"]),
                    np.array(params_dict["y0"]),
                    np.array(params_dict["alpha"]),
                    np.array(params_dict["sigma"]) if sigma_size > 1 else [params_dict["sigma"]]
                ])

            def unflatten_params(param_vector):
                idx = 0
                N = param_vector[idx]; idx += 1
                r = param_vector[idx]; idx += 1
                x0 = param_vector[idx:idx + self.num_sections]; idx += self.num_sections
                y0 = param_vector[idx:idx + self.num_sections]; idx += self.num_sections
                alpha = param_vector[idx:idx + self.num_sections]; idx += self.num_sections
                sigma = param_vector[idx:idx + sigma_size] if sigma_size > 1 else param_vector[idx]
                return {
                    "N": N,
                    "r": r,
                    "x0": x0,
                    "y0": y0,
                    "alpha": alpha,
                    "sigma": sigma
                }

            # Build bounds for all parameters
            bounds = []
            for param in self.priors.keys():
                prior = self.priors[param]
                if param in ["x0", "y0", "alpha"]:
                    size = self.num_sections
                elif param in ["sigma_r", "sigma_t"]:
                    size = 1  # anisotropic case: each has scalar sigma per direction
                else:
                    size = 1
                bounds.extend([(prior.low, prior.high)] * size)

            for i in tqdm(range(num_samples), desc="Optimizing MLE using BFGS:", leave=True):
                init_params = {k: np.array(v[i]) for k, v in prior_samples.items()}
                x0_flat = flatten_params(init_params)

                def scipy_loss(param_vector):
                    param_dict = unflatten_params(param_vector)
                    return loss_fn(param_dict, self.data)

                result = minimize(scipy_loss, x0_flat, method='L-BFGS-B', bounds=bounds)

                # 1. Extracting the inverse Hessian approximation
                if hasattr(result, "hess_inv"):
                    try:
                        hess_inv = result.hess_inv.todense()  # if it's a sparse matrix
                    except AttributeError:
                        hess_inv = np.array(result.hess_inv)  # already dense
                else:
                    hess_inv = None

                # 2. Calculate the Standard deviation on each parameter is the sqrt of diagonal of the inverse Hessian
                if hess_inv is not None:
                    std_errors = np.sqrt(np.diag(hess_inv))
                else:
                    std_errors = None  # Could not compute


                best_params = unflatten_params(result.x)
                best_params["alpha"] = (best_params["alpha"] + np.pi) % (2 * np.pi) - np.pi

                final_log_likelihood = -result.fun
                all_results.append({"params": best_params, "log_likelihood": final_log_likelihood,  "std_errors": std_errors})


        # ------------------ Stochastic Gradient Descent ------------------

        if sampling_type == 'SGD':

            # Define the optimiser and initialise it with the learning rate - optax
            optimizer = optax.sgd(learning_rate)

            # Loop over all num_samples and optimise each one
            for i in tqdm(range(num_samples), desc="Optimizing MLE using SGD:", leave=True):

                # Extract the parameters for the current sample from the dictionary - ensure they are jnp arrays
                params = {k: jnp.array(v[i]) for k, v in prior_samples.items()}

                # Enter initial state for the optimiser
                opt_state = optimizer.init(params)

                # Run the SGD loop for num_iterations 
                for _ in range(num_iterations):

                    # Store the log-likelihood history for each iteration
                    if plot_history:
                        all_results_log_likelihood.append(loss_fn(params, self.data))

                    data_subset = self.data if batch_size is None else self.data.sample(batch_size)
                    grads = grad_fn(params, data_subset)
                    updates, opt_state = optimizer.update(grads, opt_state)
                    params = optax.apply_updates(params, updates)

                # Plot the log-likelihood history using the function defined above
                if plot_history:
                    _plot_log_likelihood_history(all_results_log_likelihood)

                # Ensure that alpha is within [-π, π]
                params["alpha"] = (params["alpha"] + jnp.pi) % (2 * jnp.pi) - jnp.pi

                # Convert from minimised negative log-likelihood to log-likelihood (maximised)
                final_log_likelihood = -loss_fn(params, self.data)

                # Store the final parameter estimates and log-likelihood
                all_results.append({"params": params, "log_likelihood": final_log_likelihood})


        # ------------------ Adam Optimizer ------------------

        if sampling_type == 'Adam':

            # Define the Adam optimizer and initialize it with the learning rate
            optimizer = optax.adam(learning_rate)

            # Loop over all num_samples and optimize each one
            for i in tqdm(range(num_samples), desc="Optimizing MLE using Adam:", leave=True):

                # Extract the parameters for the current sample from the dictionary - ensure they are jnp arrays
                params = {k: jnp.array(v[i]) for k, v in prior_samples.items()}

                # This simply helps sensibly initialise sigma for the model rather than using 
                # loguniform prior
                if self.model_type == 'anisotropic':
                    params['sigma'] = jnp.array([0.1, 0.1])
                if self.model_type == 'isotropic':
                    params['sigma'] = jnp.array([0.1])

                # Initialise the optimizer state
                opt_state = optimizer.init(params)

                # Run the Adam optimization loop for num_iterations
                for iteration in range(num_iterations):

                    # Store the log-likelihood history for tracking
                    if plot_history:
                        all_results_log_likelihood.append(loss_fn(params, self.data))

                    # Sample mini-batch if batch_size is set
                    data_subset = self.data if batch_size is None else self.data.sample(batch_size)
                    grads = grad_fn(params, data_subset)
                    updates, opt_state = optimizer.update(grads, opt_state)
                    params = optax.apply_updates(params, updates)

                # Plot the log-likelihood history if enabled
                if plot_history:
                    _plot_log_likelihood_history(all_results_log_likelihood)

                # Ensure that alpha is within [-π, π]
                params["alpha"] = (params["alpha"] + jnp.pi) % (2 * jnp.pi) - jnp.pi

                # Convert from minimized negative log-likelihood to log-likelihood (maximized)
                final_log_likelihood = -loss_fn(params, self.data)

                # Store the final parameter estimates and log-likelihood
                all_results.append({"params": params, "log_likelihood": final_log_likelihood})
        

        # Apply a filter on all results that removed any unphysical values - ie N < 0 or r < 0
        filtered_results = [entry for entry in all_results if entry["params"]["N"] > 0 and entry["params"]["r"] > 0]
        num_removed = num_samples - len(filtered_results)
        logging.info(f"Removed {num_removed}/{num_samples} MLE estimates due to unphysical values (N or r < 0).")
        if not filtered_results:
            raise RuntimeError("All estimated parameters were invalid. Consider adjusting priors.")
        
        # Find the best result based on maximum log-likelihood
        best_result = max(filtered_results, key=lambda x: x["log_likelihood"])


        if summary_table is not None: 
            # Sort results by log-likelihood in descending order
            top_results = sorted(filtered_results, key=lambda x: x["log_likelihood"], reverse=True)[:summary_table]

            # Prepare data for the table
            table_data = []

            for result in top_results:
                params = result["params"]
                log_likelihood = result["log_likelihood"]

                # Convert vector parameters to a readable format
                x0_str = np.array2string(params["x0"], precision=2, separator=',')
                y0_str = np.array2string(params["y0"], precision=2, separator=',')
                alpha_str = np.array2string(params["alpha"], precision=2, separator=',')
                
                # Handle sigma separately (single value for isotropic, vector for anisotropic)
                if isinstance(params["sigma"], jax.Array) or isinstance(params["sigma"], np.ndarray):
                    sigma_value = np.array(params["sigma"])  # Convert JAX array to NumPy
                    sigma_str = f"{sigma_value.item():.2f}" if sigma_value.size == 1 else np.array2string(sigma_value, precision=2, separator=',')
                else:
                    sigma_str = f"{params['sigma']:.2f}"

                # Append row to table data
                table_data.append([
                    params["N"], params["r"], x0_str, y0_str, alpha_str, sigma_str, log_likelihood
                ])

            # Create a DataFrame
            columns = ["N", "r", "x0", "y0", "alpha", "sigma", "Log-Likelihood"]
            df = pd.DataFrame(table_data, columns=columns)
            display(df)
        
        if save_path is not None: 
            # save dictionary to pickle file
            with open(save_path, "wb") as f:
                pickle.dump(filtered_results, f)
            logging.info(f"MLE Run results saved to {save_path}")
        
        return filtered_results
    

    
    def mle_analysis(self, mle_results, figsize=(10, 8)):
        """
        Visualiwes the observed hole locations and the estimated hole locations 
        obtained via Maximum Likelihood Estimation (MLE), with clear distinctions 
        between the observed and modeled data points.

        This function:
        - Identifies the best-fit MLE result based on the maximum log-likelihood.
        - Computes the corresponding estimated hole positions.
        - Plots observed hole locations alongside MLE-modeled hole locations.
        - Annotates selected hole numbers for clarity.
        - Ensures visually clear distinctions between observed and estimated positions.

        Parameters
        ----------
        mle_results : list of dict
            A list of dictionaries containing MLE results. Each dictionary should include:
            - `"params"` : dict of best-fit parameters {N, r, x0, y0, alpha}.
            - `"log_likelihood"` : The log-likelihood value for the corresponding fit.
        
        figsize : tuple, optional, default=(10, 8)
            Figure size for the generated plot.

        Returns
        -------
        None
        Displays a **scatter plot** of observed and estimated hole positions.
        """
        # Find MLE best result parameters
        mle_best_result_dict = max(mle_results, key=lambda x: x["log_likelihood"])
        mle_best_result = mle_best_result_dict['params']

        # Find corresponding hole positions
        mle_hole_positions = self.hole_positions(N=float(mle_best_result["N"]), r=float(mle_best_result["r"]), 
                                                x0=jnp.array(mle_best_result["x0"]), y0=jnp.array(mle_best_result["y0"]), 
                                                alpha=np.array(mle_best_result["alpha"]))
        mle_hole_positions_x = mle_hole_positions.T[0]
        mle_hole_positions_y = mle_hole_positions.T[1]

        # Initialise the plot
        plt.figure(figsize=figsize)
        sns.set_context("talk")
        sns.set_style("whitegrid")
        plt.rcParams["font.family"] = "serif"

        # Define a color mapping for each unique section
        unique_sections = self.data["Section ID"].unique()
        palette = sns.color_palette("Dark2", len(unique_sections))

        # Plot observed and estimated positions
        plt.scatter(self.data["Mean(X)"], self.data["Mean(Y)"],
                    color='black', label="Observed Positions", 
                    s=80, edgecolors='black', alpha=0.6, marker='o')

        plt.scatter(mle_hole_positions_x, mle_hole_positions_y, 
                    color='red', label="MLE Estimated Positions", 
                    s=80, alpha=0.6, marker='x')

        # Annotate every 3rd hole, adjusting position for visibility
        for i in range(self.n_holes):
            hole_num_int = int(self.data.iloc[i]["Hole"])
            if (hole_num_int - 1) % 3 == 0:
                x, y = self.data.iloc[i]["Mean(X)"], self.data.iloc[i]["Mean(Y)"]
                offsets = {
                    hole_num_int < 29: (-3, 0, 'right', 'center'),
                    28 < hole_num_int < 46: (-4, 0, 'center', 'top'),
                    45 < hole_num_int < 69: (0, -1, 'center', 'top'),
                    hole_num_int >= 69: (5, 0, 'left', 'center')
                }
                dx, dy, ha, va = offsets[True]
                plt.text(x + dx, y + dy, str(hole_num_int), fontsize=12, ha=ha, va=va, color='black')

        # Configure legend
        plt.legend(title="Hole Locations", loc="upper right", fontsize=12, frameon=True)

        # Configure plot limits and labels
        x_min, x_max = self.data["Mean(X)"].min(), self.data["Mean(X)"].max()
        y_min, y_max = self.data["Mean(Y)"].min(), self.data["Mean(Y)"].max()
        plt.xlim(x_min - 5, x_max + 5.5)
        plt.ylim(y_min - 4, y_max + 4)
        plt.xlabel("X Position (mm)", fontsize=14)
        plt.ylabel("Y Position (mm)", fontsize=14)
        plt.title("Measured and MLE Modeled Hole Locations", fontsize=16)
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
        plt.minorticks_on()

        plt.show()
    

    def numpryo_model(self):
        r"""
        Defines the model and the likelihood function for Bayesian inference using NumPyro.

        The function infers the following parameters:
        - `N` : The total number of holes in the full calendar ring.
        - `r` : The estimated radius of the ring before fragmentation.
        - `x0_j, y0_j` : The center offsets for each section.
        - `alpha_j` : The rotation angles for each section.
        - `sigma` (for isotropic model) : Common standard deviation for errors.
        - `sigma_r, sigma_t` (for anisotropic model) : Standard deviations for radial and tangential errors.

        The likelihood follows a Gaussian model:

        - **Isotropic Gaussian:**

          .. math::
             p(\{d_i\} | \mathbf{a}) = (2\pi\sigma^2)^{-n} \prod_{i=1}^{n} \exp \left[ -\frac{(e_{i,x})^2 + (e_{i,y})^2}{2\sigma^2} \right]

        - **Anisotropic Gaussian:**

          .. math::
             p(\{d_i\} | \mathbf{a}) = (2\pi\sigma_r\sigma_t)^{-n} \prod_{j=0}^{s-1} \prod_{i \in j} \exp \left[ -\frac{(e_{ij} \cdot \hat{r}_{ij})^2}{2\sigma_r^2} - \frac{(e_{ij} \cdot \hat{t}_{ij})^2}{2\sigma_t^2} \right]

        The choice between isotropic and anisotropic models is determined by `self.model_type`.

        Parameters
        ----------
        None (Uses class attributes)

        Returns
        -------
        None
            This function does not return values directly; instead, it defines the probabilistic model within NumPyro.
        """

        # -------------------- Define model parameters --------------------
        # Priors defined simultaneously (Probabilistic Programming Language)

        # Universal Parameters: N, r, sigma_r, sigma_t
        # These are constant for all sections and holes as they are prior to the breaking of the ring
        # Total number of holes in the calendar ring
        N = numpyro.sample("N", self.priors["N"])
        # Ring radius
        r = numpyro.sample("r", self.priors["r"])
        
        # Section-specific Parameters: x0, y0, alpha (due to relative rotation and translation)
        # These are intialised as vectors of length = number of sections (ith element = section labelled i+1)
        x0 = numpyro.sample("x0", self.priors["x0"].expand([self.num_sections]))
        y0 = numpyro.sample("y0", self.priors["y0"].expand([self.num_sections]))
        alpha = numpyro.sample("alpha", self.priors["alpha"].expand([self.num_sections]))

        # Determine sigma based on model type
        if self.model_type == "isotropic":
            sigma = numpyro.sample("sigma", self.priors["sigma"])
        elif self.model_type == "anisotropic":
            sigma_r = numpyro.sample("sigma_r", self.priors["sigma_r"])
            sigma_t = numpyro.sample("sigma_t", self.priors["sigma_t"])
            sigma = jnp.array([sigma_r, sigma_t])

        # -------------------- Define log likelihood --------------------
        numpyro.factor("log_likelihood", self.likelihood(N, r, x0, y0, alpha, sigma, log=True, neg =False))



    def run_hmc_nuts(self, burnin_period=2000, n_samples=4000, n_chains=1, step_size=1, acceptance_prob=0.8, dense_mass=False, summary = False, random_seed=0, save_path=None, traceplot=None, progress_bar=True):
        """
        Runs the Hamiltonian Monte Carlo (HMC) inference using the No-U-Turn Sampler (NUTS).

        This function performs **Bayesian inference** using the NUTS algorithm, which efficiently
        samples from the posterior distribution of the parameters while tuning itself for
        computational efficiency.

        The function supports:
        - **Tuning hyperparameters** (step size, acceptance probability, mass matrix).
        - **Multiple independent chains** for parallelised inference.
        - **Saving posterior samples** to a NetCDF file for later analysis.
        - **Trace plots** to diagnose convergence and sampling quality.

        Parameters
        ----------
        burnin_period : int, optional, default=2000
            Number of warm-up (burn-in) samples used to adapt the step size and mass matrix.
        n_samples : int, optional, default=4000
            Number of posterior samples to collect after the burn-in period.
        n_chains : int, optional, default=1
            Number of independent MCMC chains to run in parallel.
        step_size : float, optional, default=1
            Initial step size for the NUTS sampler. If `None`, the sampler adapts the step size automatically.
        acceptance_prob : float, optional, default=0.8
            Target acceptance probability for the sampler (recommended range: `0.6 - 0.95`).
        dense_mass : bool, optional, default=False
            Whether to use a **dense** mass matrix (`True`) or a **diagonal** mass matrix (`False`).
        summary : bool, optional, default=False
            If `True`, prints summary statistics of the posterior samples.
        random_seed : int, optional, default=0
            Random seed for reproducibility.
        save_path : str or None, optional, default=None
            If provided, saves the MCMC posterior samples as a NetCDF file.
        traceplot : list or None, optional, default=None
            List of parameter names to include in **trace plots**. If `None`, no plots are generated.
        progress_bar : bool, optional, default=True
            If `True`, displays a progress bar during sampling.

        Returns
        -------
        mcmc : numpyro.infer.MCMC
            The MCMC object containing:
            - Posterior samples.
            - Log-likelihood estimates.
            - Diagnostic statistics.
        samples_time : float
            Estimated runtime (in seconds) for collecting posterior samples (excluding burn-in).
        """

        # Initialize the RNG key
        rng_key = jax.random.PRNGKey(random_seed)
        # Define the NUTS sampler with hyperparameter options
        nuts_kernel = NUTS(self.numpryo_model, step_size=step_size, target_accept_prob=acceptance_prob, dense_mass=dense_mass)
        # Initialise MCMC
        mcmc = MCMC(nuts_kernel, num_warmup=burnin_period, num_samples=n_samples, num_chains=n_chains, progress_bar=progress_bar)
        # Start timing
        start_time = time.time()
        # Run MCMC
        mcmc.run(rng_key)
        # Ensure computations are completed before stopping the timer
        jax.block_until_ready(mcmc.get_samples())
        end_time = time.time()
        # Total Run Time
        total_time = end_time - start_time 

        # Preportion of time spent not on the burn in 
        samples_time = total_time * (n_samples / (n_samples + burnin_period))

        if summary:
            # Print summary statistics
            mcmc.print_summary()


        # Save samples if path is provided
        if save_path is not None:
            posterior_data = az.from_numpyro(mcmc)
            dataset = posterior_data.posterior
            dataset.to_netcdf(save_path)
            logging.info(f" MCMC samples saved to {save_path}")

       # Generate trace plots if requested

        if traceplot is not None:
            posterior_data = az.from_numpyro(mcmc)
            plot_params = traceplot

            # Custom labels for proper
            param_labels = {
                "N": "Total Holes $N$",
                "r": "Radius $r$",
                "sigma_r": "Radial Error $\sigma_r$",
                "sigma_t": "Tangential Error $\sigma_t$",
            }

            var_names = [param for param in plot_params if param in posterior_data.posterior.keys()]
            label_list = [param_labels.get(param, param) for param in var_names]

            # ArviZ-allowed styling (limited)
            az.style.use("arviz-darkgrid")

            # Generate the trace plot
            axes = az.plot_trace(
                posterior_data,
                var_names=var_names,
                figsize=(10, len(var_names) * 3.5),
                compact=True,
                chain_prop={"color": ["red", "blue", "black", "purple"]},
                legend=True
            )

            # Apply Matplotlib customizations to all axes
            for i, ax_group in enumerate(axes):
                for ax in (ax_group if isinstance(ax_group, np.ndarray) else [ax_group]):
                    ax.tick_params(axis='x', labelsize=14)
                    ax.tick_params(axis='y', labelsize=14)
                    ax.set_xlabel("Sample", fontsize=14)
                    ax.set_title(label_list[i], fontsize=17, pad=12)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.5)
            plt.show()        
        logging.info(f"Total MCMC Run Time: {total_time:.2f} seconds \n Non Burn-In Run Time: {samples_time:.2f} seconds")

        return mcmc, samples_time
 

    def run_hmc_optimisation(self, step_size_range, acceptance_prob_range, dense_mass_options, burnin_period=2000, n_samples=3000, n_chains=4, random_seed=0, save_path=None, no_table_results=1):
        """
        Performs a grid search over HMC hyperparameters to identify the optimal configuration.

        This function systematically evaluates different combinations of:
        - Step size (`step_size_range`)
        - Target acceptance probability (`acceptance_prob_range`)
        - Mass matrix structure (`dense_mass_options`)

        Each configuration is assessed using multiple **Markov Chain Monte Carlo (MCMC)** runs
        with the No-U-Turn Sampler (NUTS). Calculates performance metrics such as the **number of effective samples**,
        **autocorrelation length**, **Gelman-Rubin diagnostic (R-hat)**, and **computational efficiency**
        are recorded.

        The best hyperparameter set is determined based on the **Time Per Effective Sample (TPES)**,
        which measures sampling efficiency.

        Parameters
        ----------
        step_size_range : list of float
            A list of step sizes to test.
        acceptance_prob_range : list of float
            A list of target acceptance probabilities (e.g., `[0.7, 0.8, 0.9]`).
        dense_mass_options : list of bool
            A list of mass matrix choices:
            - `True` → Use a **dense mass matrix** (computationally expensive, better for correlated parameters).
            - `False` → Use a **diagonal mass matrix** (cheaper, but less effective for strongly correlated parameters).
        burnin_period : int, optional, default=2000
            The number of warm-up (burn-in) samples used to adapt step size and mass matrix.
        n_samples : int, optional, default=3000
            The number of posterior samples to collect after burn-in.
        n_chains : int, optional, default=4
            The number of independent MCMC chains to run in parallel.
            **Note:** Must be at least `2` to compute the **Gelman-Rubin (R-hat) statistic**.
        random_seed : int, optional, default=0
            Random seed for reproducibility.
        save_path : str or None, optional, default=None
            If provided, saves the hyperparameter search results as a CSV file.
        no_table_results : int, optional, default=1
            The number of top configurations to display in the final summary table.

        Returns
        -------
        results_df : pandas.DataFrame
            A DataFrame containing performance metrics for each hyperparameter combination:
            - `"Step Size"` : Step size used.
            - `"Acceptance Probability"` : Target acceptance probability.
            - `"Dense Mass Matrix"` : Boolean indicating mass matrix type.
            - `"Autocorrelation Length"` : Estimated autocorrelation length.
            - `"Number Effective Samples"` : Minimum effective sample size across parameters.
            - `"GR Statistic"` : Gelman-Rubin diagnostic (values close to 1 indicate convergence).
            - `"Time per Iteration"` : Average computation time per sample.
            - `"Time per Effective Sample (TPES)"` : Efficiency metric (lower is better).

        best_params : dict
            The best hyperparameter combination based on the lowest **Time Per Effective Sample (TPES)**.

        Notes
        -----
        - **Performance Metrics**:
          - **Time per Effective Sample (TPES)**: Lower values indicate a more efficient sampler.
          - **R-hat Statistic**: Should be close to `1.0` (values > `1.1` suggest non-convergence).
          - **Autocorrelation Length**: Higher values indicate slower mixing.
        """

        if n_chains < 2:
            raise ValueError("Number of chains must be at least 2 for Gelman-Rubin diagnostics.")

        # Create a grid of all hyperparameter combinations
        param_grid = list(itertools.product(step_size_range, acceptance_prob_range, dense_mass_options))
        # Total number of search experiments
        total_iterations = len(param_grid)
        # Initialise storage of results
        results = []

        # Iterate over all the hyperparameter combinations
        for i, (step_size, accept_prob, dense_mass) in enumerate(param_grid, start=1):
            logging.info(f"Running MCMC {i}/{total_iterations} | step_size={step_size}, accept_prob={accept_prob}, dense_mass={dense_mass}")
            
            # Run MCMC with the current hyperparameters from search
            mcmc, sample_time= self.run_hmc_nuts(burnin_period=burnin_period, n_samples=n_samples, n_chains=n_chains,step_size=step_size, acceptance_prob=accept_prob, dense_mass=dense_mass,
            random_seed=random_seed, save_path=None, traceplot=None, progress_bar=False)

            # Convert results to ArviZ format
            posterior_data = az.from_numpyro(mcmc)

            # Determine preformance metrics 
            # Number off effective samples from total (mean - across all chains, min - across all parameters)
            eff_sample_size = az.ess(posterior_data).to_dataframe().min().min()
            # Calculate Auto-correlation length
            autocorr_length = n_samples/eff_sample_size
            # GR statisitc (mean - across all chains, min - across all parameters)
            gr_stat = az.rhat(posterior_data).to_dataframe().mean().mean()

            # Compute Time Per Effective Sample (TPES)
            tpes = sample_time / eff_sample_size if eff_sample_size > 0 else float('inf')
            # Compute Time per Iteration (TPI)
            tpi = sample_time / n_samples


            # Store results
            results.append({
                "Step Size": step_size,
                "Acceptance Probability": accept_prob,
                "Dense Mass Matrix": dense_mass,
                "Autocorrolation Length": autocorr_length,
                "Number Effective Samples": eff_sample_size,
                "GR Statistic": gr_stat,
                "Time per iteration": tpi,
                "Time per Effective Sample": tpes
                })
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Find best combination (lowest TPES)
        best_params = results_df.loc[results_df["Time per Effective Sample"].idxmin()].to_dict()

        # Save results if requested
        if save_path is not None:
            results_df.to_csv(save_path, index=False)
            logging.info(f"Results saved to {save_path}")

        logging.info(
            "\n" + "=" * 40 +
            "\n Best Hyperparameter Configuration:\n" +
            "\n".join([f"   {key}: {value}" for key, value in best_params.items()]) +
            "\n" + "=" * 40 
        )

        # Log Top Configurations with Smallest TPES
        if no_table_results > 0:
            top_configs = results_df.nsmallest(no_table_results, "Time per Effective Sample")
            logging.info("\n Top 3 Hyperparameter Configurations by Time per Effective Sample:\n" + top_configs.to_string(index=False))

        return results_df, best_params

    

    def run_hmc_optimised(self, best_params, burnin_period=2000, n_samples=4000, n_chains =  1, random_seed=0, save_path = None, traceplot=None):
        """
        Runs the HMC (Hamiltonian Monte Carlo) chain using the No-U-Turn Sampler (NUTS) with optimised hyperparameters.

        This function initializes and executes a Bayesian inference process using NUTS,
        a variant of HMC that automatically determines the optimal number of leapfrog steps.
        It tracks computational efficiency, provides optional summary statistics, and allows
        for sample storage.

        Parameters
        ----------
        best_params : dict
            The best hyperparameter combination based on the lowest Time Per Effective Sample (TPES).
        burnin_period : int, optional, default=2000
            The number of warm-up (burn-in) samples used to adapt step size and mass matrix.
        n_samples : int, optional, default=4000
            The number of posterior samples to collect after burn-in.
        random_seed : int, optional, default=0
            Random seed for reproducibility.
        save_path : str or None, optional, default=None
            If provided, saves the MCMC samples as a CSV file at the specified path.

        Returns
        -------
        mcmc : numpyro.infer.mcmc.MCMC
            The MCMC object containing all chain states, including posterior samples.
        samples_time : float
            The estimated runtime for collecting posterior samples (excluding burn-in time).

        """

        # Verify save_path
        if save_path is not None:
            if not isinstance(save_path, str):
                raise ValueError("Error: save_path must be a string.")
            if not save_path.lower().endswith(".nc"):
                raise ValueError("Error: The file is not a valid NetCDF (.nc) file.")
            
        # Filter for best result
        # Convert results to DataFrame
        results_df = pd.DataFrame(best_params)

        # Find best combination (lowest TPES)
        best_params = results_df.loc[results_df["Time per Effective Sample"].idxmin()].to_dict()

        # Extract best hyperparameters from the dictionary
        step_size = best_params["Step Size"]
        accept_prob = best_params["Acceptance Probability"]
        dense_mass = best_params["Dense Mass Matrix"]
        autocorr_length = max(1, math.ceil(best_params["Autocorrolation Length"]))

        # Calculate the total samples to run per chain based on the autocorrelation length
        total_samples_per_chain = int((n_samples * autocorr_length + 1)/n_chains)

        logging.info(f"Running MCMC with Optimised Hyperparameters: step_size={step_size}, accept_prob={accept_prob}, dense_mass={dense_mass}" 
                     + "\n" + "=" * 40 
                     + f"\nDue to Autocorrelation Length: {autocorr_length}, the number of effective samples is reduced."
                     + f"\nTotal samples required to achieve {n_samples} effective samples: {total_samples_per_chain* n_chains}"
                     + f"\nThis is run on {n_chains} parrallel chains each running {total_samples_per_chain} samples.")


        # Run MCMC with the best hyperparameters - using multiple chains
        mcmc, _ = self.run_hmc_nuts(burnin_period=burnin_period, n_samples=total_samples_per_chain, n_chains=n_chains, step_size=step_size, acceptance_prob=accept_prob, dense_mass=dense_mass, summary = False,
                                    random_seed=random_seed, save_path=save_path, traceplot=traceplot, progress_bar=True)
        # Extract samples
        posterior_data = az.from_numpyro(mcmc)

        # Determine preformance metrics before thinning as a final check - GR Statisitc, No of Effective Samples, Auto-correlation Length
        # The No. Effective samples is taken as the minimum across all parameters to be conservative - ie maximium autocorrolation length
        gr_stat = az.rhat(posterior_data).to_dataframe().mean().max()
        true_eff_samples = az.ess(posterior_data).to_dataframe().mean().min() * n_chains
        true_auto_corr = math.ceil(total_samples_per_chain / true_eff_samples * n_chains)

        logging.info(f" Over {n_chains} chains, {total_samples_per_chain*n_chains} were run, \nAchieved Gelman-Rubin Statistic: {gr_stat:.4f}, \nAchieved Effective Samples: {true_eff_samples:.0f}, \nTrue Auto-correlation Length: {true_auto_corr:.0f}")

        # Thin the samples to ensure they are not correlated - every true_auto_corr sample is kept
        thinned_posterior = posterior_data.copy()
        thinned_posterior.posterior = thinned_posterior.posterior.sel(draw=slice(None, None, true_auto_corr))

        remaining_samples = thinned_posterior.posterior.sizes["draw"] * n_chains
        logging.info(f"Total samples remaining after thinning: {remaining_samples}")
        
        # Save both the full and thinned posterior samples
        # Full can be used for plotting 
        # Thin can be used for statistical analysis
        if save_path is not None:
            thined_save_path = save_path.replace(".nc", "_thinned.nc")
            posterior_data.to_netcdf(save_path)
            thinned_posterior.to_netcdf(thined_save_path)
            logging.info(f"Full MCMC samples saved to {save_path}\n Thinned MCMC samples saved to {thined_save_path}")

        return posterior_data, thinned_posterior
    

    def thinned_hcm_analysis(self, posterior_data, summary_table=True, corner_plot=["r", "N", "sigma_r", "sigma_t"]): 
        """
        Performs analysis on posterior samples, including summary statistics and corner plots.

        **Features**
        - **Summary Table:** Computes detailed summary statistics for posterior parameters, including:
          - Median and standard deviation.
          - 68%, 90%, 95%, and 99% credible intervals.
        - **Corner Plot:** Visualizes parameter correlations using a triangle (corner) plot.

        Parameters
        ----------
        posterior_data : arviz.InferenceData or str
            The posterior samples as an **ArviZ InferenceData** object or a path to an **ArviZ-compatible NetCDF file**.
        summary_table : bool, optional, default=True
            Whether to compute and display summary statistics.
        corner_plot : list of str or None, optional, default=["r", "N", "sigma_r", "sigma_t"]
            List of parameters to include in the corner plot.
            - Set to `None` to **disable plotting**.

        Returns
        -------
        None
            Displays the summary table and plots the requested corner plot.

        Notes
        -----
        - If a **NetCDF file path** is provided, the function attempts to load both:
          - The **full posterior** (`posterior_samples.nc`).
          - The **thinned posterior** (`posterior_samples_thinned.nc`) for efficiency.
        """

        # Check if posterior_data is an InferenceData object, otherwise attempt to load it
        if not isinstance(posterior_data, az.InferenceData):
            # If a string is provided, attempt to load the NetCDF file assuming it is a path
            if isinstance(posterior_data, str) and os.path.isfile(posterior_data):
                try:
                    # Load both the full and thinned posterior samples
                    posterior_data_path = posterior_data
                    thinned_posterior_path = posterior_data_path.replace(".nc", "_thinned.nc")
                    posterior_data = az.from_netcdf(posterior_data)
                    thinned_posterior_data = az.from_netcdf(thinned_posterior_path)
                    logging.info(f"Loaded full and thinned posterior data {posterior_data_path}.")
                except Exception as e:
                    raise ValueError(f"Error loading NetCDF file: {e}")
            else:
                raise TypeError("posterior_data must be an ArviZ InferenceData object or a valid NetCDF file path.")
        else:
            thinned_posterior_data = None
        
        # ------------------ Summary Statistics ------------------
        if summary_table:
            # Define a series of functions to compute desired summary statistics
            # Allow these to be fed into the az.summary function and automatically computed
            def median_sd(x):
                """Compute the standard deviation around the median."""
                median = np.percentile(x, 50)
                return np.sqrt(np.mean((x - median) ** 2))

            # Dictionary of statistics functions passed into az.summary
            # Median, and 68%, 90%, 95%, 97.7, 99.5% credible intervals
            func_dict = {
                "std": np.std,
                "median_std": median_sd,
                "median": lambda x: np.percentile(x, 50),
                "68%_lower": lambda x: np.percentile(x, 16),
                "68%_upper": lambda x: np.percentile(x, 84),
                "90%_lower": lambda x: np.percentile(x, 5),
                "90%_upper": lambda x: np.percentile(x, 95),
                "95%_lower": lambda x: np.percentile(x, 2.5),
                "95%_upper": lambda x: np.percentile(x, 97.5),
                "99%_lower": lambda x: np.percentile(x, 0.5),
                "99%_upper": lambda x: np.percentile(x, 99.5),
            }

            # Compute summary statistics using the defined functions
            # If a thinned dataset is available, use that for the summary otherwise use the full dataset
            if thinned_posterior_data is not None:
                summary_df = az.summary(thinned_posterior_data, stat_funcs=func_dict, extend=True)
            else:
                summary_df = az.summary(posterior_data, stat_funcs=func_dict, extend=True)

            # Convert percentile thresholds to deviations from the median
            for ci in ["68%", "90%", "95%", "99%"]:
                summary_df[f"{ci}_upper"] = summary_df[f"{ci}_upper"] - summary_df["median"]
                summary_df[f"{ci}_lower"] = summary_df["median"] - summary_df[f"{ci}_lower"]

            # Format confidence intervals + upper, - lower for each percentile
            for ci in ["68%", "90%", "95%", "99%"]:
                summary_df[ci] = summary_df.apply(
                    lambda row: f"+{row[f'{ci}_upper']:.4f} − {abs(row[f'{ci}_lower']):.4f}",
                    axis=1
                )

            # Keep only formatted columns
            summary_df = summary_df[["median", "68%", "90%", "95%", "99%"]]
            summary_df.index.name = "parameter"

            # Warn about section indexes not being comparable because they were reindexed on import
            logging.info('NOTE: When comparing the results to the paper, the section indexes are not comparable, but there order is')
            display(summary_df)

        # ------------------ Corner Plot ------------------
        if corner_plot is not None:
            # Flatten chains - each parameter is a single row
            # For plotting use the full (unthinned) dataset
            flattened_samples = posterior_data.posterior.stack(sample=("chain", "draw"))

            # Define custom LaTeX labels for parameters for need plot formatting
            param_labels = {
                "N": "$N$",
                "r": "$r$",
                "sigma_r": r"$\sigma_r$",
                "sigma_t": r"$\sigma_t$",
            }

            # Extract parameters in requested list into a NumPy array
            try:
                samples_array = np.column_stack(
                    [flattened_samples[var].values.flatten() for var in corner_plot if var in flattened_samples]
                )
            except KeyError as e:
                raise KeyError(f"One or more requested parameters are not in the dataset: {e}")

            if samples_array.shape[1] == 0:
                logging.warning("No valid parameters found for the corner plot.")
                return

            # Create corner plot with formatted labels and larger text
            figure = corner.corner(
                samples_array, 
                labels=[param_labels.get(var, var) for var in corner_plot], 
                show_titles=True,
                title_fmt=".2f",
                title_kwargs={"fontsize": 16}, 
                label_kwargs={"fontsize": 20},  
                plot_datapoints=True,
                plot_contours=True,
                plot_density=True,
            )
            for ax in figure.get_axes():
                ax.xaxis.set_tick_params(labelsize=14)
                ax.yaxis.set_tick_params(labelsize=14)
            figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.1, hspace=0.1)
            plt.show()

        return None
    
    def plot_credible_intervals(self, posterior_data, param="N", percentiles_range=(50, 100)):
        """
        Plots credible intervals for a parameter across different confidence levels.

        This function extracts posterior samples for a specified parameter and visualizes its 
        distribution across various credible interval levels.

        Parameters
        ----------
        posterior_data : arviz.InferenceData or str
            The MCMC posterior samples (or path to a NetCDF file).
        param : str, optional, default="N"
            The parameter name to compute credible intervals for.
        percentiles_range : tuple, optional, default=(50, 100)
            The percentile range (e.g., `(50, 100)`) over which to compute credible intervals.

        Returns
        -------
        None
            Displays a plot of credible intervals.

        Notes
        --------
        - Uses **thinned posterior samples** if available for better efficiency.
        - The plot **shades the full credible interval region**, with upper and lower bounds shown.
        """

        # Mapping of parameter names to LaTeX-formatted labels
        label_map = {
            "N": r"$N$ (Number of Holes)",
            "r": r"$r$ (Radius)",
            "sigma_r": r"$\sigma_r$ (Radial Error)",
            "sigma_t": r"$\sigma_t$ (Tangential Error)",
            "sigma": r"$\sigma$ (Isotropic Error)",
        }

        # Check if posterior_data is an InferenceData object, otherwise attempt to load it
        if not isinstance(posterior_data, az.InferenceData):
            # If a string is provided, attempt to load the NetCDF file assuming it is a path
            if isinstance(posterior_data, str) and os.path.isfile(posterior_data):
                try:
                    # Load both the full and thinned posterior samples
                    posterior_data_path = posterior_data
                    thinned_posterior_path = posterior_data_path.replace(".nc", "_thinned.nc")
                    posterior_data = az.from_netcdf(posterior_data)
                    thinned_posterior_data = az.from_netcdf(thinned_posterior_path)
                    logging.info(f"Loaded full and thinned posterior data {posterior_data_path}.")
                except Exception as e:
                    raise ValueError(f"Error loading NetCDF file: {e}")
            else:
                raise TypeError("posterior_data must be an ArviZ InferenceData object or a valid NetCDF file path.")
        else:
            thinned_posterior_data = None

        
        # Generate array of desired percentile calculations (e.g., 50% to 100%)
        percentiles = np.linspace(percentiles_range[0], percentiles_range[1], percentiles_range[1] - percentiles_range[0])

        # Extract parameter for chosen param samples & flatten across chains - use thinned data if available
        if thinned_posterior_data is not None and param in thinned_posterior_data.posterior:
            param_samples = thinned_posterior_data.posterior[param].stack(sample=("chain", "draw")).values
        else:
            param_samples = posterior_data.posterior[param].stack(sample=("chain", "draw")).values

        # Compute credible intervals - both the lower and upper bounds
        lower_bounds = np.percentile(param_samples, [(100 - p) / 2 for p in percentiles])
        upper_bounds = np.percentile(param_samples, [100 - (100 - p) / 2 for p in percentiles])

        # Plot credible intervals
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot the full credible interval and shade the area
        ax.plot(percentiles, upper_bounds, color="red", label="Full")
        ax.plot(percentiles, lower_bounds, color="red")
        ax.fill_between(percentiles, lower_bounds, upper_bounds, color="red", alpha=0.3)

        y_label = label_map.get(param, param)
        ax.set_xlabel("Credible Interval Level (%)", fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.show()

        return None
    


    def plot_posterior_holes(self, posterior_data, hole_no=1, section_no=5):
        """
        Plots the posterior distribution of hole positions based on MCMC samples.

        Parameters
        ----------
        posterior_data : arviz.InferenceData or str
            The MCMC posterior samples as an ArviZ InferenceData object or a path to a NetCDF file.
        hole_no : int, optional
            The hole number to analyze.
        section_no : int, optional
            The section number to analyze.

        Returns
        -------
        None
            Displays a plot of credible intervals.
        """
        if not jnp.any((self.hole_nos_obs == hole_no) & (self.section_ids_obs == section_no)):
            raise ValueError("Error: The hole number and section number combination is not in the observed data.")

        if not isinstance(posterior_data, az.InferenceData):
            if isinstance(posterior_data, str) and os.path.isfile(posterior_data):
                try:
                    posterior_data_path = posterior_data
                    thinned_posterior_path = posterior_data_path.replace(".nc", "_thinned.nc")
                    posterior_data = az.from_netcdf(posterior_data)
                    if os.path.isfile(thinned_posterior_path):
                        thinned_posterior_data = az.from_netcdf(thinned_posterior_path)
                        mcmc_params = thinned_posterior_data.posterior
                        logging.info("Loaded thinned posterior data from NetCDF file.")
                    else:
                        mcmc_params = posterior_data.posterior
                        thinned_posterior_data = None
                    logging.info("Loaded posterior data from NetCDF file.")
                except Exception as e:
                    raise ValueError(f"Error loading NetCDF file: {e}")
            else:
                raise TypeError("posterior_data must be an ArviZ InferenceData object or a valid NetCDF file path.")
        else:
            thinned_posterior_data = None
            mcmc_params = posterior_data.posterior

        param_names = list(mcmc_params.keys())
        all_param_samples = {param: mcmc_params[param].stack(sample=("chain", "draw")).values for param in param_names}

        N_samples = jnp.array(all_param_samples['N'])
        r_samples = jnp.array(all_param_samples['r'])
        x0_samples = jnp.array(all_param_samples['x0']).T
        y0_samples = jnp.array(all_param_samples['y0']).T
        alpha_samples = jnp.array(all_param_samples['alpha']).T

        # --- Compute MCMC hole positions ---
        mcmc_hole_locations = []
        for i in range(len(N_samples)):
            hole_position = self.hole_positions(
                float(N_samples[i]), float(r_samples[i]),
                x0_samples[i], y0_samples[i], alpha_samples[i],
                section_ids=section_no, hole_nos=hole_no
            )
            mcmc_hole_locations.append(hole_position)

        mcmc_hole_locations = jnp.array(mcmc_hole_locations).T
        x_post_samples, y_post_samples = mcmc_hole_locations[0], mcmc_hole_locations[1]

        x_mean, y_mean = jnp.mean(x_post_samples), jnp.mean(y_post_samples)
        x_std, y_std = jnp.std(x_post_samples), jnp.std(y_post_samples)

        # --- Radial and tangential uncertainties (if anisotropic model) ---
        if self.model_type == "anisotropic":
            dx = x_post_samples - x_mean
            dy = y_post_samples - y_mean

            theta = alpha_samples[:, section_no - 1] if alpha_samples.ndim > 1 else alpha_samples
            cos_theta = jnp.cos(theta)
            sin_theta = jnp.sin(theta)

            dr = dx * cos_theta + dy * sin_theta
            dt = -dx * sin_theta + dy * cos_theta

            sigma_r = jnp.std(dr)
            sigma_t = jnp.std(dt)

            theta_mean = jnp.mean(theta)

            u_r = jnp.array([jnp.cos(theta_mean), jnp.sin(theta_mean)])
            u_t = jnp.array([-jnp.sin(theta_mean), jnp.cos(theta_mean)])

            delta_r = sigma_r * u_r
            delta_t = sigma_t * u_t

        # --- Get observed hole position ---
        hole_obs_index = jnp.where((self.hole_nos_obs == hole_no) & (self.section_ids_obs == section_no))
        x_obs = self.x_obs[hole_obs_index]
        y_obs = self.y_obs[hole_obs_index]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(x_post_samples, y_post_samples, s=5, color="blue", alpha=0.3, label="MCMC Samples")
        ax.scatter(x_obs, y_obs, color="red", s=100, edgecolor="black", label="Observed Hole")

        if self.model_type == "isotropic":
            ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt="rx", markersize=18, elinewidth=2, label=r"Mean Posterior ($\pm1\sigma$)")

        # Add radial/tangential error bars if anisotropic
        if self.model_type == "anisotropic":
            ax.errorbar(x_mean, y_mean, fmt="rx", markersize=18, label=r"Mean Posterior ($\pm1\sigma_r$, $\pm1\sigma_t$)")
            ax.plot([x_mean - delta_r[0], x_mean + delta_r[0]],
                    [y_mean - delta_r[1], y_mean + delta_r[1]],
                    color="red", lw=2)

            ax.plot([x_mean - delta_t[0], x_mean + delta_t[0]],
                    [y_mean - delta_t[1], y_mean + delta_t[1]],
                    color="red", lw=2)

        ax.set_xlabel("X Position", fontsize=15)
        ax.set_ylabel("Y Position", fontsize=15)
        ax.set_title(f"Posterior Hole Positions\n(Hole {hole_no}, Section {section_no})", fontsize=16)
        ax.legend(fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.show()

        return mcmc_hole_locations


######## THIS IS NESTED SAMPLING ########


    def ns_prior_transform(self, u):
        """
        Efficiently transforms unit hypercube samples u ~ U[0,1] into samples from
        the actual prior distributions defined in `self.priors`.

        This function supports a wide variety of prior distributions and efficiently
        transforms unit cube samples into the actual prior range.

        This has been written to support a wide range of prior distributions in the future

        - **Vectorized operations** where possible.
        - **Supports arbitrary prior distributions** dynamically from `self.priors`.
        - **Handles both scalar and section-dependent (vector) parameters**.
        - **Avoids unnecessary loops** for efficiency.

        Parameters
        ----------
        u : np.ndarray
            A 1D array of unit cube values (random samples from [0,1]).

        Returns
        -------
        np.ndarray
            Transformed parameters in the actual prior range.
        """

        # Ensure `u` is a numpy array
        x = np.array(u)

        param_names = list(self.priors.keys())  # Extract parameter names
        u_idx = 0  # Index tracker for `u`
        transformed_values = []

        for param in param_names:
            prior = self.priors[param]

            # Determine size: `num_sections` for section-based parameters, else scalar
            size = self.num_sections if param in ["x0", "y0", "alpha"] else 1

            u_subset = x[u_idx : u_idx + size]  

            # Apply transformation based on the prior type
            if isinstance(prior, dist.Uniform):
                values = prior.low + u_subset * (prior.high - prior.low)
            elif isinstance(prior, dist.Normal):
                values = scipy.stats.norm.ppf(u_subset, loc=prior.loc, scale=prior.scale)
            elif isinstance(prior, dist.Gamma):
                values = scipy.stats.gamma.ppf(u_subset, a=prior.concentration, scale=prior.rate)
            elif isinstance(prior, dist.Beta):
                values = scipy.stats.beta.ppf(u_subset, a=prior.concentration1, b=prior.concentration0)
            elif isinstance(prior, dist.Exponential):
                values = -np.log(1 - u_subset) / prior.rate 
            elif isinstance(prior, dist.LogUniform):
                log_low = np.log(prior.low)
                log_high = np.log(prior.high)
                values = np.exp(log_low + u_subset * (log_high - log_low))
            else:
                raise ValueError(f"Unsupported prior type for parameter {param}: {type(prior)}")

            transformed_values.append(values)
            u_idx += size  # Move index forward

        return np.concatenate(transformed_values)
    
    def ns_log_likelihood(self, transformed_values):
        """
        Computes the log-likelihood of the model given the transformed parameters.

        Parameters
        ----------
        transformed_values : np.ndarray
            Transformed parameters in the actual prior range.

        Returns
        -------
        float
            The log-likelihood of the model given the parameters.
        """

        # Unpack the transformed parameters
        N_packed = transformed_values[0]
        r_packed = transformed_values[1]
        x0_packed = transformed_values[2:2 + self.num_sections]
        y0_packed = transformed_values[2 + self.num_sections:2 + 2 * self.num_sections]
        alpha_packed = transformed_values[2 + 2 * self.num_sections: 2 + 3 * self.num_sections]
        if self.model_type == 'anisotropic':
            sigma = transformed_values[2 + 3 * self.num_sections:]
        else:
            sigma = transformed_values[-1]

        # Compute the log-likelihood using predefined function
        log_likelihood = self.likelihood(N_packed, r_packed, x0_packed, y0_packed, alpha_packed, sigma, log=True, neg = False, data = None)

        return log_likelihood

    def run_nested_sampling(self, num_live=500, max_iter=1000, tol=1e-3, seed=0, save_path = None):
        r"""
        Implements a custom Nested Sampling algorithm that stores the necessary
        quantities for plotting and evidence estimation. This implementation
        operates in log-space to prevent numerical underflow when computing
        Bayesian evidence.

        **Mathematical Background**

        Nested Sampling iteratively removes the worst live point (lowest likelihood)
        and replaces it with a new sample from the constrained prior.

        The Bayesian evidence (marginal likelihood) is given by:

            .. math::
                Z = \int L(\mathbf{\theta}) \pi(\mathbf{\theta}) d\mathbf{\theta}

        where:
            - :math:`L(\mathbf{\theta})` is the likelihood function,
            - :math:`\pi(\mathbf{\theta})` is the prior distribution.

        Since direct numerical integration is infeasible, we estimate the evidence
        iteratively using the sum:

            .. math::
                Z \approx \sum w_i L_i

        where:
            - :math:`w_i = X_i - X_{i+1}` are the **prior mass shrinkage weights**,
            - :math:`L_i` are the likelihood values of removed (dead) points.

        **Log-space Transformation**

        Since the weights :math:`w_i` and likelihoods :math:`L_i` can be **very small**,
        we work in log-space:

            .. math::
                \log Z = \log \sum \exp(\log L_i + \log w_i)

        To update log-evidence iteratively, we use the numerically stable `logaddexp`:

            .. math::
                \log Z_{\text{new}} = \text{logaddexp}(\log Z, \log L_{\text{dead}} + \log X)

        **Uncertainty Estimation**

        The uncertainty in log-evidence is approximated as:

            .. math::
                \text{Var}(\log Z) \approx \sum (w_i^2 \cdot \exp(2 \cdot \log L_i))

        Since directly computing :math:`\exp(2 \log L_i)` causes numerical underflow,
        we keep everything in **log-space**:

            .. math::
                \log Z_{\text{err}} = \text{logaddexp}(\log Z_{\text{err}}, 2 \cdot \log L_i + 2 \cdot \log w_i)

        Finally, the uncertainty is extracted as:

            .. math::
                \log Z_{\text{err}} = \exp(0.5 \cdot \log Z_{\text{err}})

        Parameters
        ----------
        num_live : int, optional, default=500
            Number of live points used in nested sampling.
        max_iter : int, optional, default=1000
            Maximum number of iterations before termination.
        tol : float, optional, default=1e-3
            Convergence tolerance for stopping criterion.
        seed : int, optional, default=0
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary containing:
            - `"logZ"` : Estimated log evidence.
            - `"logZ_err"` : Estimated uncertainty in log evidence.
            - `"samples"` : List of accepted posterior samples.
            - `"logLs"` : Log-likelihoods of sampled points.
            - `"log_Xs"` : Log prior mass shrinkage (for ξ plots).
            - `"weights"` : Importance weights of samples.
            - `"logZs"` : History of log-evidence estimates.
        """

        # Initialize random key for reproducibility
        key = jax.random.PRNGKey(seed)

        # Number of dimensions for sampling
        num_dims = 2 + 3 * self.num_sections + (1 if self.model_type == "isotropic" else 2)

        # Initialize live points
        key, subkey = jax.random.split(key)
        unit_live_points = jax.random.uniform(subkey, shape=(num_live, num_dims))
        live_points = np.array([self.ns_prior_transform(u) for u in unit_live_points])
        live_log_likelihoods = np.array([self.ns_log_likelihood(p) for p in live_points])

        # Initialize nested sampling quantities
        log_Xs = [0]  # Log prior mass shrinkage
        logZ = -np.inf  # Log-evidence estimate initialized to log(0)
        logZ_err = -np.inf  # Store log-variance in log-space
        weights = []
        logLs = []
        samples = []
        logZs = []  # Store log-evidence history
        logZ_errs = []

        # Initialize tqdm progress bar
        pbar = tqdm(total=max_iter, desc="Running Nested Sampling", unit="iter")

        # Nested Sampling Loop
        for iteration in range(max_iter):
            # Find the worst live point (lowest likelihood)
            min_idx = np.argmin(live_log_likelihoods)
            logL_dead = live_log_likelihoods[min_idx]  # Log-likelihood of removed point
            logLs.append(logL_dead)

            # Store the removed dead point
            dead_point = live_points[min_idx]
            samples.append(dead_point)

            # Compute prior mass shrinkage in log-space
            log_X = -iteration / num_live  # Shrinking prior volume
            log_Xs.append(log_X)

            # Compute weight in log-space: w = exp(log_Xs[-2]) - exp(log_Xs[-1])
            weight = np.exp(log_Xs[-2]) - np.exp(log_Xs[-1])
            weights.append(weight)

            # Constrained prior sampling: Find a new sample with L_new > L_dead
            found_valid_sample = False
            while not found_valid_sample:
                u_new = jax.random.uniform(jax.random.PRNGKey(np.random.randint(0, 2**32)), shape=(num_dims,))
                new_sample = self.ns_prior_transform(u_new)
                new_logL = self.ns_log_likelihood(new_sample)
                if new_logL > logL_dead:
                    found_valid_sample = True

            # Replace the worst live point
            live_points[min_idx] = new_sample
            live_log_likelihoods[min_idx] = new_logL

            # Update log-evidence using log-space summation:
            new_logZ = np.logaddexp(logZ, logL_dead + log_X)
            # Store log-evidence history:
            logZs.append(new_logZ)  

            # Convergence check: Stop if logZ stabilizes
            if abs(new_logZ - logZ) < tol:
                break

            # Update log-evidence
            logZ = new_logZ  

            # Update log-evidence uncertainty in log-space:
            # logZ_err = logaddexp(logZ_err, 2 * logL_dead + 2 * log(weight))
            logZ_err = np.logaddexp(logZ_err, 2 * logL_dead + 2 * np.log(weight))
            logZ_errs.append(logZ_err)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(logZ=logZ, logL_dead=logL_dead)

        # Close progress bar
        pbar.close()

        # Finalize uncertainty estimate: convert log-variance back to normal space
        logZ_errs = (-1/2*np.array(logZ_errs))
        logZ_errs[np.isinf(logZ_errs)] = 0

        results_dict = {
            "logZ": logZ,
            "logZ_err": logZ_err,
            "samples": samples,
            "weights": weights,
            "logLs": logLs,
            "log_Xs": log_Xs,
            "logZs": logZs,  
            "logZ_errs": logZ_errs
        }
        
        if save_path is not None:
            with open(save_path, "wb") as f:
                pickle.dump(results_dict, f)
            logging.info(f"Nested Sampling results saved to {save_path}")

        return results_dict

    def plot_nested_sampling(self, nested_sampling_results):
        r"""
        Generates plots for Nested Sampling results to assess convergence and sampling behavior.

        **Generated Plots**

        1. **Log-Evidence (logZ) Convergence Plot:**
           - Tracks the evolution of the estimated log-evidence (`logZ`) over iterations.

        2. **Log-Likelihood Evolution Plot:**
           - Displays how the log-likelihood (`logL`) of removed (dead) points evolves over iterations.

        3. **Log-Likelihood vs Log Prior Mass Shrinkage Plot (ξ plot):**
           - Plots log-likelihood values (`logL`) against the shrinking prior mass (`X`).

        Parameters
        ----------
        nested_sampling_results : dict
            Output from the nested sampling algorithm, containing:
            - `"logZs"` : list or ndarray
                Estimated log-evidence values at each iteration.
            - `"logZ_errs"` : list or ndarray
                Uncertainty estimates associated with log-evidence values.
            - `"logLs"` : list or ndarray
                Log-likelihood values of discarded (dead) points.
            - `"log_Xs"` : list or ndarray
                Log prior mass shrinkage values over iterations.

        """

        # Set up global style for high-quality plots
        plt.rcParams.update({
            "text.usetex": True,  # Use LaTeX for text rendering
            "font.size": 16,  # General font size
            "axes.labelsize": 18,  # Axis labels
            "axes.titlesize": 20,  # Title size
            "xtick.labelsize": 16,  # X-axis tick labels
            "ytick.labelsize": 16,  # Y-axis tick labels
            "legend.fontsize": 14,  # Legend font size
            "lines.linewidth": 2.5,  # Line thickness
            "axes.grid": True,  # Enable grid
            "grid.linestyle": "--",  # Dashed grid lines
            "grid.alpha": 0.6,  # Grid transparency
        })

        # Extract data
        iterations = np.arange(len(nested_sampling_results["logZs"]))
        logZs = np.array(nested_sampling_results["logZs"])
        logZ_errs = np.array(nested_sampling_results["logZ_errs"])
        logLs = np.array(nested_sampling_results["logLs"])
        log_Xs = np.array(nested_sampling_results["log_Xs"])

        # # Ensure no NaNs/Infs in errors
        # logZ_errs[np.isinf(logZ_errs) | np.isnan(logZ_errs)] = 0

        # --- 1. Log-Evidence (logZ) Convergence Plot with Shaded Uncertainty ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(iterations, logZs, linestyle="-", color="black", label=r"$\log Z$ Estimate")
        # ax.fill_between(iterations, logZs - logZ_errs, logZs + logZ_errs, color="gray", alpha=0.3, label=r"$\pm 1\sigma$ Uncertainty")
        ax.set_xlabel(r"Iteration")
        ax.set_ylabel(r"$\log Z$")
        ax.set_yscale('symlog')
        ax.set_title(r"Convergence of Log-Evidence $\log Z$")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.show()

        # --- 2. Log-Likelihood Evolution Plot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(iterations, logLs, linestyle="-", color="red", label=r"$\log L$ of Dead Points")
        ax.set_xlabel(r"Iteration")
        ax.set_ylabel(r"$\log L$")
        ax.set_yscale('symlog')
        ax.set_title(r"Evolution of Log-Likelihood $\log L$")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.show()

        # --- 3. Log-Likelihood vs Log Prior Mass Plot ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.exp(log_Xs[1:]), logLs, linestyle="-", color="blue", label=r"$\log L$ of Dead Points")
        ax.set_xlabel(r"Prior Mass Shrinkage $X$")
        ax.set_ylabel(r"$\log L$")
        ax.set_xscale('log')
        # Determine max of y data
        y_max_data = max(logLs)
        ax.set_ylim(bottom=None, top=y_max_data)
        ax.set_yscale('symlog')
        ax.set_title(r"Likelihood Evolution with Prior Mass")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.show()



########## Savage Dickey Comparisom ##########

    
    def savage_dickey_comparison(self, best_params, burnin_period=2000, n_samples_posterior=10000, n_samples_prior=10000, random_seed=0, show_plots=True, random_key=None):
        r"""
        Performs the Savage-Dickey density ratio test for comparing nested models.

        This method estimates the posterior and prior probability densities conditioned at
        `sigma_t = sigma_r` to compute the Savage-Dickey ratio.

        **Method Overview**

        1. **Posterior Sampling:**
           - Runs an optimized **HMC MCMC chain** to draw posterior samples.
           - Extracts the **posterior samples** for `sigma_t` and `sigma_r`.
           - Computes transformed coordinates:
             - :math:`X = \sigma_t - \sigma_r`
             - :math:`Y = \sigma_t + \sigma_r`

        2. **Prior Sampling:**
           - Generates samples from the **prior distribution**.
           - Computes the same transformed coordinates as for the posterior.

        3. **Kernel Density Estimation (KDE) & Probability Calculation:**
           - Uses **1D KDE** to estimate the probability density at :math:`X = 0`.
           - Computes:
             - :math:`P_{\text{posterior}}(\sigma_t \approx \sigma_r)`
             - :math:`P_{\text{prior}}(\sigma_t \approx \sigma_r)`
           - Computes the **Savage-Dickey ratio**:
             .. math::
                R_{SD} = \frac{P_{\text{posterior}}(X \approx 0)}{P_{\text{prior}}(X \approx 0)}

        4. **Visualizations (Optional, Controlled by `show_plots`):**
           - **Corner Plots**:
             - Visualizes :math:`(X, Y)` distributions for both **posterior** and **prior**.
           - **KDE Contour + 1D KDE Plots (Side-by-Side)**:
             - **Left:** 2D KDE of :math:`(X, Y)`.
             - **Right:** 1D KDE of :math:`X` (with probability at :math:`X=0` marked).

        Parameters
        ----------
        best_params : array-like
            Best-fit parameters to initialize HMC.
        burnin_period : int, optional (default=2000)
            Number of burn-in samples to discard.
        n_samples_posterior : int, optional (default=10000)
            Number of posterior samples to draw.
        n_samples_prior : int, optional (default=10000)
            Number of prior samples to draw.
        random_seed : int, optional (default=0)
            Seed for reproducibility.
        tol : float, optional (default=0.01)
            Defines the **small window width** for estimating
            :math:`P(\sigma_t - \sigma_r \approx 0)`.
        show_plots : bool, optional (default=True)
            If `True`, generates:
            - **Corner Plots**
            - **KDE Contour + 1D KDE Plots**
        random_key : int, optional (default=None)
            JAX random key for reproducibility.

        Returns
        -------
        dict
            Dictionary containing:
            - `"posterior_probability"` : Estimated posterior density at :math:`X = 0`.
            - `"prior_probability"` : Estimated prior density at :math:`X = 0`.
            - `"savage_dickey_ratio"` : Ratio of posterior to prior probability densities.
        """
        # Ensure we are using the anisotropic model
        if self.model_type == "isotropic":
            raise ValueError("Savage-Dickey comparison is only available for the anisotropic model class.")

        if random_key is None:
            key = jax.random.PRNGKey(42)
        else:
            key = jax.random.PRNGKey(random_key)

        logging.info(f"Running HMC to generate posterior samples, aiming for {n_samples_posterior}...")
        
        # Run HMC sampling for posterior
        _, thinned_parameters = self.run_hmc_optimised(best_params, burnin_period=burnin_period, n_samples=n_samples_posterior,
                                                    n_chains=4, random_seed=random_seed, save_path=None, traceplot=None)

        # Tolerance so can see line
        tol=0.01

        # Extract posterior samples
        sigma_t_post = thinned_parameters.posterior["sigma_t"].values.flatten()
        sigma_r_post = thinned_parameters.posterior["sigma_r"].values.flatten()

        # Transform posterior samples into (X_post, Y_post)
        X_post = sigma_t_post - sigma_r_post
        Y_post = sigma_t_post + sigma_r_post
        samples_post = np.vstack([X_post, Y_post]).T

        # Compute posterior sample range
        x_min_post, x_max_post = X_post.min(), X_post.max()
        y_min_post, y_max_post = Y_post.min(), Y_post.max()
        x_extended_min_post = min(x_min_post, -tol)
        x_extended_max_post = max(x_max_post, tol)

        # ------------------ Generate Prior Samples ------------------
        logging.info(f"Generating {n_samples_prior} prior samples ...")
        prior_samples = np.array(self.sample_from_priors(key, num_samples=n_samples_prior)['sigma']).T
        sigma_r_prior = prior_samples[0]
        sigma_t_prior = prior_samples[1]

        # Transform prior samples into (X_prior, Y_prior)
        X_prior = sigma_t_prior - sigma_r_prior
        Y_prior = sigma_t_prior + sigma_r_prior
        samples_prior = np.vstack([X_prior, Y_prior]).T

        # Compute prior sample range
        x_min_prior, x_max_prior = X_prior.min(), X_prior.max()
        y_min_prior, y_max_prior = Y_prior.min(), Y_prior.max()
        x_extended_min_prior = min(x_min_prior, -tol)
        x_extended_max_prior = max(x_max_prior, tol)

        # ------------------ Compute KDE Probability ------------------
        def probability_kde(samples):
            """
            Computes the estimated probability density at x = 0 using 1D KDE.
            """
            kde = gaussian_kde(samples)
            return kde(0)[0]


        prob_post = probability_kde(X_post)
        logging.info(f"Posterior Conditional Probability, Prior(sigma_r = sigma_t), generated to be: {prob_post:.5f}")
        prob_prior = probability_kde(X_prior)
        logging.info(f"Prior Conditional Probability, Posterior(sigma_r = sigma_t), generated to be: {prob_prior:.5f}")
        savage_dickey_ratio = prob_post / prob_prior if prob_prior > 0 else np.nan
        logging.info(f"Savage-Dickey Ratio: {savage_dickey_ratio:.5f}")

        # ------------------ Plotting Configuration ------------------

        config = {"figsize_corner": (6, 4), "figsize_kde": (4.5, 4.5), "font_size": 15, "tick_size": 14,"hist_color": "navy",
        "hist_alpha": 0.7, "cmap": "coolwarm", "kde_levels": 50, "tol": tol, "line_color": "red", "line_style": "dashed",
        "line_alpha": 0.7, "legend_loc": "best", "colorbar_label": "Density Estimate"}

        # ------------------ Generate Corner Plots ------------------
        def plot_corner(samples, x_min, x_max, y_min, y_max, title):
            fig = corner.corner(
                samples,
                labels=[r"$\sigma_t - \sigma_r$", r"$\sigma_t + \sigma_r$"],
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt=".2f",
                label_kwargs={"fontsize": config["font_size"]},
                title_kwargs={"fontsize": config["font_size"]},
                tick_kwargs={"labelsize": config["tick_size"]},
                hist_kwargs={"color": config["hist_color"], "alpha": config["hist_alpha"]},
                figsize=config["figsize_corner"]
            )

            # Extract axes for modifying
            axes_corner = np.array(fig.axes).reshape(2, 2)

            # Overlay slice lines
            ax = axes_corner[0, 0]  
            ax.axvline(0, color=config["line_color"], linestyle=config["line_style"], alpha=config["line_alpha"])
            ax.set_xlim(x_min, x_max)

            ax = axes_corner[1, 0]
            ax.axvline(0, color=config["line_color"], linestyle=config["line_style"], alpha=config["line_alpha"], lw=2)
            ax.set_xlim(x_min, x_max)


            plt.show()

        if show_plots:
            plot_corner(samples_post, x_extended_min_post, x_extended_max_post, y_min_post, y_max_post, "Posterior Sample Distribution")
            plot_corner(samples_prior, x_extended_min_prior, x_extended_max_prior, y_min_prior, y_max_prior, "Prior Sample Distribution")

        # ------------------ Generate KDE Contour Plots ------------------

        def plot_kde(samples, x_min, x_max, y_min, y_max, title):
            kde_2d = gaussian_kde(samples.T)

            # Create a grid for KDE visualization
            x_grid = np.linspace(x_min, x_max, 200)
            y_grid = np.linspace(y_min, y_max, 200)
            X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
            density_values = kde_2d(np.vstack([X_mesh.ravel(), Y_mesh.ravel()])).reshape(X_mesh.shape)

            # Extract 1D KDE data (σ_t - σ_r)
            X_samples = samples[:, 0]  # First column (σ_t - σ_r)
            kde_1d = gaussian_kde(X_samples)
            x_grid_1d = np.linspace(x_min, x_max, 200)
            density_1d = kde_1d(x_grid_1d)

            # Create side-by-side plots
            fig, axes = plt.subplots(1, 2, figsize=(2 * config["figsize_kde"][0], config["figsize_kde"][1]))

            # -------------------- 2D KDE Contour Plot --------------------
            ax = axes[0]
            contour = ax.contourf(X_mesh, Y_mesh, density_values, levels=config["kde_levels"], cmap=config["cmap"])
            
            ax.axvline(0, color=config["line_color"], linestyle=config["line_style"], alpha=config["line_alpha"], lw=2, label=r"$\sigma_t - \sigma_r$ = 0")

            ax.set_xlabel(r"$\sigma_t - \sigma_r$", fontsize=config["font_size"])
            ax.set_ylabel(r"$\sigma_t + \sigma_r$", fontsize=config["font_size"])
            cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(config["colorbar_label"], fontsize=config["font_size"] * 0.7)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.legend(loc=config["legend_loc"])

            # -------------------- 1D KDE Plot --------------------
            ax = axes[1]
            ax.plot(x_grid_1d, density_1d, color="black", lw=2, label="1D KDE")
            ax.plot([0, 0], [0, kde_1d(0)[0]], color="red", linestyle='dashed', lw=2)
            ax.plot([x_min, 0], [kde_1d(0)[0], kde_1d(0)[0]], color="red", linestyle='dashed', lw=2)

            ax.set_xlabel(r"$\sigma_t - \sigma_r$", fontsize=config["font_size"])
            ax.set_ylabel("Density", fontsize=config["font_size"])
            ax.set_xlim(x_min, x_max)
            ax.legend(loc='upper left')

            plt.tight_layout()
            plt.show()

        if show_plots:
            plot_kde(samples_post, x_extended_min_post, x_extended_max_post, y_min_post, y_max_post, "Posterior KDE")
            plot_kde(samples_prior, x_extended_min_prior, x_extended_max_prior, y_min_prior, y_max_prior, "Prior KDE")

        return {"posterior_probability": prob_post, "prior_probability": prob_prior, "savage_dickey_ratio": savage_dickey_ratio}
