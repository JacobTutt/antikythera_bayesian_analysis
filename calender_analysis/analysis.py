import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import time
import tracemalloc
import logging
import sys
import jax.scipy.optimize as jso
import optax
import jax.tree_util as jtu
from tqdm.notebook import tqdm  
from IPython.display import display
from scipy.optimize import minimize

 # Set Logging level - INFO and above
logging.basicConfig(level=logging.INFO,  format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


class Calender_Analysis:
    def __init__(self, data, model_type="anisotropic", priors=None):
        """
        Initializes the Bayesian model for the calendar ring.

        Parameters
        ----------
        data : pd.DataFrame
            The observed hole position data.
        model_type : str, optional
            The error model type. Can be:
            - "isotropic": Uses a single `sigma` for errors.
            - "anisotropic" (default): Uses `sigma_r` and `sigma_t` for radial/tangential errors.
        priors : dict, optional
            A dictionary defining prior distributions for the model parameters.
            - Entries include: "N", "r", "x0", "y0", "alpha", "sigma", "sigma_r", "sigma_t
            If None, default priors are used.
        """
        self.data = self._load_data(data)
        self.model_type = model_type
        self.num_sections = len(self.data["Section ID"].unique())
        self.n_holes = len(self.data["Hole"].unique())


        #Import relevent data as class atributes store as jax arrays for efficient computation later
        self.section_ids_obs = jnp.array(self.data["Section ID"].values)
        self.hole_nos_obs = jnp.array(self.data["Hole"].values)
        self.x_obs = jnp.array(self.data["Mean(X)"].values)
        self.y_obs = jnp.array(self.data["Mean(Y)"].values)


        # Validate model type
        if model_type not in ["anisotropic", "isotropic"]:
            raise ValueError("Invalid model_type. Choose either 'anisotropic' or 'isotropic'.")
        self.model_type = model_type

        # Default priors if none provided
        default_priors = {
            "N": dist.Uniform(300, 400),
            "r": dist.Uniform(50, 100),
            "x0": dist.Normal(80, 10),
            "y0": dist.Normal(135, 10),
            "alpha": dist.Normal(jnp.pi, jnp.pi / 6),
            "sigma": dist.Exponential(0.5),  # For isotropic
            "sigma_r": dist.Exponential(0.5),  # For anisotropic
            "sigma_t": dist.Exponential(0.5),  # For anisotropic
        }

        # Use user-defined priors if provided, otherwise use default priors
        self.priors = default_priors if priors is None else {**default_priors, **priors}


    def _load_data(self, data):
        """
        Load the hole position data from various formats, convert it into a Pandas DataFrame, 
        and ensure it is properly sorted by hole number.

        This method supports input in the form of:
        - A file path pointing to a CSV, TXT, or DAT file.
        - A Pandas DataFrame containing the hole position data.
        - A NumPy array, which will be converted into a Pandas DataFrame.

        The dataset must contain the following required columns:
        - "Section ID": Identifies the section of the calendar ring.
        - "Hole": The hole index number.
        - "Inter-hole Distance": Distance between consecutive holes.
        - "Mean(X)": X-coordinate of the hole's mean position.
        - "Mean(Y)": Y-coordinate of the hole's mean position.

        Parameters
        ----------
        data : str, pd.DataFrame, or np.ndarray
            The input data, which can be a file path, a Pandas DataFrame, or a NumPy array.

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame containing the loaded and sorted hole position data.

        Raises
        ------
        ValueError
            If the required columns are missing or if the file format is unsupported.
        TypeError
            If the input data format is not one of the supported types.
        RuntimeError
            If there is an issue reading the file.
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
        
        return df
    
    def plot_hole_locations(self, figsize=(10, 8)):
        """
        Plots the measured (mean) hole locations in the x-y plane.
        
        This function visualizes the hole positions (`Mean(X)`, `Mean(Y)`) from the dataset, 
        color-coded by `Section ID`. It also includes annotations for hole numbers at regular intervals 
        and adds perpendicular bisecting lines at section transitions.

        Features:
        - **Color-coded markers**: Each section ID is assigned a unique color.
        - **Different marker styles**: Cycles between circle, square, and triangle markers.
        - **Annotations**: Labels every third hole with its hole number.
        - **Section transition markers**: Red dashed perpendicular bisecting lines indicate section changes.
        - **Multiple legends**:
          - Section ID legend.
          - Bisector legend listing splits with corresponding hole numbers.

        Parameters
        ----------
        figsize : tuple, optional
            The figure size in inches, default is (10, 8).

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

    def hole_positions(self, N, r, x0, y0, alpha, section_ids = None , hole_nos = None):
        """
        Compute the expected positions of the holes on the calendar ring based on the model parameters.

        This function calculates the predicted x and y coordinates of holes based on a circular 
        calendar ring model. The model assumes:
        - The ring is circular with radius `r`.
        - The ring originally had `N` evenly spaced holes before fragmentation.
        - Each section has a relative offset `(x0, y0)` from the center and a rotation `alpha`.

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
        section_ids : jnp.ndarray, optional
            Section IDs for each hole being modeled (default: uses stored dataset values).
        hole_nos : jnp.ndarray, optional
            Hole numbers for each hole being modeled (default: uses stored dataset values).

        Returns
        -------
        jnp.ndarray
            A 2D array of shape `(N_holes, 2)`, where each row contains the expected (x, y) position 
            of a hole based on the model parameters.

        Raises
        ------
        ValueError
            If `section_ids` and `hole_nos` are not the same length.
            If `N` is not a positive integer.
            If `r` is not a positive float.
            If `x0`, `y0`, and `alpha` do not match the number of sections.
        """


        # ---- Input Validation ---- #
        if section_ids.shape != hole_nos.shape:
            raise ValueError(f"section_ids and hole_nos must have the same shape, got {section_ids.shape} and {hole_nos.shape}.")

        if not isinstance(N, (float, jnp.floating)) or N <= 0:
            raise ValueError(f"N must be a positive integer, got {N}.")

        if not isinstance(r, (float, jnp.floating)) or r <= 0:
            raise ValueError(f"r must be a positive float, got {r}.")

        if x0.shape[0] != self.num_sections or y0.shape[0] != self.num_sections or alpha.shape[0] != self.num_sections:
            raise ValueError(f"x0, y0, and alpha must have length {self.num_sections}, got {x0.shape[0]}, {y0.shape[0]}, and {alpha.shape[0]}.")

        # Convert data to JAX arrays for compatibility with NumPyro
        if section_ids is None:
            section_ids = self.section_ids_obs
        else:
            section_ids = jnp.array(section_ids)
        

        if hole_nos is None:
            hole_nos = self.hole_nos_obs
        else:
            hole_nos = jnp.array(hole_nos)
        
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

        # Stack x and y values to get 2D array of hole positions
        hole_posn_model = jnp.stack([x_model, y_model], axis=1)

        return hole_posn_model

    def likelihood(self, N, r, x0, y0, alpha, sigma, log=False, data = None):
        """
        Computes the likelihood (or log-likelihood) of the observed hole positions given model parameters.

        This function evaluates how well the observed hole positions match the expected positions 
        under a Gaussian noise model. It supports two types of likelihoods:
        
        - **Isotropic Gaussian**: A single standard deviation `sigma` applies to both x and y errors.
        - **Anisotropic Gaussian**: Separate standard deviations `sigma_r` and `sigma_t` for radial 
          and tangential errors.

        The likelihood is calculated as:
        
        - **Isotropic Model**:
        \[
        \log L = -\frac{1}{2} \sum_i \left(\frac{(e_{i,x})^2 + (e_{i,y})^2}{\sigma^2} \right) 
        - n \log(2\pi\sigma)
        \]
        
        - **Anisotropic Model**:
        \[
        \log L = -\frac{1}{2} \sum_i \left(\frac{(e_{i,r})^2}{\sigma_r^2} + \frac{(e_{i,t})^2}{\sigma_t^2} \right) 
        - n \log(2\pi\sigma_r\sigma_t)
        \]

        If a subset of data (`data`) is provided, the gradients are computed over only that subset. 
        This enables efficient **stochastic gradient descent (SGD)** or **mini-batch optimization**.

        **Parameters**
        ----------
        N : float
            The total number of holes in the original (pre-fragmented) calendar ring.
        r : float
            The estimated radius of the ring.
        x0 : jnp.ndarray
            Array of x-offsets for each section.
        y0 : jnp.ndarray
            Array of y-offsets for each section.
        alpha : jnp.ndarray
            Array of angular offsets for each section.
        sigma : float or jnp.ndarray
            - If **isotropic**, a single float `sigma` is used for both x and y errors.
            - If **anisotropic**, a JAX array `[sigma_r, sigma_t]` is used for radial and tangential errors.
        log : bool, optional
            If `True`, returns the **log-likelihood**.
            If `False`, returns the **likelihood**. Default is `False`.
        data : pd.DataFrame, optional
            A subset of the dataset to compute likelihood on. If `None`, the full dataset is used.
            This enables **stochastic optimization** by computing likelihood using **mini-batches**.

        **Returns**
        -------
        jnp.ndarray
            - If `log=True`, returns the **sum of log-likelihood values**.
            - If `log=False`, returns the **exponentiated likelihood**.
        """

        # This implements a similiar analysis to `hole_positions` 
        # Goes on to evaluate discrepancy between model and observed data and calculate log likelihood


        # Compute the expected hole positions based on the model parameters
        # It does no not use the `hole_positions` function but instead calculates the modelled hole positions directly 
        # Stops type checking inputs from slowing preformance when many calls are made

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
            
            log_likelihoods_constant =  -log_likelihoods_variable.size * jnp.log(2 * jnp.pi * sigma_val)

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
            log_likelihoods_variable = (-0.5 * (error_r**2 / sigma[0]**2 + error_t**2 / sigma[1]**2))

            # Compute constant term once for efficiency - no. holes * log(const)
            log_likelihoods_constant = -log_likelihoods_variable.size * jnp.log(2 * jnp.pi * sigma[0] * sigma[1])
        
        # If Log = True - Return log-likelihood 
        if log:
            return jnp.sum(log_likelihoods_variable) + log_likelihoods_constant
        
        # If Log = False - Return likelihood
        else: 
            return jnp.exp(jnp.sum(log_likelihoods_variable) + log_likelihoods_constant)
        
        
    def grad_likelihood(self, N, r, x0, y0, alpha, sigma, log=True, data = None):
        """
        Computes the gradients of the likelihood or log-likelihood with respect to the model parameters.

        This function uses JAX automatic differentiation (`jax.grad()`) to compute:
        - The gradients of the log-likelihood \( \log L(\mathcal{D} | \theta) \)
        - The gradients of the likelihood \( L(\mathcal{D} | \theta) \)
        
        If a subset of data (`data`) is provided, the gradients are computed over only that subset. 
        This enables efficient **stochastic gradient descent (SGD)** or **mini-batch optimization**.

        **Parameters**
        ----------
        N : float
            The total number of holes in the original (pre-fragmented) calendar ring.
        r : float
            The estimated radius of the ring.
        x0 : jnp.ndarray
            Array of x-offsets for each section.
        y0 : jnp.ndarray
            Array of y-offsets for each section.
        alpha : jnp.ndarray
            Array of angular offsets for each section.
        sigma : float or tuple of floats
            - If **isotropic**, a single float `sigma` is used for both x and y errors.
            - If **anisotropic**, a tuple `(sigma_r, sigma_t)` is used for radial and tangential errors.
        log : bool, optional
            If `True`, computes gradients of the **log-likelihood**.
            If `False`, computes gradients of the **likelihood**. Default is `True`.
        data : pd.DataFrame, optional
            A subset of the dataset to compute gradients on. If `None`, the full dataset is used.
            This enables **stochastic optimization** by computing gradients using **mini-batches**.

        **Returns**
        -------
        dict
            A dictionary containing gradients of the log-likelihood or likelihood 
            with respect to each parameter:  
            - `"N"` : Gradient w.r.t. total number of holes.  
            - `"r"` : Gradient w.r.t. radius of the ring.  
            - `"x0"` : Gradients w.r.t. x-offsets (shape: `num_sections`).  
            - `"y0"` : Gradients w.r.t. y-offsets (shape: `num_sections`).  
            - `"alpha"` : Gradients w.r.t. angular offsets (shape: `num_sections`).  
            - `"sigma"` : Gradient w.r.t. sigma (scalar for isotropic, array for anisotropic).
        """

        # Compute gradients using JAX's automatic differentiation tool
        loss_grad_fn = jax.grad(self.likelihood, argnums=(0, 1, 2, 3, 4, 5))
        gradients = loss_grad_fn(N, r, x0, y0, alpha, sigma, log, data)

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
    
    def analytic_grad_loglikelihood(self, N, r, x0, y0, alpha, sigma, data = None):
        """
        Computes the analytical gradients of the log-likelihood function.

        This function calculates the partial derivatives of the log-likelihood function
        with respect to each model parameter using the chain rule. It supports both isotropic 
        and anisotropic error models. 

        If a subset of data (`data`) is provided, the gradients are computed over only that subset. 
        This enables efficient **stochastic gradient descent (SGD)** or **mini-batch optimization**.

        **Log-likelihood expressions:**
        - *Isotropic model* (single sigma for all errors):
          \[
          \log L = -\frac{1}{2\sigma^2} \sum_{i=1}^{n} \left( e_{x,i}^2 + e_{y,i}^2 \right) - n \log(2\pi \sigma)
          \]
        - *Anisotropic model* (independent radial and tangential sigma):
          \[
          \log L = -\frac{1}{2} \sum_{i=1}^{n} \left( \frac{e_{r,i}^2}{\sigma_r^2} + \frac{e_{t,i}^2}{\sigma_t^2} \right) - n \log(2\pi \sigma_r \sigma_t)
          \]

        **Gradient Computation Steps:**
        - The derivatives of the hole positions w.r.t. model parameters are computed using:
          - \(\phi\) (angular position)
          - \(x_{\text{model}}\) and \(y_{\text{model}}\) (modeled positions)
          - \(e_x, e_y\) (errors in Cartesian coordinates)
          - \(e_r, e_t\) (errors in radial and tangential coordinates, for anisotropic case)

        **Parameters**
        ----------
        N : float
            The total number of holes in the original (pre-fragmented) circular ring.
        r : float
            The estimated radius of the ring.
        x0 : jnp.ndarray (shape: `num_sections`)
            x-offsets for each section.
        y0 : jnp.ndarray (shape: `num_sections`)
            y-offsets for each section.
        alpha : jnp.ndarray (shape: `num_sections`)
            Angular offsets for each section.
        sigma : float or jnp.ndarray
            - If **isotropic**, a single float `sigma` is used for both x and y errors.
            - If **anisotropic**, a tuple `(sigma_r, sigma_t)` is used for radial and tangential errors.
        data : pd.DataFrame, optional
            A subset of the dataset to compute gradients on. If `None`, the full dataset is used.
            This is useful for **stochastic gradient descent (SGD)** where mini-batches of data 
            are processed iteratively.

        **Returns**
        -------
        dict
            A dictionary containing gradients of the log-likelihood with respect to:
            - `"N"` : Gradient w.r.t. total number of holes.
            - `"r"` : Gradient w.r.t. radius of the ring.
            - `"x0"` : Gradients w.r.t. x-offsets (shape: `num_sections`).
            - `"y0"` : Gradients w.r.t. y-offsets (shape: `num_sections`).
            - `"alpha"` : Gradients w.r.t. angular offsets (shape: `num_sections`).
            - `"sigma"` : Gradient w.r.t. sigma (scalar for isotropic, array for anisotropic).

        **Notes**
        --------
        - This function **does not use automatic differentiation**; instead, it explicitly derives 
          and applies the chain rule to compute analytical derivatives.
        - If `data` is provided, the gradients are **only computed on that subset**, 
          making the function **compatible with stochastic optimization methods**.
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
            grad_sigma = + jnp.sum((error_x**2 + error_y**2) / sigma_val**3) - len(error_x) / sigma_val

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

            grad_sigma_r = jnp.sum(error_r**2 / sigma_r**3) - len(error_r) / sigma_r # dL/dsigma_r
            grad_sigma_t = jnp.sum(error_t**2 / sigma_t**3)- len(error_t) / sigma_t # dL/dsigma_t
            grad_sigma = jnp.array([grad_sigma_r, grad_sigma_t]) # dL/dsigma

            grad_dict = {
                "N": grad_N,
                "r": grad_r,
                "x0": grad_x0,
                "y0": grad_y0,
                "alpha": grad_alpha,
                "sigma": grad_sigma
            }
            
            return grad_dict

    
    def compare_performance_grad(self, N, r, x0, y0, alpha, sigma, tolerance=1e-3, num_runs=100, subset_size = 40, return_results=False):
        """
        Compares execution time, memory usage, and numerical agreement between:
        - `grad_likelihood` (automatic differentiation via JAX)
        - `analytic_grad_loglikelihood` (manually derived gradients)

        This function now supports **minibatch processing**, where each gradient computation 
        is performed on a randomly sampled subset of the dataset to simulate stochastic 
        gradient descent (SGD). The subset size is controlled by `subset_size`.

        Runs each method `num_runs` times, taking random minibatches of data, 
        to obtain averaged performance metrics for a more robust comparison.

        Parameters
        ----------
        N : float
            Total number of holes in the full calendar ring.
        r : float
            Estimated radius of the ring.
        x0 : jnp.ndarray
            X-offsets for each section.
        y0 : jnp.ndarray
            Y-offsets for each section.
        alpha : jnp.ndarray
            Angular offsets for each section.
        sigma : float or tuple
            - If **isotropic**, a single float `sigma`.
            - If **anisotropic**, a tuple `(sigma_r, sigma_t)`.
        tolerance : float, optional
            The acceptable numerical precision for comparison. Default is `1e-3`.
        num_runs : int, optional
            Number of times to run each method to get averaged metrics. Default is `100`.
        subset_size : int, optional
            Number of data points to randomly sample for each gradient computation, 
            simulating minibatch training. Default is `40`.
        return_results : bool, optional
            Whether to return the performance metrics as a dictionary. Default is `False`.

        Returns
        -------
        dict or None
            If `return_results=True`, returns a dictionary containing:
            - `"Auto-Diff"` : Average execution time and memory usage for JAX automatic differentiation.
            - `"Manual-Diff"` : Average execution time and memory usage for manually derived gradients.
            - `"Agreement"` : Boolean indicating if gradients from both methods agree within tolerance.
            - `"Max Deviation"` : Maximum absolute difference in gradient values between methods.
            - `"Deviations"` : Dictionary of parameters where numerical mismatches were found.
        
            If `return_results=False`, only logs the results and returns `None`.
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

            grad_analytic = self.analytic_grad_loglikelihood(N, r, x0, y0, alpha, sigma, data_subset)

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

            grad_auto = self.grad_likelihood(N, r, x0, y0, alpha, sigma, log=True, data = data_subset)

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

    def sample_from_priors(self, key, num_samples=100):
        """
        Samples multiple parameter sets from the prior distributions using NumPyro.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for JAX sampling.
        num_samples : int, optional
            Number of samples to generate. Default is `100`.

        Returns
        -------
        dict
            Dictionary containing arrays of sampled parameters. Each array has shape `(num_samples, ...)`.
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
    def Max_Likelihood_Est(self, sampling_type, num_samples=1000, num_iterations=500, learning_rate=0.01, batch_size=None, key=None, derivative='analytic', analyse_results=False, plot_history = False):
        """
        Generalized function to compute Maximum Likelihood Estimation (MLE) using different optimization methods.

        Supported methods:
        - Stochastic Gradient Descent (SGD)
        - L-BFGS (Limited-memory BFGS)
        - Newton's Method
        - Simulated Annealing

        Recommended Hyperparameters for Each Method:
        -----------------------------------------------------
        - **SGD (Stochastic Gradient Descent)**
        - `learning_rate`: **0.01 - 0.1** (Recommended: **0.01**)
        - `num_iterations`: **500 - 5000** (Recommended: **1000**)
        - `batch_size`: **20 - 100** (If `None`, full batch is used)
        - `derivative`: **'analytic'** (Recommended) or `'auto'`

        - **L-BFGS (Limited-memory BFGS)**
        - `num_iterations`: **50 - 500** (Recommended: **200**)
        - `batch_size`: **Optional, use if data is large**
        - `derivative`: **'analytic'** (Recommended) or `'auto'`

        - **Newton's Method**
        - `num_iterations`: **10 - 100** (Recommended: **50**)  
        - `learning_rate`: **0.01 - 0.1** (Recommended: **0.01**)  
        - `batch_size`: **Optional, use if data is large**  
        - `derivative`: **'analytic'** (Recommended) or `'auto'`  
        - (Newton's method is **computationally expensive** and may be unstable for large parameter spaces)

        - **Simulated Annealing**
        - `num_iterations`: **100 - 10,000** (Recommended: **5000**)
        - `batch_size`: **Not required** (uses full likelihood evaluation)
        - `temperature`: **Initial `1.0`, cooling rate `0.99`** (default)
        - `perturbation step size`: **0.1** (default, can be tuned)
        - (Simulated Annealing is useful for **non-convex likelihood surfaces**)


        Parameters
        ----------
        sampling_type : str
            Optimization method: 'SGD', 'L-BFGS', 'Newton', or 'Simulated Annealing'.
        num_samples : int, optional
            Number of parameter initializations to optimize. Default is `1000`.
        num_iterations : int, optional
            Number of iterations for the optimization algorithm. Default is `500`.
        learning_rate : float, optional
            Learning rate (used for SGD and Newt’s method). Default is `0.01`.
        batch_size : int, optional
            Size of minibatches for stochastic gradient estimation. Default is `None` (full-batch).
        key : jax.random.PRNGKey, optional
            Random key for reproducibility. Default is a fixed seed.
        derivative : str, optional
            - `'analytic'`: Uses manually computed gradients.
            - `'auto'`: Uses automatic differentiation via `jax.grad()`.
            Default is `'analytic'`.
        analyse_results : bool, optional
            If `True`, computes and visualizes statistics of estimated MLE parameters.

        Returns
        -------
        dict
            Dictionary containing:
            - `"best_params"` : dict, the best parameter set found.
            - `"max_log_likelihood"` : float, the corresponding maximum log-likelihood value.
        """

        # Ensure that the sampling type is valid
        if sampling_type not in ['SGD', 'L-BFGS', 'Newton', 'Simulated Annealing', 'Adam']:
            raise ValueError("sampling_type must be one of ['SGD', 'L-BFGS', 'Newton', 'Simulated Annealing'].")
        
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
                plt.figure(figsize=(8, 6), dpi=300) 
                sns.set_context("talk")  
                sns.set_style("whitegrid") 

                # Plot log-likelihood history
                plt.plot(plot_data, linestyle='-', color='darkblue', linewidth=2, alpha=0.8, label="Negative Log-Likelihood")

                # Labels and formatting
                plt.xlabel("Iterations", fontsize=12, fontweight='bold')
                plt.ylabel("Negative Log-Likelihood", fontsize=12, fontweight='bold')
                plt.title("Log-Likelihood Convergence", fontsize=12, fontweight='bold', pad=15)
                plt.yscale("log")
                plt.grid(which="both", linestyle="--", linewidth=0.7, alpha=0.6)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                plt.legend(fontsize=10)
                plt.show()

        # Define the loss function to be minimized - negative log-likelihood
        def loss_fn(params, data_subset):
            """Negative log-likelihood function that will be minimised."""
            return -self.likelihood(params["N"], params["r"], params["x0"], params["y0"], params["alpha"], params["sigma"], 
                                    log=True, data=data_subset)
        
        # Define the gradient function based on user choice for relevant methods
        if sampling_type in ['SGD', 'L-BFGS', 'Newton', 'Adam']:
            if derivative == 'auto':
                grad_fn = jax.grad(loss_fn)
            else:
                def grad_fn(params, data_subset):
                    """Derivative Negative log-likelihood function that will be minimised."""
                    grads = self.analytic_grad_loglikelihood(params["N"], params["r"], params["x0"], params["y0"], params["alpha"], 
                                                            params["sigma"], data=data_subset)
                    # tree_map negates each element of the dictionary returned by previously defined function
                    return jtu.tree_map(lambda x: -x, grads)
        
        # Define the hessian function using jax automatic differentiation
        if sampling_type == 'Newton': 
            hessian_fn = jax.hessian(loss_fn)

        # Sample from the priors simply as initialisation position for MLE optimisation
        # All initial samples are retrieved in one go
        prior_samples = self.sample_from_priors(key, num_samples=num_samples)

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

        # ------------------ L-BFGS ------------------
        
        if sampling_type == 'L-BFGS':
            for i in tqdm(range(num_samples), desc="Optimizing MLE using L-BFGS:", leave=True):

                # Extract the parameters for the current sample from the dictionary - ensure they are jnp arrays
                params = {k: jnp.array(v[i]) for k, v in prior_samples.items()}

                def objective_fn(p):
                    """Computes the negative log-likelihood over a batch."""
                    data_subset = self.data if batch_size is None else self.data.sample(batch_size)
                    return loss_fn(dict(zip(params.keys(), p)), data_subset)

                def objective_grad_fn(p):
                    """Computes the gradient of the negative log-likelihood over a batch."""
                    data_subset = self.data if batch_size is None else self.data.sample(batch_size)
                    grads = grad_fn(dict(zip(params.keys(), p)), data_subset)
                    return jnp.concatenate([grads[k].flatten() for k in grads.keys()])  # Convert dict to flat array

                # Track training progress
                training_log = []

                def callback(p):
                    """Callback function to track log-likelihood progress during training."""
                    log_likelihood = -objective_fn(p)  # Convert back to log-likelihood
                    training_log.append(log_likelihood)
                    print(f"Iteration {len(training_log)}: Log-Likelihood = {log_likelihood:.4f}")

                # Convert initial params to a flat array
                initial_params = jnp.concatenate([params[k].flatten() for k in params.keys()])

                # Run L-BFGS optimizer from SciPy with callback tracking
                result = minimize(fun=objective_fn, x0=initial_params, jac=objective_grad_fn, 
                                method='L-BFGS-B', options={'maxiter': num_iterations},
                                callback=callback)  # Track optimization progress

                # Convert optimized flat array back to dictionary format
                optimized_params = dict(zip(params.keys(), jnp.split(result.x, [params[k].size for k in params.keys()][:-1])))

                # Ensure that alpha is within [-π, π]
                optimized_params["alpha"] = (optimized_params["alpha"] + jnp.pi) % (2 * jnp.pi) - jnp.pi

                # Convert from minimized negative log-likelihood to log-likelihood (maximized)
                final_log_likelihood = -objective_fn(result.x)

                # Store the final parameter estimates and log-likelihood
                all_results.append({"params": optimized_params, "log_likelihood": final_log_likelihood})

                # Plot training log-likelihood history
                if plot_history:
                    _plot_log_likelihood_history(all_results_log_likelihood)

        # ------------------ Newton's Method ------------------

        if sampling_type == 'Newton':
            for i in tqdm(range(num_samples), desc="Optimizing MLE using Newton's Method:", leave=True):

                # Extract the parameters for the current sample from the dictionary - ensure they are jnp arrays
                params = {k: jnp.array(v[i]) for k, v in prior_samples.items()}

                for _ in range(num_iterations):

                    # Store the log-likelihood history for each iteration
                    if plot_history:
                        all_results_log_likelihood.append(loss_fn(params, self.data))

                    data_subset = self.data if batch_size is None else self.data.sample(batch_size)
                    grads = grad_fn(params, data_subset)
                    hessian = hessian_fn(params, data_subset)
                    # Add Regularisation to make more stable inverse
                    hessian_inv = jnp.linalg.inv(hessian + 1e-6 * jnp.eye(len(params)))
                    params = {k: v - jnp.dot(hessian_inv, grads[k]) for k, v in params.items()}

                # Plot the log-likelihood history using the function defined above
                if plot_history:
                    _plot_log_likelihood_history(all_results_log_likelihood)

                # Ensure that alpha is within [-π, π]
                params["alpha"] = (params["alpha"] + jnp.pi) % (2 * jnp.pi) - jnp.pi

                # Convert from minimised negative log-likelihood to log-likelihood (maximised)
                final_log_likelihood = -loss_fn(params, self.data)

                # Store the final parameter estimates and log-likelihood
                all_results.append({"params": params, "log_likelihood": final_log_likelihood})


        # ------------------ Simulated Anealing ------------------

        if sampling_type == 'Simulated Annealing':
            for i in tqdm(range(num_samples), desc="Optimizing MLE using Simulated Annealing:", leave=True):

                # Extract the parameters for the current sample from the dictionary - ensure they are jnp arrays
                params = {k: jnp.array(v[i]) for k, v in prior_samples.items()}

                # Define the objective function: negative log-likelihood (since we are minimizing)
                def objective_fn(p):
                    data_subset = self.data if batch_size is None else self.data.sample(batch_size)
                    return -self.likelihood(p["N"], p["r"], p["x0"], p["y0"], p["alpha"], p["sigma"], log=True, data=data_subset)

                # Simulated Annealing hyperparameters
                temperature = 1.0  
                cooling_rate = 0.99  
                best_params = params
                best_score = objective_fn(params)
                accepted_moves = 0  # Track how many worse solutions are accepted

                for iteration in range(num_iterations):
                    # Generate a new candidate by applying small random noise
                    new_params = {k: v + 0.1 * jax.random.normal(key, v.shape) for k, v in params.items()}
                    new_score = objective_fn(new_params)

                    # Check if the new score is valid (avoid NaN or Inf)
                    if jnp.isnan(new_score) or jnp.isinf(new_score):
                        continue  # Skip this iteration and generate a new candidate

                    # Accept new parameters if the score is better or with probability exp(-ΔE / T)
                    delta = new_score - best_score
                    if delta < 0 or jax.random.uniform(key) < jnp.exp(-delta / temperature):
                        params = new_params  # Accept new parameters
                        best_score = new_score
                        accepted_moves += 1

                    # Gradual cooling schedule
                    temperature = 1.0 / (1 + iteration)  

                # Ensure that alpha is within [-π, π]
                params["alpha"] = (params["alpha"] + jnp.pi) % (2 * jnp.pi) - jnp.pi

                # Convert from minimized negative log-likelihood to log-likelihood (maximized)
                final_log_likelihood = -best_score

                # Store the final parameter estimates and log-likelihood
                all_results.append({"params": params, "log_likelihood": final_log_likelihood})

            # Log acceptance rate
            logging.info(f"Simulated Annealing Acceptance Rate: {accepted_moves / (num_samples * num_iterations):.2f}")

        

        # Apply a filter on all results that removed any unphysical values - ie N < 0 or r < 0
        filtered_results = [entry for entry in all_results if entry["params"]["N"] > 0 and entry["params"]["r"] > 0]
        num_removed = num_samples - len(filtered_results)
        logging.info(f"Removed {num_removed}/{num_samples} MLE estimates due to unphysical values (N or r < 0).")
        if not filtered_results:
            raise RuntimeError("All estimated parameters were invalid. Consider adjusting priors.")
        
        # Find the best result based on maximum log-likelihood
        best_result = max(filtered_results, key=lambda x: x["log_likelihood"])

        if analyse_results:
            logging.info("Running Analysis on MLE Results...")
            self._analyse_mle_results(filtered_results)

        return filtered_results

        # return {"best_params": best_result["params"], "max_log_likelihood": best_result["log_likelihood"]}
    

    # ------------------ Analyse MLE Results ------------------

    def _analyse_mle_results(self, mle_results):
        """
        Analyses Maximum Likelihood Estimation (MLE) results by:
        - Identifying the best log-likelihood estimate
        - Computing gradients at the best estimates (top 20%)
        - Finding the top 20% of log-likelihoods
        - Plotting histograms of log-likelihoods, gradients, and parameters.

        Parameters
        ----------
        mle_results : list of dict
            Each dictionary contains:
            - `"params"`: A dictionary of parameter estimates.
            - `"log_likelihood"`: The log-likelihood value.
        """

        if not mle_results:
            raise ValueError("No MLE results provided for analysis.")

        # Convert results into structured format
        log_likelihoods = np.array([entry["log_likelihood"] for entry in mle_results])

        # Separate scalar parameters from vector parameters
        param_dict = {}
        for key in mle_results[0]["params"]:
            values = np.array([entry["params"][key] for entry in mle_results])

            # If the parameter is a vector (e.g., x0, y0, alpha), split into components
            if values.ndim > 1:
                for i in range(values.shape[1]):
                    param_dict[f"{key}_{i+1}"] = values[:, i]
            else:
                param_dict[key] = values

        # ------------------ Best Log-Likelihood and Gradients ------------------

        # Find the best MLE estimate
        best_idx = np.argmax(log_likelihoods)
        best_params = mle_results[best_idx]["params"]
        best_log_likelihood = log_likelihoods[best_idx]

        logging.info(f"Best MLE estimate found at index {best_idx} with log-likelihood = {best_log_likelihood:.4f}")

        # Compute gradient at best estimate (if gradient function is available)
        try:
            best_gradient = self.analytic_grad_loglikelihood(**best_params, data=self.data)
            gradient_values = np.concatenate([v.flatten() for v in best_gradient.values()])
            gradient_values_mag = np.linalg.norm(gradient_values)
            logging.info(f"Gradient at best MLE estimate found to be magnitude: {gradient_values_mag:.4f}")

        except Exception as e:
            best_gradient = None
            logging.info("Gradient computation failed:", e)


        # ------------------ Top 20% of Log-Likelihoods ------------------
        cutoff = np.percentile(log_likelihoods, 80)  # Find 80th percentile threshold
        top_20_mask = log_likelihoods >= cutoff  # Selects the least negative log-likelihoods (best fits)
        
        # Store the top 20% of parameters
        top_20_params = {key: values[top_20_mask] for key, values in param_dict.items()}
        top_20_log_likelihoods = log_likelihoods[top_20_mask]

        logging.info(f"Top 20% of best log-likelihoods have values above {cutoff:.4f}")

        # ------------------ Compute Gradient Magnitudes for Top 20% ------------------

        gradient_magnitudes = []
        for i, entry in enumerate(mle_results):
            if top_20_mask[i]:  # Only compute for top 20% subset
                try:
                    gradient = self.analytic_grad_loglikelihood(**entry["params"], data=self.data)
                    grad_vector = np.concatenate([v.flatten() for v in gradient.values()])  # Flatten to vector
                    grad_magnitude = np.linalg.norm(grad_vector)  # Compute L2 norm
                    gradient_magnitudes.append(grad_magnitude)
                except Exception as e:
                    logging.warning(f"Skipping gradient for sample {i}: {e}")

        gradient_magnitudes = np.array(gradient_magnitudes)  # Convert to NumPy array for plotting

        # ------------------ Plots: Filtered Log-Likelihood Distribution ------------------

        plt.figure(figsize=(8, 6), dpi=300)
        sns.histplot(top_20_log_likelihoods, bins=15, kde=True, color="darkblue")
        plt.xlabel("Log-Likelihood", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Distribution of Log-Likelihoods (Top 20%)", fontsize=16, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

        # ------------------ Plots: Gradient Magnitude Distribution ------------------

        if len(gradient_magnitudes) > 0:
            plt.figure(figsize=(8, 6), dpi=300)
            sns.histplot(gradient_magnitudes, bins=20, kde=True, color="purple")
            plt.xlabel("Gradient Magnitude", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.title("Gradient Magnitude Distribution (Top 20%)", fontsize=16, fontweight="bold")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()
        else:
            logging.info("Skipping gradient magnitude plot: No valid gradients found.")

        # ------------------ Plots: Parameter Distributions ------------------

        num_params = len(top_20_params)
        fig, axes = plt.subplots(nrows=num_params, figsize=(8, 3 * num_params), dpi=300)

        if num_params == 1:
            axes = [axes]  # Ensure it's iterable

        for ax, (param, values) in zip(axes, top_20_params.items()):
            sns.histplot(values.flatten(), ax=ax, kde=True, bins=30, color="green")
            ax.set_title(f"Distribution of {param} (Top 20%)", fontsize=14, fontweight="bold")
            ax.set_xlabel(param, fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

        # ------------------ Display Summary Table ------------------

        return None
















            # # Convert trajectory to structured format for analysis
            # filtered_resultls_reord = {key: np.array([entry["params"][key] for entry in filtered_results])
            #     for key in filtered_results[0]["params"]}
            
            # logging.info("Filtering outliers before analysis...")
            
            # # Apply outlier detection only for analysis (not for finding best params)
            # filter_mask = self._detect_outliers_mle(filtered_resultls_reord)
            # param_trajectory_filtered = {key: values[filter_mask] for key, values in filtered_resultls_reord.items()}
            
            # self._analyse_mle_results(param_trajectory_filtered)
            # self._plot_mle_distributions(param_trajectory_filtered)


    

    # def _analyse_mle_results(self, mle_results_filtered):
    #     """Computes and displays mean and variance of parameters across valid MLE runs, removing entire runs if any parameter is extreme."""
    #     # Initialize a dictionary to store statistics
    #     stats_dict = {}

    #     for param, values in mle_results_filtered.items():
    #         if values.ndim == 1:
    #             # Scalar parameters (N, r, sigma)
    #             stats_dict[param] = {
    #                 "Mean": np.mean(values),
    #                 "Variance": np.var(values)
    #             }
    #         else:
    #             # Vector parameters (x0, y0, alpha) – process each component separately
    #             for i in range(values.shape[1]):
    #                 stats_dict[f"{param}_{i+1}"] = {
    #                     "Mean": np.mean(values[:, i]),
    #                     "Variance": np.var(values[:, i])
    #                 }

    #     # Convert to a DataFrame for a cleaner display
    #     stats_df = pd.DataFrame(stats_dict).T
    #     display(stats_df)
        
    #     return None


    # def _plot_mle_distributions(self, mle_results_filtered):
    #     """
    #     Plots histograms of MLE-estimated parameters, removing entire MLE estimates if any parameter is an outlier.
    #     """

    #     # Flatten vector parameters (x0, y0, alpha)
    #     flattened_params = {}

    #     for param, values in mle_results_filtered.items():
    #         if values.ndim == 1:
    #             # Scalar parameters (N, r, sigma)
    #             flattened_params[param] = values
    #         else:
    #             # Vector parameters (x0, y0, alpha) – store each component separately
    #             for i in range(values.shape[1]):
    #                 flattened_params[f"{param}_{i+1}"] = values[:, i]

    #     # Plot each parameter's histogram
    #     num_params = len(flattened_params)
    #     fig, axes = plt.subplots(nrows=num_params, figsize=(8, 3 * num_params))

    #     if num_params == 1:
    #         axes = [axes]

    #     for ax, (param, values) in zip(axes, flattened_params.items()):
    #         sns.histplot(values, ax=ax, kde=True, bins=30)
    #         ax.set_title(f"Distribution of {param} (Outliers Removed)")
    #         ax.set_xlabel(param)
    #         ax.set_ylabel("Frequency")

    #     plt.tight_layout()
    #     plt.show()
        
    #     return None


        # def _detect_outliers_mle(self, mle_results, lower_percentile=0.1, upper_percentile=99.9):
    #     """
    #     Identifies MLE runs containing at least one extreme parameter based on percentile-based filtering.

    #     Parameters
    #     ----------
    #     mle_results : dict of np.ndarrays
    #         Dictionary where each key corresponds to a parameter and each value is an array of shape `(num_samples,)` or `(num_samples, dim)`.
    #     lower_percentile : float, optional
    #         The lower percentile threshold (default is 10th percentile).
    #     upper_percentile : float, optional
    #         The upper percentile threshold (default is 90th percentile).

    #     Returns
    #     -------
    #     valid_mask : np.ndarray
    #         Boolean mask where `True` indicates a valid MLE estimate and `False` indicates an outlier.
    #     """
    #     num_samples = len(next(iter(mle_results.values())))
    #     valid_mask = np.ones(num_samples, dtype=bool)

    #     for param, values in mle_results.items():
    #         lower_bound = np.percentile(values, lower_percentile)
    #         upper_bound = np.percentile(values, upper_percentile)

    #         if values.ndim == 1:
    #             valid_mask &= (values >= lower_bound) & (values <= upper_bound)
    #         else:
    #             for i in range(values.shape[1]):
    #                 valid_mask &= (values[:, i] >= lower_bound) & (values[:, i] <= upper_bound)
        
    #     logging.info(f"Removed {num_samples - sum(valid_mask)} extreme MLE estimates out of {num_samples}.")
    #     return valid_mask
            



    def NumPryo_model(self):
        """
        Defines the model and the likelihood function for Bayesian inference using NumPyro.

        This function models the likelihood of the hole positions assuming either:
        - **Isotropic Gaussian errors**: A single standard deviation `sigma` applies to both x and y errors.
        - **Anisotropic Gaussian errors** (default): Separate standard deviations `sigma_r` and `sigma_t` describe 
        independent radial and tangential accuracies of hole placement.

        The function infers the following parameters:
        - `N` : The total number of holes in the full calendar ring.
        - `r` : The estimated radius of the ring before fragmentation.
        - `x0_j, y0_j` : The center offsets for each section.
        - `alpha_j` : The rotation angles for each section.
        - `sigma` (for isotropic model) : Common standard deviation for errors.
        - `sigma_r, sigma_t` (for anisotropic model) : Standard deviations for radial and tangential errors.

        The likelihood follows a Gaussian model:

        - **Isotropic Gaussian:**
        \[
        p(\{d_i\} | \mathbf{a}) = (2\pi\sigma^2)^{-n} \prod_{i=1}^{n} \exp \left[ -\frac{(e_{i,x})^2 + (e_{i,y})^2}{2\sigma^2} \right]
        \]
        - **Anisotropic Gaussian:**
        \[
        p(\{d_i\} | \mathbf{a}) = (2\pi\sigma_r\sigma_t)^{-n} \prod_{j=0}^{s-1} \prod_{i \in j} 
        \exp \left[ -\frac{(e_{ij} \cdot \hat{r}_{ij})^2}{2\sigma_r^2} - \frac{(e_{ij} \cdot \hat{t}_{ij})^2}{2\sigma_t^2} \right]
        \]

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
            sigma_r = numpyro.sample("sigma_r", dist.Exponential(1.0))
            sigma_t = numpyro.sample("sigma_t", dist.Exponential(1.0))
            sigma = jnp.array([sigma_r, sigma_t])

        # Store deterministic model predictions
        numpyro.deterministic("hole_positions",  self.hole_positions(N, r, x0, y0, alpha, section_ids = None , hole_nos = None))

        # -------------------- Define likelihood --------------------
        numpyro.factor("likelihood", self.likelihood(N, r, x0, y0, alpha, sigma, log=True))

    
            