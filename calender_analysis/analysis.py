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
            "r": dist.Uniform(100, 200),
            "x0": dist.Normal(0, 10),
            "y0": dist.Normal(0, 10),
            "alpha": dist.Normal(0, 0.1),
            "sigma": dist.Exponential(1.0),  # For isotropic
            "sigma_r": dist.Exponential(1.0),  # For anisotropic
            "sigma_t": dist.Exponential(1.0),  # For anisotropic
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
    
    # CONSIDER MAKING THIS BE ABLE TO ACCEPT MANY DIFFERENT MODEL PARAMATERS AT ONCE
    def likelihood(self, N, r, x0, y0, alpha, sigma, log=False):
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

        **Automatic Differentiation Support**:
        - This function is compatible with JAX's `jax.grad()` for computing derivatives with respect to model parameters.
        - Gradients can be used for parameter optimization or inference methods like Hamiltonian Monte Carlo (HMC).
        
        Parameters
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
            - If **anisotropic**, a jax array `[sigma_r, sigma_t]` is used for radial and tangential errors.
        log : bool, optional
            If `True`, returns the log-likelihood. If `False`, returns the likelihood. Default is `False`.

        Returns
        -------
        jnp.ndarray
            - If `log=True`, returns the sum of log-likelihood values.
            - If `log=False`, returns the exponentiated likelihood.
            """

        # This implements a similiar analysis to `hole_positions` 
        # Goes on to evaluate discrepancy between model and observed data and calculate log likelihood


        # Compute the expected hole positions based on the model parameters
        # It does no not use the `hole_positions` function but instead calculates the modelled hole positions directly 
        # Stops type checking inputs from slowing preformance when many calls are made

        # ------------------------- Repeat of code from `hole_positions` -------------------------
        
        # Compute expected positions of model from parameters for each hole/ data point
        # The phi value of each hole is calculated based on the hole number anf the section it belongs to
        # Hole value - anguluar position in the ring - from fraction of 2 pi based on hole number/ N 
        # Section value - using alpha gives relative rotation of each section/ anguluar offset
        # List of phi values for each hole
        phi = (2 * jnp.pi * (self.hole_nos_obs - 1) / N) + alpha[self.section_ids_obs - 1]
        # Expected x and y positions of the hole based on the model - using r, phi, x0, y0
        # List of modelled x and y values of for each hole
        x_model = r * jnp.cos(phi) + x0[self.section_ids_obs - 1]
        y_model = r * jnp.sin(phi) + y0[self.section_ids_obs - 1]

        # -----------------------------------------------------------------------------------------

        # Compute error vectors in cartesian coordinates
        # List of error vectors for each hole - difference between observed and modelled x and y values
        error_x = self.x_obs - x_model
        error_y = self.y_obs - y_model

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
        
    def likelihood_grad(self, N, r, x0, y0, alpha, sigma, log=True):
        """
        Computes the gradients of the likelihood or log-likelihood with respect to the model parameters.

        This function uses JAX automatic differentiation to compute:
        - The gradients of the log-likelihood \( \log L(\mathcal{D} | \theta) \)
        - The gradients of the likelihood \( L(\mathcal{D} | \theta) \)

        Parameters
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

        Returns
        -------
        dict
            A dictionary containing gradients of the log-likelihood or likelihood 
            with respect to each parameter: `{"N": dL/dN, "r": dL/dr, ...}`
        """

        # Compute gradients using JAX's automatic differentiation tool
        loss_grad_fn = jax.grad(self.likelihood, argnums=(0, 1, 2, 3, 4, 5))
        gradients = loss_grad_fn(N, r, x0, y0, alpha, sigma, log)

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
    
    def Log_likelihood_grad_analytic(self, N, r, x0, y0, alpha, sigma):
        """
        Computes the analytical gradients of the log-likelihood function.

        This function calculates the partial derivatives of the log-likelihood function
        with respect to each model parameter using the chain rule. It supports both isotropic 
        and anisotropic error models.

        **Log-likelihood expressions:**
        - *Isotropic model* (single σ for all errors):
          \[
          \log L = -\frac{1}{2\sigma^2} \sum_{i=1}^{n} \left( e_{x,i}^2 + e_{y,i}^2 \right) - n \log(2\pi \sigma)
          \]
        - *Anisotropic model* (independent radial and tangential σ):
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
        """
        # Uses the chain rule to determine the analytical gradients of the log likelihood
        # For each item of the graphical model we word out gradients of next dependant variable
        # Total derivatives will be product of all partial derivatives

        ## ------------------ Compute intermediate values ------------------

        # phi: functions on N, alpha
        phi = (2 * jnp.pi * (self.hole_nos_obs - 1) / N) + alpha[self.section_ids_obs - 1] # - (n_holes)

        # x_model: function of r, phi, x0
        x_model = r * jnp.cos(phi) + x0[self.section_ids_obs - 1] # - (n_holes)

        # y_model: function of r, phi, y0
        y_model = r * jnp.sin(phi) + y0[self.section_ids_obs - 1] # - (n_holes)

        # Error terms: function of modeled positions (x_model, y_model)
        error_x = self.x_obs - x_model  # - (n_holes)
        error_y = self.y_obs - y_model  # - (n_holes)

        ## ------------------ Compute derivatives using chain rule ------------------

        # Partial derivatives of phi
        grad_phi_N = -2 * jnp.pi * (self.hole_nos_obs - 1) / N**2  # dphi/dN - (n_holes)
        grad_phi_alpha = jnp.eye(self.num_sections)[self.section_ids_obs - 1].T # dphi/dalpha - one-hot encoded matrix of shape (num_sections, n_holes) -- identity vector where 1 at section index, 0 elsewhere

        # Partial derivatives of x_model
        grad_x_model_r = jnp.cos(phi)  # dx_model/dr - (n_holes)
        grad_x_model_x0 = jnp.eye(self.num_sections)[self.section_ids_obs - 1].T  # dx_model/dx0 - (num_sections, n_holes) 
        # grad_x_model_y0 = 0 
        grad_x_model_phi = -r * jnp.sin(phi)  # dx_model/dphi - (n_holes)
        grad_x_model_N = grad_x_model_phi * grad_phi_N  # dx_model/dN - (n_holes)
        grad_x_model_alpha = grad_phi_alpha * grad_x_model_phi  # dx_model/dalpha - (num_sections, n_holes) 

        # Partial derivatives of y_model
        grad_y_model_r = jnp.sin(phi)  # dy_model/dr
        # grad_y_model_x0 = 0
        grad_y_model_y0 = jnp.eye(self.num_sections)[self.section_ids_obs - 1].T  # dy_model/dy0 - (num_sections, n_holes) 
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

            # Believe this part is wrong - need to check! FLAG HERE
            grad_sigma = - jnp.sum((error_x**2 + error_y**2) / sigma_val**3 - len(error_x) / sigma_val)


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
            grad_N = - jnp.sum((error_r * grad_error_r_N) / sigma_r**2 + (error_t * grad_error_t_N) / sigma_t**2)
            grad_r = - jnp.sum((error_r * grad_error_r_r) / sigma_r**2 + (error_t * grad_error_t_r) / sigma_t**2)
            grad_x0 = - jnp.sum((error_r * grad_error_r_x0) / sigma_r**2 + (error_t * grad_error_t_x0) / sigma_t**2, axis=1)
            grad_y0 = - jnp.sum((error_r * grad_error_r_y0) / sigma_r**2 + (error_t * grad_error_t_y0) / sigma_t**2, axis=1)

            # Believe this part is wrong - need to check FLAG HERE
            grad_alpha = - jnp.sum((error_r * grad_error_r_alpha + error_t * grad_error_t_alpha) / sigma_r**2 + sigma_t**2, axis=1)
            grad_sigma_r = - jnp.sum(error_r**2 / sigma_r**3 - len(error_r) / sigma_r)
            grad_sigma_t = - jnp.sum(error_t**2 / sigma_t**3 - len(error_t) / sigma_t)
            grad_sigma = - jnp.array([grad_sigma_r, grad_sigma_t])

            grad_dict = {
                "N": grad_N,
                "r": grad_r,
                "x0": grad_x0,
                "y0": grad_y0,
                "alpha": grad_alpha,
                "sigma": grad_sigma
            }
            
            return grad_dict


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
            