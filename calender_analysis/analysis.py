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

class CalendarAnalysis:
    def __init__(self, data, model_type="anisotropic"):
        """
        Initialize the Bayesian analysis for the Antikythera mechanism.

        Parameters
        ----------
        data : str, pd.DataFrame, or np.ndarray
            The input dataset.
        model_type : str, optional
            Specifies the error model:
            - "anisotropic" (default) → Uses separate σ_r and σ_t.
            - "isotropic" → Uses a single σ for both directions.
        """
        self.data = self._load_data(data)
        self.n_holes = len(self.data)
        self.n_sections = len(self.data["Section ID"].unique())
        
        # Validate model type
        if model_type not in ["anisotropic", "isotropic"]:
            raise ValueError("Invalid model_type. Choose either 'anisotropic' or 'isotropic'.")
        self.model_type = model_type


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

    def bayesian_model(self):
        """
        Defines the model and the likelihood function for Bayesian inference using NumPyro.

        This function models the likelihood of the hole positions assuming either:
        - **Isotropic Gaussian errors**: A single standard deviation `sigma` applies to both x and y errors.
        - **Anisotropic Gaussian errors** (default): Separate standard deviations `sigma_r` and `sigma_t` describe independent 
          radial and tangential accuracies of hole placement.

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
          p(\{d_i\} | \mathbf{a}) = (2\pi\sigma_r\sigma_t)^{-n} \prod_{j=0}^{s-1} \prod_{i \in j} \exp \left[ -\frac{(e_{ij} \cdot \hat{r}_{ij})^2}{2\sigma_r^2} - \frac{(e_{ij} \cdot \hat{t}_{ij})^2}{2\sigma_t^2} \right]
          \]

        The choice between isotropic and anisotropic models is determined by `self.model_type`.

        Parameters
        ----------
        None (Uses class attributes)

        Returns
        -------
        None
            This function does not return values directly; instead, it defines the probabilistic model within NumPyro.

        Notes
        -----
        - The function stores **deterministic intermediate values** (`hole_positions`, and `error_vectors`)
          to be accessible during posterior analysis.
        """

        # Convert data to JAX arrays for compatibility with NumPyro
        section_ids = jnp.array(self.data["Section ID"].values)
        hole_no = jnp.array(self.data["Hole"].values)
        x_obs = jnp.array(self.data["Mean(X)"].values)
        y_obs = jnp.array(self.data["Mean(Y)"].values)
        
        # Define model parameters - with their priors by default
        # Define priors at the same time as Probabilistic Programming Language

        # Parameters: N, r, sigma_r, sigma_t
        # These are constant for all sections and holes as they are prior to the breaking of the ring

        # Model parameters:
        # Total number of holes in the calendar ring
        N = numpyro.sample("N", dist.Uniform(300, 400))
        # Ring radius
        r = numpyro.sample("r", dist.Uniform(100, 200))
        # Radial and tangential errors (standard deviations)
        # These are defined later based on the model type
        
        # Parameters: x0, y0, alpha
        # These are section-specific parameters - due to relative rotation and translation of each sections
        # These are intialised as vectors of length = number of sections, each element representing a section
        # The ith element of each vector represents the ith section labelled i+1
        x0 = numpyro.sample("x0", dist.Normal(0, 10).expand([self.num_sections]))
        y0 = numpyro.sample("y0", dist.Normal(0, 10).expand([self.num_sections]))
        alpha = numpyro.sample("alpha", dist.Normal(0, 0.1).expand([self.num_sections]))

        
        # Compute expected positions of model from parameters for each hole/ data point
        # The phi value of each hole is calculated based on the hole number anf the section it belongs to
        # Hole value - anguluar position in the ring - from fraction of 2 pi based on hole number/ N 
        # Section value - using alpha gives relative rotation of each section/ anguluar offset
        # List of phi values for each hole
        phi = (2 * jnp.pi * (hole_no - 1) / N) + alpha[section_ids - 1]
        # Expected x and y positions of the hole based on the model - using r, phi, x0, y0
        # List of modelled x and y values of for each hole
        x_model = r * jnp.cos(phi) + x0[section_ids - 1]
        y_model = r * jnp.sin(phi) + y0[section_ids - 1]
        hole_posn_model = jnp.stack([x_model, y_model], axis=1)


        # Store deterministic model predictions
        numpyro.deterministic("hole_positions", hole_posn_model)
        
        # Compute error vectors in cartesian coordinates
        # List of error vectors for each hole - difference between observed and modelled x and y values
        error_x = x_obs - x_model
        error_y = y_obs - y_model
        
        # Translate error vectors to radial and tangential coordinates
        # For each hole, determine unit radial and tangential vectors - ie transform error vectors x and y to radial and tangential components
        unit_r = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=1)
        unit_t = jnp.stack([jnp.sin(phi), -jnp.cos(phi)], axis=1)

        # List of radial and tangential error values for each hole
        error_r = error_x * unit_r[:, 0] + error_y * unit_r[:, 1]
        error_t  = error_x * unit_t[:, 0] + error_y * unit_t[:, 1]

        # Store deterministic transformed errors (optional)
        error_vectors = jnp.stack([error_r, error_t], axis=1)
        numpyro.deterministic("error_vectors", error_vectors)


        # Define likelihood based on error model
        # Plate can be used as independent random variables
        # Allows the likelihood to be calculated in parallel for each hole as they are independent
        # Each normal likelihood is conditioned on the error_r and error_t values and total likelihood is product of all likelihoods

        # Isotropic Gaussian model - single sigma for both radial and tangential errors
        if self.model_type == "isotropic":
            sigma = numpyro.sample("sigma", dist.Exponential(1.0))
            
            with numpyro.plate("holes", self.n_holes):
                numpyro.sample("obs_x", dist.Normal(0, sigma), obs=error_x)
                numpyro.sample("obs_y", dist.Normal(0, sigma), obs=error_y)

        # Anisotropic Gaussian model - separate sigmas for radial and tangential errors
        elif self.model_type == "anisotropic":
            sigma_r = numpyro.sample("sigma_r", dist.Exponential(1.0))
            sigma_t = numpyro.sample("sigma_t", dist.Exponential(1.0))

            with numpyro.plate("holes", self.n_holes):
                numpyro.sample("obs_r", dist.Normal(0, sigma_r), obs=error_r)
                numpyro.sample("obs_t", dist.Normal(0, sigma_t), obs=error_t)
            


## Have not rigurously checked priors
## It is asking about log likelihood? 
## It is asking about deritives
        