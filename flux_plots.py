import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, IntSlider
import cmcrameri.cm as cm
from skimage import measure

def plot_concentration_contours_yz_plane_interactive(X, Y, Z, concentration):
    """
    Creates an interactive plot of concentration contours in the z-y plane at a specified x index.
    The color bar is fixed to span the entire range of concentration values across all x indices.
    The color bar tick labels are made more uniform and consistent.
    
    Parameters:
    - X, Y, Z: np.ndarray, coordinate grids
    - concentration: np.ndarray, concentration values at each grid point
    
    Returns:
    - fig, ax: Matplotlib figure and axes objects
    """
    # Compute global minimum and maximum concentration values across all x indices
    concentration_global = np.maximum(concentration, 1e-8)  # Handle zero or negative values
    global_min_conc = concentration_global.min()
    global_max_conc = concentration_global.max()
    
    # Create the figure and axes once
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initial index
    x_index = 0  # You can set this to any valid index
    concentration_slice = concentration_global[x_index, :, :]
    
    # Generate levels and grids
    levels = np.logspace(np.floor(np.log10(global_min_conc)), np.ceil(np.log10(global_max_conc)), num=50)
    y_coords = Y[x_index, :, 0]
    z_coords = Z[x_index, 0, :]
    Y_grid, Z_grid = np.meshgrid(y_coords, z_coords, indexing='ij')
    
    # Create a normalization object with fixed limits
    norm = mcolors.LogNorm(vmin=10**np.floor(np.log10(global_min_conc)), vmax=10**np.ceil(np.log10(global_max_conc)))
    
    # Initial contour plot
    contour = ax.contourf(
        Y_grid, Z_grid, concentration_slice,
        levels=levels,
        norm=norm,
        cmap=cm.batlow
    )
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Concentration (g/m³)')
    
    # Set custom ticks for the colorbar
    # Generate logarithmically spaced ticks
    tick_min = np.floor(np.log10(global_min_conc))
    tick_max = np.ceil(np.log10(global_max_conc))
    tick_values = np.logspace(tick_min, tick_max, num=int(tick_max - tick_min + 1))
    cbar.set_ticks(tick_values)
    
    # Format the tick labels to be more readable
    cbar.set_ticklabels([f'$10^{{{int(tick)}}}$' for tick in range(int(tick_min), int(tick_max) + 1)])
    
    # Set labels and initial title
    ax.set_xlabel('Crosswind Distance y (m)')
    ax.set_ylabel('Height z (m)')
    x_value = X[x_index, 0, 0]
    title_text = ax.set_title(f'Concentration at x = {x_value:.2f} m')
    
    # Define the update function
    def update_plot(x_index):
        nonlocal contour  # Access variables from the outer scope

        # Remove the previous contour plot
        if hasattr(contour, 'remove'):
            # For Matplotlib 3.8 and newer
            contour.remove()
        else:
            # For older versions
            for coll in contour.collections:
                coll.remove()
            contour.collections.clear()
        
        # Update the concentration slice
        concentration_slice = concentration_global[x_index, :, :]
        
        # Update contour plot with fixed levels and norm
        contour = ax.contourf(
            Y_grid, Z_grid, concentration_slice,
            levels=levels,
            norm=norm,
            cmap=cm.batlow
        )
        
        # Update colorbar
        cbar.update_normal(contour)
        # Ensure the ticks remain consistent
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'$10^{{{int(tick)}}}$' for tick in range(int(tick_min), int(tick_max) + 1)])
        
        # Update title
        x_value = X[x_index, 0, 0]
        title_text.set_text(f'Concentration at x = {x_value:.2f} m')
        
        # Redraw the canvas
        fig.canvas.draw_idle()
    
    # Create an interactive slider for x_index
    x_slider = IntSlider(
        min=0, max=X.shape[0]-1, step=1, value=x_index, description='x_index'
    )
    
    # Use the interact function to link the slider to the update function
    interact(update_plot, x_index=x_slider)
    
    # Return the figure and axes for further manipulation if needed
    return fig, ax
def plot_isosurfaces(X, Y, Z, concentration, iso_concentrations=None):
    """
    Plots isosurfaces of pollutant concentration.

    Parameters:
    - X, Y, Z: np.ndarray, coordinate grids
    - concentration: np.ndarray, concentration values at each grid point
    - iso_concentrations: list of float, concentration levels for isosurfaces (default: [1e-4, 1e-5, 1e-6])

    Returns:
    - fig, ax: Matplotlib figure and axes objects
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if iso_concentrations is None:
        iso_concentrations = [1e-4, 1e-5, 1e-6]  # Default concentration levels

    x = X[:, 0, 0]
    y = Y[0, :, 0]
    z = Z[0, 0, :]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot isosurfaces
    for idx, iso_concentration in enumerate(iso_concentrations):
        # Extract the isosurface
        verts, faces, normals, values = measure.marching_cubes(
            volume=concentration, level=iso_concentration, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0])
        )
        ## Adjust the vertices to align with the coordinate system
        verts[:, 0] += x.min()
        verts[:, 1] += y.min()
        verts[:, 2] += z.min()

        # Plot the isosurface
        mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                              lw=1, alpha=0.5, label=f'{iso_concentration} g/m³')

    # Add a legend
    ax.legend(title='Concentration Levels')

    # Set labels and title
    ax.set_xlabel('Downwind Distance x (m)')
    ax.set_ylabel('Crosswind Distance y (m)')
    ax.set_zlabel('Height z (m)')
    ax.set_title('Isosurfaces of Pollutant Concentration')

    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=-60)

    # Set axis limits
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min(), z.max()])

    return fig, ax


def plot_concentration_surface(X, Y, Z, concentration, z_level=10):
    """
    Plots the concentration field at a specified height z_level.

    Parameters:
    - X, Y, Z: np.ndarray, coordinate grids
    - concentration: np.ndarray, concentration values at each grid point
    - z_level: float, height at which the concentration was calculated

    Returns:
    - fig, ax: Matplotlib figure and axes objects
    """
    # Find the index in Z that is closest to the desired z_level
    z_index = (np.abs(Z[0, 0, :] - z_level)).argmin()

    # Extract the X, Y coordinates and concentration values at the desired height
    x_slice = X[:, :, z_index]
    y_slice = Y[:, :, z_index]
    concentration_slice = concentration[:, :, z_index]

    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x_slice, y_slice, concentration_slice, cmap='viridis', edgecolor='none')

    # Add a color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Concentration (g/m³)')

    # Set labels and title
    ax.set_xlabel('Downwind Distance x (m)')
    ax.set_ylabel('Crosswind Distance y (m)')
    ax.set_zlabel('Concentration (g/m³)')
    ax.set_title(f'Concentration Field at Height z ≈ {z_level} m')

    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=-60)

    return fig, ax

def plot_dispersion_ellipse(x, h, case, ax=None):
    from matplotlib.patches import Ellipse
    import flux_model as fm

    a,b,c,d = fm.pg_coeffs()
    print(a,b,c,d)
    # Convert x to kilometers
    x_km = x / 1000.0

    # Retrieve parameters for the given case
    c_case = c[case]
    d_case = d[case]
    a_case = a[case]
    b_case = b[case]

    # Calculate the angle in degrees and then convert to radians
    angle_deg = c_case - d_case * np.log(x_km)
    angle_rad = np.radians(angle_deg)

    # Calculate the tangent term
    tan_term = np.tan(angle_rad)

    # Constants
    constant = 465.11628

    # Calculate dispersion parameters
    sigma_z = a_case * x_km ** b_case
    sigma_y = constant * x_km * tan_term

    if ax == None:
        # Create figure for the ellipse representation
        fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the ellipse using the dispersion terms as the radii
    ellipse = Ellipse((0, h), 2 * sigma_y, 2 * sigma_z, edgecolor='b', facecolor='none', linestyle='--')
    ax.add_patch(ellipse)

    # Plotting the stack height point
    ax.plot(0, h, 'ro')  # Mark the stack height

    # Add arrows for sigma_y and sigma_z
    ax.arrow(0, h, sigma_y, 0, head_width=5, head_length=10, fc='g', ec='g', linestyle='-')
    ax.text(sigma_y / 2, h + 10, r'$\sigma_y$', color='g', fontsize=12)

    ax.arrow(0, h, 0, sigma_z, head_width=5, head_length=10, fc='r', ec='r', linestyle='-')
    ax.text(10, h + sigma_z / 2, r'$\sigma_z$', color='r', fontsize=12)

    # Set plot limits
    ax.set_xlim(-sigma_y * 1.5, sigma_y * 1.5)
    ax.set_ylim(0, h + sigma_z * 1.5)

    # Labels and title
    ax.set_title('Elliptical Representation of Gaussian Plume Dispersion')
    ax.set_xlabel('Crosswind Distance y (m)')
    ax.set_ylabel('Vertical Distance z (m)')
    ax.set_aspect('equal', adjustable='datalim')

    try:
        return fig, ax
    except:
        return ax