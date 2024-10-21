"""Calculating concentrations for Gaussian mixing models of various formulations."""
import numpy as np
from scipy.special import erf

def pg_coeffs():
    """Zero indexed by PG case."""

    a = np.array([170.0, 98.0, 61.0, 32.0, 21.0, 14.0])
    b = np.array([1.09, 0.98, 0.91, 0.81, 0.75, 0.68])
    c = np.array([24.0, 18.0, 12.0, 8.0, 6.0, 4.0])
    d = np.array([2.5, 1.8, 1.1, 0.72, 0.54, 0.36])
    return a, b, c, d

def conc_point(q, u, x, y, z, h, case):
    """
    Calculate the concentration based on the given parameters.
    Exponential crosswind term, no refelection terms.

    Parameters:
    - q: Emission rate (e.g., grams per second)
    - u: Wind speed (meters per second)
    - x: Downwind distance (meters)
    - y: Crosswind distance (meters)
    - z: Vertical distance (meters)
    - h: Effective stack height (meters)
    - case: Index for atmospheric stability class (0-based index)

    Returns:
    - concentration: The calculated concentration at the given point
    """

    a, b, c, d = pg_coeffs()

    # Convert x to kilometers
    x_km = x / 1000.0

    # Initialize concentration array with zeros
    concentration = np.zeros_like(x_km)

    # Only calculate concentration where x >= 0
    positive_x = x_km >= 0.001  # Avoid log(0) and negative x values

    if np.any(positive_x):
        x_km_pos = x_km[positive_x]

        # Retrieve parameters for the given case (0-based indexing)
        c_case = c[case]
        d_case = d[case]
        a_case = a[case]
        b_case = b[case]

        # Calculate the angle in degrees and then convert to radians
        angle_deg = c_case - d_case * np.log(x_km_pos)
        angle_rad = np.radians(angle_deg)

        # Calculate the tangent term
        tan_term = np.tan(angle_rad)

        # Constants
        constant = 465.11628

        # Calculate dispersion parameters
        sigma_z = a_case * x_km_pos ** b_case
        sigma_y = constant * x_km_pos * tan_term

        # Denominator of the concentration equation
        denominator = (2 * np.pi * u * constant * x_km_pos * tan_term * sigma_z)

        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-10)

        # Extract corresponding y, z values
        y_pos = y[positive_x]
        z_pos = z[positive_x]

        # Exponential terms in the numerator
        exp1 = np.exp(-((z_pos - h) ** 2) / (2 * sigma_z ** 2))
        exp2 = np.exp(-((z_pos + h) ** 2) / (2 * sigma_z ** 2))
        exp3 = np.exp(-y_pos ** 2 / (2 * sigma_y ** 2))

        # Calculate concentration for positive x
        concentration_pos = (q / denominator) * (exp1 + exp2) * exp3

        # Assign calculated concentrations back to the full array
        concentration[positive_x] = concentration_pos

    return concentration



def conc_line(q, u, x, y, z, h, ls, case):
    """
    Calculate the concentration based on the given parameters using the new expression.
    Error function cross wind term, no reflection.

    Parameters:
    - q: Emission rate (e.g., grams per second)
    - u: Wind speed (meters per second)
    - x: Downwind distance (meters)
    - y: Crosswind distance (meters)
    - z: Vertical distance (meters)
    - h: Effective stack height (meters)
    - ls: Source width (meters) - added parameter
    - case: Index for atmospheric stability class (0-based index)

    Returns:
    - concentration: The calculated concentration at the given point
    """
    
    a, b, c, d = pg_coeffs()

    # Convert x to kilometers
    x_km = x / 1000.0

    # Initialize concentration array with zeros
    concentration = np.zeros_like(x_km)

    # Only calculate concentration where x >= 0
    positive_x = x_km >= 0.001  # Avoid log(0) and negative x values

    if np.any(positive_x):
        x_km_pos = x_km[positive_x]

        # Retrieve parameters for the given case (0-based indexing)
        c_case = c[case]
        d_case = d[case]
        a_case = a[case]
        b_case = b[case]

        # Calculate dispersion parameters
        sigma_z = a_case * x_km_pos ** b_case

        # Angle in degrees and then convert to radians
        angle_deg = c_case - d_case * np.log(x_km_pos)
        angle_rad = np.radians(angle_deg)

        # Calculate the tangent term
        tan_term = np.tan(angle_rad)

        # Constants
        constant = 465.11628

        # Calculate sigma_y
        sigma_y = constant * x_km_pos * tan_term

        # Denominator of the concentration equation
        denominator = 2 * np.sqrt(2 * np.pi) * u * a_case * x_km_pos ** b_case

        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-10)

        # Exponential terms in the numerator
        exp1 = np.exp(-((z[positive_x] - h) ** 2) / (2 * sigma_z ** 2))
        exp2 = np.exp(-((z[positive_x] + h) ** 2) / (2 * sigma_z ** 2))

        # Error function arguments
        sqrt_2 = np.sqrt(2)
        y1 = (y[positive_x] + ls / 2) / (sqrt_2 * sigma_y)
        y2 = (y[positive_x] - ls / 2) / (sqrt_2 * sigma_y)

        # Error function differences
        erf_diff = erf(y1) - erf(y2)

        # Calculate concentration for positive x
        concentration_pos = (q / denominator) * (exp1 + exp2) * erf_diff

        # Assign calculated concentrations back to the full array
        concentration[positive_x] = concentration_pos

    return concentration
