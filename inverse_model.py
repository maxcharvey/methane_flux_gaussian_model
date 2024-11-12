import numpy as np 
import matplotlib.pyplot as plt
import flux_maps as maps
import flux_model as model
import pandas as pd
from scipy.special import erf

# Load the dataset
file_path = 'data/msr_ch4_met_hrly_310524_270924.csv'
try:
    data = pd.read_csv(file_path)

    # Convert 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')

    # Filter data to only include June and July 2024
    data = data[(data['date'] >= '2024-06-01') & (data['date'] <= '2024-07-01')]
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    exit()
except pd.errors.EmptyDataError:
    print(f"Error: The file {file_path} is empty.")
    exit()
except pd.errors.ParserError:
    print(f"Error: The file {file_path} could not be parsed.")
    exit()


# Function to determine Pasquill-Gifford stability class
def determine_stability_class(row):
    hour = row['date'].hour
    wind_speed = row['ws']
    
    if 6 <= hour <= 18:  # Daytime hours (approx. 6 AM to 6 PM)
        if wind_speed < 2:
            return 0  # Very Unstable
        elif 2 <= wind_speed <= 5:
            return 1  # Moderately Unstable
        else:
            return 2  # Slightly Unstable
    else:  # Nighttime hours
        if wind_speed < 2:
            return 3  # Stable
        elif 2 <= wind_speed <= 5:
            return 4  # Slightly Stable
        else:
            return 5  # Neutral

# Apply the function to the dataframe
data['stability_class'] = data.apply(determine_stability_class, axis=1)


def find_coeffs(row):
    case = row['stability_class']
    A = model.pg_coeffs()[0][case]
    B = model.pg_coeffs()[1][case]
    C = model.pg_coeffs()[2][case]
    D = model.pg_coeffs()[3][case]
    return A, B, C, D

data[['A', 'B', 'C', 'D']] = data.apply(find_coeffs, axis=1, result_type='expand')


# Next problem is to try and find the correct x and y values based upon the wind direction
def find_basis_vectors(row):
    # Assuming the wind direction is in degrees and in a column named 'wind_direction'
    wind_direction = np.deg2rad(row['wd'])  # Convert degrees to radians
    
    # Define x_hat as the unit vector in the wind direction
    x_hat = np.asarray([np.cos(wind_direction), np.sin(wind_direction)])
    
    # Define y_hat as the unit vector perpendicular to x_hat (90 degrees rotated)
    y_hat = np.asarray([-np.sin(wind_direction), np.cos(wind_direction)])
    
    return x_hat, y_hat

data[['x_hat', 'y_hat']] = data.apply(find_basis_vectors, axis=1, result_type='expand')



# Introduce the fixed lat-lon coords for the points of interest
landfill = np.asarray([0.1436, 52.246])
sewage = np.asarray([0.157, 52.2335])
sampler = np.asarray([0.144343, 52.237111])

x_dist = np.asarray([(sampler[0]-landfill[0])*110000*np.cos(sampler[1]), 0])
y_dist = np.asarray([0, (sampler[1]-landfill[1])*110000])


# Now need to turn these into distance for each timestep based upon dot produt with wind vector 
def find_relative_distance(row):
    x_unit = row['x_hat']
    y_unit = row['y_hat']
    x_rel_dist = np.dot(x_dist, x_unit) + np.dot(y_dist, x_unit)
    y_rel_dist = np.dot(y_dist, y_unit) + np.dot(x_dist, y_unit)
    return abs(x_rel_dist), y_rel_dist

data[['x_rel_dist', 'y_rel_dist']] = data.apply(find_relative_distance, axis=1, result_type='expand')


# Define some parameters that we need for the function below 
z=10
h=10
ls=500


def inverse_conc_line(row):

    A = row['A']
    B = row['B']
    C = row['C']
    D = row['D']
    x = row['x_rel_dist']
    y = row['y_rel_dist']
    methane = row['ch4_ppm']
    u = row['ws']

    # Calculating sigma_z
    sigma_z = A * (x*0.001) ** B

    # Angle term required
    angle_deg = C - D *np.log(x/1000)
    angle_rad = np.radians(angle_deg)

    # Calculate the tangent term 
    tan_term = np.tan(angle_rad)

    # Constants
    constant = 465.11628

    # Calculate sigma_y
    sigma_y = constant * x * 0.001 * tan_term 

    # Denominator from the plume equation 
    denominator = 2 * np.sqrt(2 * np.pi) * u * A * (x/1000)**B
    denominator = np.maximum(denominator, 1e-10)

    # Exponential terms:
    exp1 = np.exp(-((z - h) ** 2) / (2 * sigma_z ** 2))
    exp2 = np.exp(-((z + h) ** 2) / (2 * sigma_z ** 2))

    # Error functions:  
    sqrt_2 = np.sqrt(2)
    y1 = (y + ls / 2) / (sqrt_2 * sigma_y)
    y2 = (y - ls / 2) / (sqrt_2 * sigma_y)

    # Error function differences:
    erf_diff = erf(y1) - erf(y2)
    erf_diff = np.maximum(erf_diff, 1e-10)

    # Calculating the flux q
    q = methane * denominator / ((exp1 + exp2) * erf_diff)

    return q

data['q'] = data.apply(inverse_conc_line, axis=1)

fig, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(data['date'], data['q'])
plt.show()
