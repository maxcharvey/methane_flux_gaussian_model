import numpy as np 
import matplotlib.pyplot as plt
import flux_maps as maps
import flux_model as model
import pandas as pd


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










