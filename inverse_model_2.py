import numpy as np 
import matplotlib.pyplot as plt
import flux_maps as maps
import flux_model as model
import pandas as pd
from scipy.special import erf
import openpyxl
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=Warning)

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

# Some spatial data for the diferent plume locations:
landfill = np.asarray([0.1436, 52.246])
sewage = np.asarray([0.157, 52.2335])
sampler = np.asarray([0.144343, 52.237111])
x_dist = np.asarray([(sampler[0]-landfill[0])*110000*np.cos(sampler[1]), 0])
y_dist = np.asarray([0, (sampler[1]-landfill[1])*110000])

x_dist2 = np.asarray([-(sampler[0]-sewage[0])*110000*np.cos(sampler[1]), 0])
y_dist2 = np.asarray([0, -(sampler[0]-sewage[0])*110000*np.cos(sampler[1])])


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


def find_coeffs(row):
    case = row['stability_class']
    A = model.pg_coeffs()[0][case]
    B = model.pg_coeffs()[1][case]
    C = model.pg_coeffs()[2][case]
    D = model.pg_coeffs()[3][case]
    return A, B, C, D


def find_basis_vectors(row):
    # Assuming the wind direction is in degrees and in a column named 'wind_direction'
    wind_direction = np.deg2rad(row['wd'])  # Convert degrees to radians
    
    # Define x_hat as the unit vector in the wind direction
    x_hat = np.asarray([-np.sin(wind_direction), -np.cos(wind_direction)])
    
    # Define y_hat as the unit vector perpendicular to x_hat (90 degrees rotated)
    y_hat = np.asarray([np.cos(wind_direction), -np.sin(wind_direction)])
    
    return x_hat, y_hat


def find_relative_distance(row):
    x_unit = row['x_hat']
    y_unit = row['y_hat']
    x_rel_dist = np.dot(x_dist, x_unit) + np.dot(y_dist, x_unit)
    y_rel_dist = np.dot(y_dist, y_unit) + np.dot(x_dist, y_unit)
    return x_rel_dist, y_rel_dist


def find_relative_distance2(row):
    x_unit = row['x_hat']
    y_unit = row['y_hat']
    x_rel_dist2 = np.dot(x_dist2, x_unit) + np.dot(y_dist2, x_unit)
    y_rel_dist2 = np.dot(y_dist2, y_unit) + np.dot(x_dist2, y_unit)
    return x_rel_dist2, y_rel_dist2


# Define some parameters that we need for the function below 
z=10
h=10
background=1.978524191




def inverse_conc_line(row, threshold=1e-4, ls=500):
    if 60<=row['wd'] <= 190:
        A = row['A']
        B = row['B']
        C = row['C']
        D = row['D']
        x = row['x_rel_dist2']
        y = row['y_rel_dist2']
        methane = row['ch4_ppm']
        u = row['ws']
        #u = np.maximum(row['ws'], 0.5)
        
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

        # Exponential terms:
        exp1 = np.exp(-((z - h) ** 2) / (2 * sigma_z ** 2))
        exp2 = np.exp(-((z + h) ** 2) / (2 * sigma_z ** 2))

        # Error functions:  
        sqrt_2 = np.sqrt(2)
        y1 = (y + ls / 2) / (sqrt_2 * sigma_y)
        y2 = (y - ls / 2) / (sqrt_2 * sigma_y)

        # Error function differences:
        erf_diff = erf(y1) - erf(y2)
        erf_diff = np.maximum(erf_diff, threshold)

        # Calculating the flux q
        q = (methane-background) * denominator / ((exp1 + exp2) * erf_diff)

        return q

    elif row['wd'] < 60 or row['wd'] > 290: 
        A = row['A']
        B = row['B']
        C = row['C']
        D = row['D']
        x = row['x_rel_dist']
        y = row['y_rel_dist']
        methane = row['ch4_ppm']
        u = row['ws']
        #u = np.maximum(row['ws'], 0.5)

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

        # Exponential terms:
        exp1 = np.exp(-((z - h) ** 2) / (2 * sigma_z ** 2))
        exp2 = np.exp(-((z + h) ** 2) / (2 * sigma_z ** 2))

        # Error functions:  
        sqrt_2 = np.sqrt(2)
        y1 = (y + ls / 2) / (sqrt_2 * sigma_y)
        y2 = (y - ls / 2) / (sqrt_2 * sigma_y)

        # Error function differences:
        erf_diff = erf(y1) - erf(y2)
        erf_diff = np.maximum(erf_diff, threshold)

        # Calculating the flux q
        q = (methane-background) * denominator / ((exp1 + exp2) * erf_diff)

        return q

    else:
        return np.nan


def g_m3_to_ppm(methane_g_m3, temperature=273.15, pressure=101325):
    # Constants
    MOLAR_MASS_METHANE = 16.04  # g/mol for methane (CH4)
    R = 8.314  # J/(mol*K), ideal gas constant
    
    # Calculate the molar concentration of methane in mol/m^3
    molar_concentration = methane_g_m3 / MOLAR_MASS_METHANE  # mol/m^3
    
    # Calculate volume at specified conditions (temperature in Kelvin, pressure in Pascals)
    molar_volume = R * temperature / pressure  # in m^3/mol (ideal gas law)
    
    # Convert molar concentration to ppm
    ppm = molar_concentration / molar_volume * 1e6
    
    return ppm


def ppm_to_g_m3(methane_ppm, temperature=273.15, pressure=101325):
    # Constants
    MOLAR_MASS_METHANE = 16.04  # g/mol for methane (CH4)
    R = 8.314  # J/(mol*K), ideal gas constant
    
    # Calculate the molar volume at specified conditions (temperature in Kelvin, pressure in Pascals)
    molar_volume = R * temperature / pressure  # in m^3/mol (ideal gas law)
    
    # Calculate molar concentration in mol/m^3
    molar_concentration = methane_ppm / 1e6 * molar_volume  # mol/m^3
    
    # Convert molar concentration to g/m^3
    methane_g_m3 = molar_concentration * MOLAR_MASS_METHANE
    
    return methane_g_m3


if __name__ == '__main__':
    data['stability_class'] = data.apply(determine_stability_class, axis=1)
    data[['A', 'B', 'C', 'D']] = data.apply(find_coeffs, axis=1, result_type='expand')
    data[['x_hat', 'y_hat']] = data.apply(find_basis_vectors, axis=1, result_type='expand')

    # Introduce the fixed lat-lon coords for the points of interest
    landfill = np.asarray([0.1436, 52.246])
    sewage = np.asarray([0.157, 52.2335])
    sampler = np.asarray([0.144343, 52.237111])

    x_dist = np.asarray([(sampler[0]-landfill[0])*110000*np.cos(sampler[1]), 0])
    y_dist = np.asarray([0, (sampler[1]-landfill[1])*110000])

    x_dist2 = np.asarray([-(sampler[0]-sewage[0])*110000*np.cos(sampler[1]), 0])
    y_dist2 = np.asarray([0, -(sampler[0]-sewage[0])*110000*np.cos(sampler[1])])

    data[['x_rel_dist', 'y_rel_dist']] = data.apply(find_relative_distance, axis=1, result_type='expand')
    data[['x_rel_dist2', 'y_rel_dist2']] = data.apply(find_relative_distance2, axis=1, result_type='expand')

    data['q'] = data.apply(inverse_conc_line, axis=1)
    window=12

    # Applyiing the necessary background offset for the methane 
    data['co2_rolling_average'] = data['co2_ppm'].ewm(span=12, adjust=False).mean()
    data['ch4_ppb'] =  data['ch4_ppb'] - 1.978524191
    data['ch4_ppm'] =  data['ch4_ppm'] - 1978.524191

    # Let's attempt to m ake the offset for the CO2 sinusoidal...
    data['co2_ppm'] =  data['co2_ppm'] - data['co2_rolling_average']

    data['ch4_ppb_ewm'] = data['ch4_ppb'].ewm(span=window, adjust=False).mean()
    data['q_rolling_avg'] = data['q'].ewm(span=window, adjust=False).mean()
    data['q_rolling_avg'] = data['q_rolling_avg'].where(data['q'].notna())

    corr_new = (data['q'].corr(data['ch4_ppb']))
    corr_new_2 = data['ch4_ppb_ewm'].corr(data['q_rolling_avg'])
    print(f'R squared: {corr_new**2}')
    print(f'R squared of rolling averages {corr_new_2**2}')


    # Plot a correlation matrix for the dataframe for certain columns

    # Potentially want to see what happens when you remove a background CO2 conc and background ch4 conc

    selected_columns = ['ch4_ppm', 'ch4_ppb', 'q', 'ws', 'temp', 'rh', 'stability_class', 'co2_ppm']  # Replace with your column names
    selected_data = data[selected_columns]
    corr_matrix = selected_data.corr()

    # Generate heat map of corr_matrix 
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()


    plt.show()
