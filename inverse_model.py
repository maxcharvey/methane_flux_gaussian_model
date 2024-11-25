import numpy as np 
import matplotlib.pyplot as plt
import flux_maps as maps
import flux_model as model
import pandas as pd
from scipy.special import erf
import openpyxl

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
    x_hat = np.asarray([-np.sin(wind_direction), -np.cos(wind_direction)])
    
    # Define y_hat as the unit vector perpendicular to x_hat (90 degrees rotated)
    y_hat = np.asarray([np.cos(wind_direction), -np.sin(wind_direction)])
    
    return x_hat, y_hat


data[['x_hat', 'y_hat']] = data.apply(find_basis_vectors, axis=1, result_type='expand')


# Introduce the fixed lat-lon coords for the points of interest
landfill = np.asarray([0.1436, 52.246])
sewage = np.asarray([0.157, 52.2335])
sampler = np.asarray([0.144343, 52.237111])

x_dist = np.asarray([(sampler[0]-landfill[0])*110000*np.cos(sampler[1]), 0])
y_dist = np.asarray([0, (sampler[1]-landfill[1])*110000])

x_dist2 = np.asarray([(sampler[0]-sewage[0])*110000*np.cos(sampler[1]), 0])
y_dist2 = np.asarray([(sampler[0]-sewage[0])*110000*np.cos(sampler[1]), 0])


# Now need to turn these into distance for each timestep based upon dot produt with wind vector 
def find_relative_distance(row):
    x_unit = row['x_hat']
    y_unit = row['y_hat']
    x_rel_dist = np.dot(x_dist, x_unit) + np.dot(y_dist, x_unit)
    y_rel_dist = np.dot(y_dist, y_unit) + np.dot(x_dist, y_unit)
    return x_rel_dist, y_rel_dist


data[['x_rel_dist', 'y_rel_dist']] = data.apply(find_relative_distance, axis=1, result_type='expand')


# Define some parameters that we need for the function below 
z=10
h=10
ls=500
background=1.978524191

def inverse_conc_line(row):

    if row['x_rel_dist'] <= 100:
        return np.nan  # or return None

    else: 
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

        # Exponential terms:
        exp1 = np.exp(-((z - h) ** 2) / (2 * sigma_z ** 2))
        exp2 = np.exp(-((z + h) ** 2) / (2 * sigma_z ** 2))

        # Error functions:  
        sqrt_2 = np.sqrt(2)
        y1 = (y + ls / 2) / (sqrt_2 * sigma_y)
        y2 = (y - ls / 2) / (sqrt_2 * sigma_y)

        # Error function differences:
        erf_diff = erf(y1) - erf(y2)
        erf_diff = np.maximum(erf_diff, 1e-2)

        # Calculating the flux q
        q = (methane-background) * denominator / ((exp1 + exp2) * erf_diff)

        return q/1000

data['q'] = data.apply(inverse_conc_line, axis=1)
window=12

data['q_rolling_avg'] = (
    data['q']
    .ewm(span=window, adjust=False)
    .mean()
)



#data['q_rolling_avg'] = data['q'].ewm(span=window, adjust=True).mean()


def plot_time_series_subplot(ax, x, y, ylabel, label, color, window):
    rolling_avg = y.ewm(span=window, adjust=False).mean()
    ax.plot(x, y, label=label, color=color, alpha=0.5)
    ax.plot(x, rolling_avg, label=f'{label} (Smoothed Avg)', color='black', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.legend()
   

def plot_combined_time_series(data, window=12):
    fig, axes = plt.subplots(7, 1, figsize=(12, 9), sharex=True)

    # Plot Temperature
    plot_time_series_subplot(axes[0], data['date'], data['temp'], 'Temperature (°C)', 'Temperature (°C)', 'tab:red', window)

    # Plot Methane Concentration
    plot_time_series_subplot(axes[4], data['date'], data['ch4_ppb'], 'Methane Concentration (ppb)', 'Methane (CH₄)', 'tab:green', window)

    # Plot Relative Humidity
    plot_time_series_subplot(axes[2], data['date'], data['rh'], 'Relative Humidity (%)', 'Relative Humidity', 'tab:blue', window)

    # Plot Wind Speed
    plot_time_series_subplot(axes[3], data['date'], data['ws'], 'Wind Speed (units)', 'Wind Speed', 'tab:purple', window)

    # Plot Stability Class
    stability_class_numeric = data['stability_class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5})
    plot_time_series_subplot(axes[1], data['date'], data['stability_class'], 'Stability Class', 'Stability Class', 'tab:cyan', window)
    
    # Plot calculated methane flux from inverse model
    plot_time_series_subplot(axes[5], data['date'], data['q'], 'Source flux', 'Source flux', 'tab:cyan', window)
    
    plot_time_series_subplot(axes[6], data['date'], data['wd'], 'Wind direction', 'Wind direction', 'tab:cyan', window)

    plt.tight_layout()
    plt.show()

plot_combined_time_series(data)



# The correlation between the conc at source and the conc at the measurement point

corr_s_m = (data['ch4_ppb'].ewm(span=window, adjust=False).mean()).corr(data['q_rolling_avg'].where(data['q'].notna()))
corr_t_m = (data['temp'].ewm(span=window, adjust=False).mean()).corr(data['q_rolling_avg'])
coor_s_w = (data['ws'].ewm(span=window, adjust=False).mean()).corr(data['q_rolling_avg'])
coor_m_w = (data['ws'].ewm(span=window, adjust=False).mean()).corr(data['ch4_ppb'].ewm(span=window, adjust=False).mean())
corr_sc_q = (data['stability_class'].ewm(span=window, adjust=False).mean()).corr(data['q_rolling_avg'])


corr_new = (data['q'].corr(data['ch4_ppb']))
print(corr_new)


# Bit of a testing area here

data['ch4_ppb_ewm'] = data['ch4_ppb'].ewm(span=window, adjust=False).mean()
data['q_rolling_avg'] = data['q'].ewm(span=window, adjust=False).mean()
data['q_rolling_avg'] = data['q_rolling_avg'].where(data['q'].notna())

# Define lag range
max_lag = 30
correlations = []

# Calculate correlation for each lag
for lag in range(-max_lag, max_lag + 1):
    shifted_q_avg = data['q_rolling_avg'].shift(lag)
    corr = data['ch4_ppb_ewm'].corr(shifted_q_avg)
    correlations.append((lag, corr))

# Find optimal lag
optimal_lag, max_corr = max(correlations, key=lambda x: x[1])

print(f"Optimal lag: {optimal_lag}")
print(f"Maximum correlation: {max_corr}")

# Optionally, visualize the results
import matplotlib.pyplot as plt



lags, corr_values = zip(*correlations)
plt.plot(lags, corr_values, marker='o')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Correlation vs Lag')
plt.xlabel('Lag')
plt.ylabel('Correlation Coefficient')
plt.grid(True)
plt.show()

fig, ax3 = plt.subplots(figsize=(12, 6))
ax3.scatter(data['q'], data['ch4_ppb'], label='Methane Concentration vs. Source Flux', color='tab:blue', alpha=0.5)
ax3.set_yscale('log')
ax3.set_xscale('log')
plt.show()

print(data)