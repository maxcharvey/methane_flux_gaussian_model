from inverse_model_2 import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


# Now we can do some differnt plottings here: 

# In particular let's look at applying a sinusoidal offset - we want to optimise a sinusoidal offset to the CO2 data:

print(data)

# We can use the curve_fit function from scipy to fit a sinusoidal function to the data
# The function we want to fit is of the form:
# f(x) = A * sin(B * x + C) + Dx + E

# Define the function we want to fit
def sinusoidal(x, A, C, D, E):
    return A * np.sin(np.pi/12 * x + C) + D * x + E

# Define the x values (time) and y values (CO2 concentration)
y = data['co2_ppm'].interpolate(method='linear').values
# Fit the sinusoidal function to the data after we have removed any NaN values
x = np.arange(len(y))
popt, pcov = curve_fit(sinusoidal, x, y)

# Generate the y values for the fitted function
data['offset_co2'] = y - sinusoidal(x, *popt) 


def plot_time_series_subplot(ax, x, y, ylabel, label, color, window):
    rolling_avg = y.ewm(span=window, adjust=False).mean()
    ax.plot(x, y, label=label, color=color, alpha=0.5)
    ax.plot(x, rolling_avg, label=f'{label} (Smoothed Avg)', color='black', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_combined_time_series2(data, window=12):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Plot Methane Concentration
    plot_time_series_subplot(axes[0], data['date'], data['ch4_ppb'], 'Methane Concentration (ppb)', 'Methane (CH₄)', 'tab:green', window)
    
    # Plot calculated methane flux from inverse model
    plot_time_series_subplot(axes[1], data['date'], data['q'], 'Source flux (ppb/meter/sec)', 'Source flux', 'tab:cyan', window)
    
    # Need to try and look at a potential offset here for the CO2 that is based upon daily variation
    plot_time_series_subplot(axes[2], data['date'], data['offset_co2'], 'CO2 Concentration (ppm)', 'CO2 (ppm)', 'tab:cyan', window)

    plt.tight_layout()
    plt.show()


# Figure 1: Time series of methane concentration, calculated methane flux, and CO2 concentration
plot_combined_time_series2(data)


# Figure 2: Correlation matrix:
selected_columns = ['ch4_ppm', 'ch4_ppb', 'q', 'ws', 'temp', 'rh', 'stability_class', 'co2_ppm', 'offset_co2']  # Replace with your column names
selected_data = data[selected_columns]
corr_matrix = selected_data.corr()

# Generate heat map of corr_matrix 
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()


