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
    data = data[(data['date'] >= '2024-06-01') & (data['date'] <= '2024-7-01')]
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

data['q'] = data.apply(lambda row: inverse_conc_line(row, threshold=1e-4), axis=1)
window=12


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
# Corrected condition for 'q_1'
data['q_1'] = data['q'].where((data['wd'] < 60) | (data['wd'] > 290))

# Corrected condition for 'q_2'
data['q_2'] = data['q'].where((data['wd'] >= 60) & (data['wd'] <= 190))

data['q_grams'] = ppm_to_g_m3(data['q'])
data['q1_grams'] = ppm_to_g_m3(data['q_1'])
data['q2_grams'] = ppm_to_g_m3(data['q_2'])


def plot_time_series_subplot(ax, x, y, ylabel, label, color, window, alpha):
    rolling_avg = y.ewm(span=window, adjust=False).mean()
    ax.plot(x, y, label=label, color=color, alpha=alpha)
    ax.plot(x, rolling_avg, label=f'{label} (Smoothed Avg)', color='black', linestyle='--')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=12)


def plot_time_series_subplot_split(ax, x, y, ylabel, label, color, window, alpha):
    ax.plot(x, y, label=label, color=color, alpha=alpha)
    ax.set_xlabel('Date')
    ax.legend()


def plot_combined_time_series2(data, window=12):
    #fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)


    # Plot Methane Concentration
    plot_time_series_subplot(axes[0], data['date'], data['ch4_ppm'], 'Sampler Methane \n Concentration (ppm)', 'Methane (CH₄)', 'tab:purple', window, 0.5)
    
    plot_time_series_subplot(axes[1], data['date'], data['q_grams'], 'Source flux \n (g/meter/sec)', 'Source Flux', 'tab:orange', window, 0.5)
    #Plot calculated methane flux from inverse model
    #plot_time_series_subplot_split(axes[2], data['date'], data['q1_grams'], 'Source flux (ppm/meter/sec)', 'Landfill Source Flux', 'tab:blue', window, 0.75)
    #plot_time_series_subplot_split(axes[2], data['date'], data['q2_grams'], 'Source flux (ppm/meter/sec)', 'Sewage Source Flux', 'tab:green', window, 0.75)
    #plot_time_series_subplot(axes[2], data['date'], data['q_grams'], 'Source flux (g/meter/sec)', 'Combined Source Flux', 'tab:gray', window, 0.35)

    # Need to try and look at a potential offset here for the CO2 that is based upon daily variation
    #plot_time_series_subplot(axes[1], data['date'], data['offset_co2'], 'Offeset sampler CO2 \n Concentration (ppm)', 'CO2 (ppm)', 'tab:cyan', window, 0.5)
    
    #axes[3].set_xlabel('Date')
    axes[1].set_xlabel('Date', fontsize=12)
    #plt.savefig('inverse_model2_2plots.png', dpi=500)
    for a in axes:
        a.tick_params(axis='both', labelsize=12)

    #plt.tight_layout()
    plt.show()


# Figure 1: Time series of methane concentration, calculated methane flux, and CO2 concentration
plot_combined_time_series2(data)



window=12

# Applyiing the necessary background offset for the methane 
data['co2_rolling_average'] = data['offset_co2'].ewm(span=12, adjust=False).mean()


data['ch4_ppb'] =  data['ch4_ppb'] - 1.978524191
data['ch4_ppm'] =  data['ch4_ppm'] - 1978.524191


data['ch4_ppb_ewm'] = data['ch4_ppb'].ewm(span=window, adjust=False).mean()
data['q_rolling_avg'] = data['q'].ewm(span=window, adjust=False).mean()
data['q_rolling_avg'] = data['q_rolling_avg'].where(data['q'].notna())
data['co2_ppm_ewm'] = data['co2_ppm'].ewm(span=window, adjust=False).mean()

filtered_data = data[data['q_grams'] < 0.75]


corr_new = (data['q'].corr(data['ch4_ppb']))
corr_new_2 = data['ch4_ppb_ewm'].corr(data['q_rolling_avg'])
print(f'R squared: {corr_new**2}')
print(f'R squared of rolling averages {corr_new_2**2}')
print(np.nanmean(filtered_data['q_grams']))

corr_new2 = (data['ch4_ppb_ewm'].corr(data['co2_ppm_ewm']))
print(corr_new2)