from inverse_model_2 import inverse_conc_line, determine_stability_class, find_coeffs, find_basis_vectors, find_relative_distance, find_relative_distance2, ppm_to_g_m3
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset in:
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


# First experiment to be run is sensitivity to the denominator restriction

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

q_maxes = []
threshold_values = np.logspace(-12,-1, 100)

q_averages = []
ls_values = np.linspace(100,800, 100)

q_averages2 = []
threshold_values2 = np.logspace(-20,-1, 200)


for i in range(len(threshold_values)):
    threshold = threshold_values[i]
    data['q'] = data.apply(lambda row: inverse_conc_line(row, threshold=threshold), axis=1)
    q_maxes.append(np.nanmax(data['q']))

for i in range(len(ls_values)):
    filtered_data=None
    ls = ls_values[i]
    data['q'] = data.apply(lambda row: inverse_conc_line(row, ls=ls), axis=1)
    filtered_data = data[data['q'] < 0.75]
    q_averages.append(np.nanmean(data['q']))

for i in range(len(threshold_values2)):
    filtered_data=None
    threshold = threshold_values2[i]
    data['q'] = data.apply(lambda row: inverse_conc_line(row, threshold=threshold), axis=1)
    data['q_grams'] = ppm_to_g_m3(data['q'])
    filtered_data = data[data['q_grams'] < 0.75]
    temp = np.nanmean(filtered_data['q_grams']).real
    q_averages2.append(temp)

fig, ax1 = plt.subplots(figsize=(10, 12), nrows=3)

ax1[0].plot(threshold_values, ppm_to_g_m3(np.array(q_maxes)), 'k-')
ax1[0].set_xscale('log')  # Set the x-axis to a logarithmic scale
ax1[0].set_yscale('log')  # Set the y-axis to a logarithmic scale
ax1[0].set_xlabel('Minimum Denominator Value')
ax1[0].set_ylabel('Maximum Calculated Methane \n Flux Intensity (g/m/s)')


ax2 = ax1[1].twinx()
ax1[1].plot(ls_values, ppm_to_g_m3(np.array(q_averages)*ls_values), 'k-', label='Total emissions')
ax2.plot(ls_values, ppm_to_g_m3(np.array(q_averages)), color='cyan', ls='-',alpha=0.5, label='Emission rate')


ax2.set_ylabel('Emission rate from \n line source (g/m/s)')
ax1[1].set_xlabel('Line source length (m)')
ax1[1].set_ylabel('Total emission from \n line source (g/s)')
ax1[1].legend(loc='center left')
ax2.legend(loc='center right')


ax1[2].plot(threshold_values2, q_averages2, 'k-') 
ax1[2].set_xlabel('Maximimum denominator value')
ax1[2].set_ylabel('Total emission from \n line source (g/s)')
ax1[2].set_xscale('log')  # Set the x-axis to a logarithmic scale
ax1[2].set_yscale('log')  # Set the y-axis to a logarithmic scale


plt.savefig('sensitivity_experiment_4.png')
plt.show()