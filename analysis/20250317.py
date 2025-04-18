from analysis_functions.pmt_caen import find_baseline, find_time_and_energy_of_peaks, load_pmt_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit

date_string = '20250321'

folder_list = glob.glob(f'../data/{date_string}/*')
run_id_list = [folder.split('/')[-1] for folder in folder_list]


def gauss(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))

# prep plot
ax = plt.axes()

# 2d histogram (heatmap)
for run in run_id_list:
    file_names = glob.glob(f'../data/{date_string}/{run}/RAW/Data*.CSV')
    run_data = []
    bad_counter = 0
    for file_name in file_names:
        data = load_pmt_data(file_name)
        for _, datarow in data.iterrows():
            good, baseline, std = find_baseline(datarow, max_std=4.0, window_size=20)
            if True:
                peaks, energy_and_time_list = find_time_and_energy_of_peaks(datarow, baseline, std)
                run_data.extend(energy_and_time_list)
            if not good:
                bad_counter += 1
    print(f'Run {run} had {bad_counter} bad data points')
    if len(run_data) > 0:
        plt.hist2d(*zip(*run_data), bins=50, norm="log")
        plt.title(run)
        plt.xlabel('Time[ns]')
        plt.ylabel('Energy[ADC]')
        plt.savefig(f'plots/{run}_heatmap.png', dpi=600)
        plt.close()
        # break

    if len(run_data) > 0:
        run_data = np.array(run_data)
        xdata = run_data[(run_data[:, 0] > 150) & (run_data[:, 0] < 200)][:, 0]
        ydata = run_data[(run_data[:, 0] > 150) & (run_data[:, 0] < 200)][:, 1]
        parameters, covariance = curve_fit(gauss, xdata, ydata)
        plt.hist(run_data[(run_data[:, 0] > 150) & (run_data[:, 0] < 200)][:, 1], bins=2000, histtype='step', label=run)
        plt.yscale('log')
        plt.xlabel('Energy[ADC]')
        plt.ylabel('Counts')
        plt.legend()
        plt.savefig(f'plots/{run}_energy_histogram.png', dpi=600)
        plt.close()
        # break

# 1d histogram (energy) for time between 150 and 200ns
for run in run_id_list:
    break
    file_names = glob.glob(f'../data/{date_string}/{run}/RAW/Data*.CSV')
    run_data = []
    bad_counter = 0
    for file_name in file_names:
        data = load_pmt_data(file_name)
        for _, datarow in data.iterrows():
            good, baseline, std = find_baseline(datarow, max_std=4.0, window_size=20)
            if True:
                peaks, energy_and_time_list = find_time_and_energy_of_peaks(datarow, baseline, std)
                run_data.extend(energy_and_time_list)
            if not good:
                bad_counter += 1
    print(f'Run {run} had {bad_counter} bad data points')
    if len(run_data) > 0:
        run_data = np.array(run_data)
        plt.hist(run_data[(run_data[:, 0] > 150) & (run_data[:, 0] < 200)][:, 1], bins=2000, histtype='step', label=run)
        plt.xlabel('Energy[ADC]')
        plt.ylabel('Counts')
        plt.legend()
        plt.savefig(f'plots/{run}_energy_histogram.png', dpi=600)
        plt.close()
        # break



