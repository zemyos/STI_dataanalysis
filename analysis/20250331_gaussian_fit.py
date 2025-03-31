from analysis_functions.pmt_caen import find_baseline, find_time_and_energy_of_peaks, load_pmt_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
from scipy.optimize import curve_fit

# prep variables
data_date_string = '20250321'
folder_list = glob.glob(f'../data/{data_date_string}/*')
run_id_list = [folder.split('/')[-1] for folder in folder_list]

# make plot folder with current date
now = datetime.now()
date_string = now.strftime("%Y%m%d")
if not os.path.exists(f'plots/{date_string}'):
    os.makedirs(f'plots/{date_string}')
    print(f'Created folder plots/{date_string}')
else:
    print(f'Folder plots/{date_string} already exists')


# Gaussian function
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def double_gauss_function(x, a1, x01, sigma1, a2, x02, sigma2):
    return gauss_function(x, a1, x01, sigma1) + gauss_function(x, a2, x02, sigma2)

# create 1d histogram (energy) for time between 150 and 200ns

for run in run_id_list:
    file_names = glob.glob(f'../data/{data_date_string}/{run}/RAW/Data*.CSV')
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
        xdata = run_data[(run_data[:, 0] > 150) & (run_data[:, 0] < 200)][:, 0]
        ydata = run_data[(run_data[:, 0] > 150) & (run_data[:, 0] < 200)][:, 1]
        hist = plt.hist(run_data[(run_data[:, 0] > 150) & (run_data[:, 0] < 200)][:, 1],
                        bins=500, histtype='step', label=run)
        # plt.yscale('log')
        plt.xlabel('Energy[ADC]')
        plt.ylabel('Counts')
        hist_x_all = hist[1][:-1]
        hist_y_all = hist[0]

        hist_x = hist_x_all[33:105]
        hist_y = hist_y_all[33:105]

        # estimate mean an stddev
        mean = np.sum(hist_x * hist_y) / np.sum(hist_y)
        stddev = np.sqrt(np.sum(hist_y * (hist_x - mean)**2) / np.sum(hist_y))

        # fit Gaussian to histogram
        popt, pcov = curve_fit(gauss_function, hist_x, hist_y, p0=[max(hist_y), mean, stddev])

        # fit double Gaussian to histogram
        # x02 should be 2*x01, sigma2 should be sigma1, put those values in the p0. Use values from the first fit
        hist_x = hist_x_all[33:260]
        hist_y = hist_y_all[33:260]
        p0 = [popt[0], popt[1], popt[2], popt[0]/10, 2*popt[1], popt[2]]
        popt2, pcov2 = curve_fit(double_gauss_function, hist_x, hist_y, p0=p0)

        # plot Gaussian fit
        x_fit = np.linspace(min(hist_x_all), max(hist_x_all), 1000)
        y_fit = double_gauss_function(x_fit, *popt2)
        plt.plot(x_fit, y_fit, 'g-', label='Double Gaussian fit')

        # plot two gaussians individually
        y_fit1 = gauss_function(x_fit, *popt2[:3])
        plt.plot(x_fit, y_fit1, 'r-', label='1 Photon Gaussian')
        y_fit2 = gauss_function(x_fit, *popt2[3:])
        plt.plot(x_fit, y_fit2, 'b-', label='2 Photon Gaussian')

        plt.legend()
        plt.savefig(f'plots/{date_string}/{run}_energy_histogram.png', dpi=600)
        plt.close()