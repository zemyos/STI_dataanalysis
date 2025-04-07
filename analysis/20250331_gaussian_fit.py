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


def triple_gauss_function(x, a1, x01, sigma1, a2, x02, sigma2, a3, x03, sigma3):
    return gauss_function(x, a1, x01, sigma1) + gauss_function(x, a2, x02, sigma2) + gauss_function(x, a3, x03, sigma3)


def triple_gauss_function_shared_sigma(x, a1, x01, sigma1, a2, x02, a3, x03):
    return gauss_function(x, a1, x01, sigma1) + gauss_function(x, a2, x02, sigma1) + gauss_function(x, a3, x03, sigma1)


def triple_gauss_function_shared_sigma_shared_x01(x, a1, x01, sigma1, a2, a3):
    return (gauss_function(x, a1, x01, sigma1) + gauss_function(x, a2, x01*2, sigma1) +
            gauss_function(x, a3, x01*3, sigma1))


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
        # hist_x = hist_x_all[33:260]
        # hist_y = hist_y_all[33:260]
        hist_x = hist_x_all[33:200]
        hist_y = hist_y_all[33:200]
        p0 = [popt[0], popt[1], popt[2], popt[0]/10, 2*popt[1], popt[2]]
        popt2, pcov2 = curve_fit(double_gauss_function, hist_x, hist_y, p0=p0)

        # fit triple Gaussian to histogram
        hist_x = hist_x_all[33:300]
        hist_y = hist_y_all[33:300]
        p0 = [popt[0], popt[1], popt[2], popt[0]/10, 2*popt[1], popt[2], popt[0]/100, 3*popt[1], popt[2]]
        popt3, pcov3 = curve_fit(triple_gauss_function, hist_x, hist_y, p0=p0)

        # fit triple Gaussian to histogram with shared sigma
        hist_x = hist_x_all[33:300]
        hist_y = hist_y_all[33:300]
        p0 = [popt[0], popt[1], popt[2], popt[0]/10, 2*popt[1], popt[0]/100, 3*popt[1]]
        popt4, pcov4 = curve_fit(triple_gauss_function_shared_sigma, hist_x, hist_y, p0=p0)

        # fit triple Gaussian to histogram with shared sigma and shared x01
        hist_x = hist_x_all[33:300]
        hist_y = hist_y_all[33:300]
        p0 = [popt[0], popt[1], popt[2], popt[0]/10, popt[0]/100]
        popt5, pcov5 = curve_fit(triple_gauss_function_shared_sigma_shared_x01, hist_x, hist_y, p0=p0)


        # plot Gaussian fit
        # x_fit = np.linspace(min(hist_x_all), max(hist_x_all), 1000)
        # y_fit = triple_gauss_function(x_fit, *popt3)
        # plt.plot(x_fit, y_fit, 'g-', label='Triple Gaussian fit')

        x_fit = np.linspace(min(hist_x_all), max(hist_x_all), 1000)
        y_fit = triple_gauss_function_shared_sigma_shared_x01(x_fit, *popt5)
        plt.plot(x_fit, y_fit, 'g-', label='Triple Gaussian fit shared x01')

        # # plot two gaussians individually
        y_fit1 = gauss_function(x_fit, *popt2[:3])
        plt.plot(x_fit, y_fit1, 'c--', label='1 Photon Gaussian')
        y_fit2 = gauss_function(x_fit, *popt2[3:])
        plt.plot(x_fit, y_fit2, 'm--', label='2 Photon Gaussian')

        # plot three gaussians individually
        y_fit1 = gauss_function(x_fit, *popt3[:3])
        plt.plot(x_fit, y_fit1, 'k--', label='1 Photon Gaussian')
        y_fit2 = gauss_function(x_fit, *popt3[3:6])
        plt.plot(x_fit, y_fit2, 'w--', label='2 Photon Gaussian')
        y_fit_3 = gauss_function(x_fit, *popt3[6:])
        plt.plot(x_fit, y_fit_3, 'g--', label='3 Photon Gaussian')

        # plt shared sigma gaussians individually
        # y_fit1 = gauss_function(x_fit, popt4[0], popt4[1], popt4[2])
        # plt.plot(x_fit, y_fit1, 'r-', label='1 Photon Gaussian')
        # y_fit2 = gauss_function(x_fit, popt4[3], popt4[4], popt4[2])
        # plt.plot(x_fit, y_fit2, 'b-', label='2 Photon Gaussian')
        # y_fit_3 = gauss_function(x_fit, popt4[5], popt4[6], popt4[2])
        # plt.plot(x_fit, y_fit_3, 'y-', label='3 Photon Gaussian')

        y_fit1 = gauss_function(x_fit, popt5[0], popt5[1], popt5[2])
        plt.plot(x_fit, y_fit1, 'r--', label='1 Photon Gaussian shared x01')
        y_fit2 = gauss_function(x_fit, popt5[3], popt5[1]*2, popt5[2])
        plt.plot(x_fit, y_fit2, 'b--', label='2 Photon Gaussian shared x01')
        y_fit_3 = gauss_function(x_fit, popt5[4], popt5[1]*3, popt5[2])
        plt.plot(x_fit, y_fit_3, 'y--', label='3 Photon Gaussian shared x01')

        plt.legend()
        plt.yscale('log')
        plt.ylim(1.0, 1e4)
        plt.savefig(f'plots/{date_string}/{run}_energy_histogram.png', dpi=600)
        plt.close()