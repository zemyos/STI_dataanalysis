from analysis_functions.pmt_caen import find_baseline, find_time_and_energy_of_peaks, load_pmt_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob


folder_list = glob.glob('../data/20250317/*highvoltage*')
run_id_list = [folder.split('/')[-1] for folder in folder_list]


# prep plot
ax = plt.axes()


for run in run_id_list:
    file_names = glob.glob(f'../data/20250317/{run}/RAW/Data*.CSV')
    run_data = []
    bad_counter = 0
    for file_name in file_names:
        data = load_pmt_data(file_name)
        for _, datarow in data.iterrows():
            good, baseline, std = find_baseline(datarow)
            if good:
                peaks, energy_and_time_list = find_time_and_energy_of_peaks(datarow, baseline, std)
                run_data.extend(energy_and_time_list)
            else:
                bad_counter += 1
    print(f'Run {run} had {bad_counter} bad data points')
    if len(run_data) > 0:
        ax.scatter(*zip(*run_data), label=run, alpha=0.5, marker='.')
    plt.legend()
    plt.savefig('test.png')

