import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
from statistics import mean


def load_pmt_data(file_name: str):
    # search for cut data, if found return it
    cut_files = glob.glob(file_name[:-4] + '_cut*.csv')
    if cut_files:
        print('Cut data found')
        return pd.read_csv(cut_files[0], header=None, delimiter=';')

    # load csv file as file
    with open(file_name, 'r') as file:
        lines = file.readlines()
        file.close()

    # delete first line and save as temp file
    new_file_name = file_name[:-4] + '_temp.csv'
    with open(new_file_name, 'w') as file:
        for line in lines[1:]:
            file.write(line)
        file.close()

    # load temp file as data
    data = pd.read_csv(new_file_name, header=None, delimiter=';')

    # cut off first 6 columns
    data = data.iloc[:, 6:]

    # delete temp file
    os.remove(new_file_name)

    # save data as csv with prefix 'cut_' in same directory
    file_name_list = file_name.split('/')
    file_name_list[-1] = 'cut_' + file_name_list[-1]
    new_file_name = '/'.join(file_name_list)
    data.to_csv(new_file_name, index=False)

    # return data
    return data


def find_baseline(datarow: pd.Series, max_std: float = 3.5, window_size: int = 30) -> tuple[bool, float, float]:
    # set variables
    std = max_std + 16000
    baseline = 8000

    # iterate over data with window_size
    for i in range(len(datarow)//window_size):
        std = datarow[window_size*i:window_size*i+window_size].std()
        baseline = datarow[window_size*i:window_size*i+window_size].mean()
        if std < max_std:
            # baseline found, break loop
            break
    if std < max_std:
        # baseline found, return baseline and std
        return True, baseline, std
    else:
        # baseline not found, return mean and std of whole data
        return False, datarow.mean(), datarow.std()


def find_time_and_energy_of_peaks(
        datarow: pd.Series,
        baseline: float,
        std: float,
        n_threshold_stds = 10,
        time_unit: float = 2.0,
) -> (list, list):
    # define variables and prepare lists
    peaks = []
    current_peak_list = []
    peak_index_list = []
    peak_current = False
    energy_sum = 0
    energy_and_time_list = []

    # find values above threshold, append value to current peak, if value is below threshold,
    # append peak to peaks and empty list
    for _index, value in datarow.items():
        if value > baseline + std*n_threshold_stds:
            # value is above threshold
            peak_current = True
            energy_sum += value-baseline
            current_peak_list.append((_index, value))
            peak_index_list.append(_index)
        if value < baseline + std*n_threshold_stds and peak_current:
            # value is below threshold, peak ended
            energy_and_time_list.append([mean(peak_index_list)*time_unit, energy_sum])
            peaks.append(current_peak_list)
            current_peak_list = []
            peak_index_list = []
            peak_current = False
        else:
            # no peak
            pass

    return peaks, energy_and_time_list


if __name__ == '__main__':
    data = load_pmt_data(r'../data/20250317/led_on_highvoltage_ptfe_block_1/RAW/DataR_CH0@DT5730S_59483_led_on_highvoltage_ptfe_block_1.CSV')
    bad_counter = 0
    for _, datarow in data.iterrows():
        good, baseline, std = find_baseline(datarow)
        if not good:
            bad_counter += 1
        peaks, energy_and_time_list = find_time_and_energy_of_peaks(datarow, baseline, std)
        if len(peaks) > 0:
            print('at least one peak')
            print(peaks)
            print(energy_and_time_list)
            print(f'baseline: {baseline}, std: {std}')
