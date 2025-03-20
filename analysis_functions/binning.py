import pandas as pd


def select_data(
        file_name: str,
        x_lim_low: float,
        x_lim_high: float,
        prefix: str,
        save: bool = True,
):
    # read data from csv
    data = pd.read_csv(file_name, header=4)

    # select data according to time stamp
    data_selected = data.loc[(data['Time'] >= x_lim_low) & (data['Time'] <= x_lim_high)]

    # save data
    if save:
        file_name_list = file_name.split('/')
        file_name_list[-1] = prefix + file_name_list[-1]
        new_file_name = '/'.join(file_name_list)
        data_selected.to_csv(new_file_name, index=False)

    return data_selected


def bin_data(
    bin_size: int,
    file_name: str = None,
    data: pd.DataFrame = None,
    prefix: str = 'binned_',
    save: bool = True,
):
    # load data
    if file_name and data is None:
        data = pd.read_csv(file_name)

    # calculate rolling average
    # data['binned_time'] = data['Time'].rolling(bin_size).mean()
    data['binned_ampl'] = data['Ampl'].rolling(bin_size).mean()

    # select every 10th value and drop old data
    df_binned = data.iloc[::bin_size, :]
    df_binned = df_binned.drop('Ampl', axis=1)
    # df_binned = df_binned.drop('Time', axis=1)

    # save data
    if save:
        if file_name is None:
            print('No filename provided cannot save')
            return df_binned
        file_name_list = file_name.split('/')
        file_name_list[-1] = prefix + file_name_list[-1]
        new_file_name = '/'.join(file_name_list)
        df_binned.to_csv(new_file_name, index=False)
    return df_binned
