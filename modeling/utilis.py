import pandas as pd
import numpy as np
from tqdm import trange
from time import sleep


def load_data(args, filepath=None, columns_list=None, norm_inf=False, rnn_full_name=None):
    if filepath is None and args.filepath is not None:
        filepath = args.val_file_name

    if columns_list is None and (args.inputs_list is not None or args.outputs_list is not None):
        columns_list = list(set(args.inputs_list).union(set(args.outputs_list)))

    if type(filepath) == list:
        filepaths = filepath
    else:
        filepaths = [filepath]

    all_dfs = []  # saved separately to get normalization
    all_time_axes = []

    print('Loading data files:')
    sleep(0.1)
    for file_number in trange(len(filepaths)):
        one_filepath = filepaths[file_number]
        # Load dataframe
        # print('loading data from ' + str(one_filepath))
        # print('')
        df_dense = pd.read_csv(one_filepath, comment='#')
        max_pos = df_dense['s.position'].abs().max()
        # print('Max_pos: {}'.format(max_pos))
        # Here time calculation


        if 'time' in df_dense.columns and 'dt' not in df_dense.columns:
            dt = [0.0]
            row_iterator = df_dense.iterrows()
            _, last = row_iterator.next()  # take first item from row_iterator
            for i, row in row_iterator:
                dt.append(row['value']-last['value'])
                last = row
            df_dense['dt'] = np.array(dt)
        elif 'dt' in df_dense.columns and 'time' not in df_dense.columns:
            dt = df_dense['dt']
            t = dt.cumsum()
            df_dense['time'] = t

        # You can shift dt by one time step to know "now" the timestep till the next row
        if args.cheat_dt:
            if 'dt' in df_dense:
                df_dense['dt'] = df_dense['dt'].shift(-1)
                df_dense = df_dense[:-1]

        for i in range(args.downsampling):
            df = df_dense.iloc[i::args.downsampling].reset_index()

            # Get time axis as separate Dataframe
            if 'time' in df.columns:
                t = df['time']
            else:
                t = pd.Series([])
                t.rename('time', inplace=True)

            all_time_axes.append(t)

            # Get only relevant subset of columns
            if columns_list == 'all':
                pass
            else:
                df = df[columns_list]

            all_dfs.append(df)

    return all_dfs, all_time_axes


# This way of doing normalization is fine for long data sets and (relatively) short sequence lengths
# The points from the edges of the datasets count too little
def calculate_normalization_info(df, path_save, rnn_full_name):
    if type(df) is list:
        df_total = pd.concat(df)
    else:
        df_total = df

    if 'time' in df_total.columns:
        df_total.drop('time',
                      axis='columns', inplace=True)

    df_mean = df_total.mean(axis=0)
    df_std = df_total.std(axis=0)
    df_max = df_total.max(axis=0)
    df_min = df_total.min(axis=0)
    frame = {'mean': df_mean, 'std': df_std, 'max': df_max, 'min': df_min}
    df_norm_info = pd.DataFrame(frame).transpose()

    df_norm_info.to_csv(path_save + rnn_full_name + '-norm' + '.csv')

    # Plot historgrams to make the firs check about gaussian assumption
    # for feature in df_total.columns:
    #     plt.hist(df_total[feature].to_numpy(), 50, density=True, facecolor='g', alpha=0.75)
    #     plt.title(feature)
    #     plt.show()

    return df_norm_info


def load_normalization_info(path_save, rnn_full_name):
    return pd.read_csv(path_save + rnn_full_name + '-norm' + '.csv', index_col=0)


def normalize_feature(feature, normalization_info, normalization_type='minmax_sym', name=None):
    """feature needs to have atribute name!!!"""

    if hasattr(feature, 'name'):
        name = feature.name
    else:
        pass

    if name in normalization_info.columns:
        pass
    else:
        return feature

    if normalization_type == 'gaussian':
        col_mean = normalization_info.loc['mean', name]
        col_std = normalization_info.loc['std', name]
        if col_std == 0:
            return 0
        else:
            return (feature - col_mean) / col_std
    elif normalization_type == 'minmax_pos':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        if (col_max - col_min) == 0:
            return 0
        else:
            return (feature - col_min) / (col_max - col_min)
    elif normalization_type == 'minmax_sym':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        if (col_max - col_min) == 0:
            return 0
        else:
            return -1.0 + 2.0 * (feature - col_min) / (col_max - col_min)


def normalize_df(dfs, normalization_info, normalization_type='minmax_sym'):

    if type(dfs) is list:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].apply(normalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)
    else:
        dfs = dfs.apply(normalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)

    return dfs

def denormalize_feature(feature, normalization_info, normalization_type='minmax_sym', name=None):
    """feature needs to have atribute name!!!"""

    if hasattr(feature, 'name'):
        name = feature.name
    else:
        pass

    if normalization_type == 'gaussian':
        col_mean = normalization_info.loc['mean', name]
        col_std = normalization_info.loc['std', name]
        return feature * col_std + col_mean
    elif normalization_type == 'minmax_pos':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        return feature * (col_max - col_min) + col_min
    elif normalization_type == 'minmax_sym':
        col_min = normalization_info.loc['min', name]
        col_max = normalization_info.loc['max', name]
        return ((feature + 1.0) / 2.0) * (col_max - col_min) + col_min

def denormalize_df(dfs, normalization_info, normalization_type='minmax_sym'):

    if type(dfs) is list:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].apply(denormalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)
    else:
        dfs = dfs.apply(denormalize_feature, axis=0,
                                  normalization_info=normalization_info,
                                  normalization_type=normalization_type)

    return dfs