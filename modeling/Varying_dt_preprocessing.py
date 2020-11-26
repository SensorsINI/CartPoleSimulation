import pandas as pd
import numpy as np

INPUT_FILES = ['data_rnn_big-1.csv']
path_save = './data/'
max_downsampling = 10
new_name = 'dt_variable_test-'

for one_filepath in INPUT_FILES:
    df_dense = pd.read_csv(path_save+one_filepath, comment='#')

    # Add time column
    if 'time' in df_dense.columns and 'dt' not in df_dense.columns:
        dt = [0.0]
        row_iterator = df_dense.iterrows()
        _, last = row_iterator.next()  # take first item from row_iterator
        for i, row in row_iterator:
            dt.append(row['value'] - last['value'])
            last = row
        df_dense['dt'] = np.array(dt)
    elif 'dt' in df_dense.columns and 'time' not in df_dense.columns:
        dt = df_dense['dt']
        t = dt.cumsum()
        df_dense['time'] = t
    elif 'dt' in df_dense and 'time' in df_dense:
        pass
    else:
        raise('No time information in dataset')

    # Prepare the indexes
    for idx in range(max_downsampling):
        indexes = [idx]
        while True:
            new_index = indexes[-1]+np.random.randint(1, max_downsampling+1)
            if new_index >= len(df_dense.index):
                break
            else:
                indexes.append(new_index)
        df = df_dense.iloc[indexes]

        dt = [0.0]
        row_iterator = df.iterrows()
        _, last = next(row_iterator)  # take first item from row_iterator
        for i, row in row_iterator:
            dt.append(np.around(row['time'] - last['time'], 5))
            last = row
        df['dt'] = np.array(dt)


        df.to_csv(path_save+ new_name + str(idx) + '.csv', index=False)





