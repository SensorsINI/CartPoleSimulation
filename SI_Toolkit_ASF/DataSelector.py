import numpy as np
import pandas as pd

from tqdm import trange

class DataSelector:
    def __init__(self, args):

        self.args = args

        self.data = None
        self.df_lengths = []
        self.df_lengths_cs = []

        self.indices = None

        self.wash_out_len = args.wash_out_len  # if recurrent you have to make sure you can cut out sequence
        self.post_wash_out_len = args.post_wash_out_len  # 1 is minimum here! Also for dense net
        self.exp_len = self.wash_out_len + self.post_wash_out_len

        self.num = 20  # You get one more bean as num
        self.points_per_bin = 1
        self.nr_states_per_bin = np.ones((self.num,self.num,self.num,self.num))*self.points_per_bin
        self.nr_states_per_bin_current = np.zeros_like(self.nr_states_per_bin)
        self.table_empty_places_init = np.sum(self.nr_states_per_bin)
        self.table_empty_places = self.table_empty_places_init

        self.position_bin_boundries = None
        self.positionD_bin_boundries = None
        self.angle_bin_boundries = None
        self.angleD_bin_boundries = None

        self.max_iter = None

        self.selected_indeces = []
        self.collected_points = 0

    def load_data_into_selector(self, data):

        self.data = data
        maxs = []
        mins = []
        if type(self.data) == list:
            for data_set in self.data:

                # Get lengths
                self.df_lengths.append(data_set.shape[0] - self.exp_len)
                if not self.df_lengths_cs:
                    self.df_lengths_cs.append(self.df_lengths[0])
                else:
                    self.df_lengths_cs.append(self.df_lengths_cs[-1] + self.df_lengths[-1])

                # Get max/min values
                maxs.append(data_set.max())    # Take care! There is intentionally min here!
                mins.append(data_set.min().abs())

            maxs = pd.concat(maxs, axis=1).T.max()
            mins = pd.concat(mins, axis=1).T.max()
            self.maxs = pd.concat([maxs, mins], axis=1).T.min()
            self.number_of_samples = self.df_lengths_cs[-1]
            # Get global max/min values

        else:
            self.number_of_samples = self.data.shape[0] - self.exp_len
            self.df_lengths_cs = [self.number_of_samples]
            self.maxs = pd.concat([self.data.max(), self.data.min().abs()]).min()


        print('There are {} datapoints available'.format(self.number_of_samples))
        print('You requested to collect {} points'.format(self.table_empty_places))

        first = 1.0/(self.num-1)
        last = (self.num-1-1)/(self.num-1)

        self.position_bin_boundries = np.linspace(start=-self.maxs['position']*first, stop=self.maxs['position']*last, num=self.num-1)
        self.positionD_bin_boundries = np.linspace(start=-self.maxs['positionD']*first, stop=self.maxs['positionD']*last, num=self.num-1)
        self.angle_bin_boundries = np.linspace(start=-self.maxs['angle']*first, stop=self.maxs['angle']*last, num=self.num-1)
        self.angleD_bin_boundries = np.linspace(start=-self.maxs['angleD']*first, stop=self.maxs['angleD']*last, num=self.num-1)

        self.max_iter = self.number_of_samples-1

        self.indices = np.arange(self.number_of_samples)
        np.random.shuffle(self.indices)

        progress_bar = trange(self.max_iter)
        for iterator in progress_bar:

            idx = self.indices[iterator]
            idx_data_set = next(i for i, v in enumerate(self.df_lengths_cs) if v > idx)
            if idx_data_set == 0:
                pass
            else:
                idx -= self.df_lengths_cs[idx_data_set - 1]

            idx = idx+self.wash_out_len # This should be a point just after washout, this way if we cut out sequences this point matters for loss
            data_point = self.data[idx_data_set].iloc[idx, :]


            # Get index of this point in the grid
            position_idx = next((i for i, v in enumerate(self.position_bin_boundries) if v > data_point['position']), len(self.position_bin_boundries))
            positionD_idx = next((i for i, v in enumerate(self.positionD_bin_boundries) if v > data_point['positionD']), len(self.positionD_bin_boundries))
            angle_idx = next((i for i, v in enumerate(self.angle_bin_boundries) if v > data_point['positionD']), len(self.angle_bin_boundries))
            angleD_idx = next((i for i, v in enumerate(self.angleD_bin_boundries) if v > data_point['positionD']), len(self.angleD_bin_boundries))

            bin_idx = (position_idx, positionD_idx, angle_idx, angleD_idx)

            # Check if there is already a max number of points there
            if self.nr_states_per_bin_current[bin_idx] <= self.nr_states_per_bin[bin_idx]:
                self.selected_indeces.append([idx_data_set, idx])
                self.nr_states_per_bin_current[bin_idx] += 1
                self.collected_points += 1
                self.table_empty_places -= 1

            if iterator == self.max_iter-1:
                print('Max. iterations reached, still there are {} missing data points'.format(self.table_empty_places))
                print('Proceed with {} data points'.format(self.collected_points))

            if self.table_empty_places == 0:
                print('All data points collected after {} iterations'.format(iterator))
                break

            if iterator%1000 == 0:
                progress_bar.set_postfix(collected_points=self.collected_points)


    def return_dataset_for_training(self,
                                    inputs=None,
                                    outputs=None,
                                    batch_size=None,
                                    shuffle=True,
                                    raw=False
                                    ):

        if batch_size is None and self.args.batch_size is not None:
            batch_size = self.args.batch_size

        if inputs is None and self.args.inputs is not None:
            inputs = self.args.inputs

        if outputs is None and self.args.outputs is not None:
            outputs = self.args.outputs

        input_data = []
        output_data = []
        time_axes = []

        for df in self.data:
            time_axes.append(df['time'])
            input_data.append(df[inputs])
            output_data.append(df[outputs])

        data_x = [input_data[idx_data_set].iloc[idx-self.wash_out_len:idx+self.post_wash_out_len, :].to_numpy() for idx_data_set, idx in self.selected_indeces]
        data_y = [output_data[idx_data_set].iloc[idx-self.wash_out_len+1:idx+self.post_wash_out_len+1, :].to_numpy() for idx_data_set, idx in self.selected_indeces]

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        if raw:
            return data_x, data_y
        else:
            return Dataset_Selector(data_x, data_y, batch_size, shuffle)


from tensorflow import keras

class Dataset_Selector(keras.utils.Sequence):
    def __init__(self,
                 data,
                 labels,
                 batch_size=None,
                 shuffle=True):

        self.data = data
        self.labels = labels

        self.batch_size = 1
        self.number_of_batches = None
        self.shuffle = shuffle
        self.indices = []

        self.number_of_samples = data.shape[0]

        self.reset_batch_size(batch_size)

    def reset_batch_size(self, batch_size=None):

        if batch_size is None:
            raise ValueError("Batch size cannot be None!")
        else:
            self.batch_size = batch_size

        self.number_of_batches = int(np.ceil(self.number_of_samples / float(self.batch_size)))

        self.indices = np.arange(self.number_of_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # In TF it must return the number of batches
        return self.number_of_batches

    def __getitem__(self, idx_batch):

        sample_idx = self.indices[self.batch_size * idx_batch: self.batch_size * (idx_batch + 1)]
        features_batch = self.data[sample_idx, :, :]
        targets_batch = self.labels[sample_idx, :, :]

        return features_batch, targets_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)