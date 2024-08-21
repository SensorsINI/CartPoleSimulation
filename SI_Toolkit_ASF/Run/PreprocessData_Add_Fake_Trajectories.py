from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = './CartPoleSimulation/SI_Toolkit_ASF/Experiments/Trial_12_raw/'
save_files_to = './CartPoleSimulation/SI_Toolkit_ASF/Experiments/Trial_12_raw_augmented2/'
fake_trajectory_every = 25
fake_trajectory_min_len = 1
fake_trajectory_max_len = 5

transform_dataset(get_files_from, save_files_to, transformation='add_fake_trajectories',
                  fake_trajectory_every=fake_trajectory_every,
                  fake_trajectory_min_len=fake_trajectory_min_len, fake_trajectory_max_len=fake_trajectory_max_len)
