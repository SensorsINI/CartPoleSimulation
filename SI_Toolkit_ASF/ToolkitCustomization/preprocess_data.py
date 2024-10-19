from CartPole import Generate_Random_Trace_Function
import numpy as np
import pandas as pd


def add_new_target_position(df, target_position_config, new_target_position_variable_name, **kwargs):

    time = (df['time'] - df['time'].iloc[0]).to_numpy()
    length_of_experiment = time[-1]
    target_position_generating_function = Generate_Random_Trace_Function(
        length_of_experiment=length_of_experiment,
        **target_position_config
    )

    df[new_target_position_variable_name] = target_position_generating_function(time)

    return df


def flip_target_equilibrium(df, new_target_equilibrium_variable_name, **kwargs):

    df[new_target_equilibrium_variable_name] = -df['target_equilibrium']

    return df


def add_fake_trajectories(df, fake_trajectory_every_min, fake_trajectory_every_max, fake_trajectory_min_len, fake_trajectory_max_len):
    # Copy of the original DataFrame to avoid modifying it directly
    df_copy = df.copy()

    df_copy['fake_trajectory'] = 0

    # Precompute the normalized angle from target_position
    fake_angle = df_copy['target_position'].values
    fake_angle_max = np.max(fake_angle)
    fake_angle_min = np.min(fake_angle)
    fake_angle = (fake_angle - fake_angle_min) / (fake_angle_max - fake_angle_min)  # Normalize into [0, 1]
    fake_angle = fake_angle * 2 * np.pi - np.pi  # Normalize into [-pi, pi]

    # Identify columns that should remain unchanged
    columns_to_preserve = df.columns.difference(['angle', 'angle_cos', 'angle_sin', 'angleD', 'position', 'positionD'])

    i = 0
    fake_trajectory_every = np.random.randint(fake_trajectory_every_min, fake_trajectory_every_max + 1)
    last_fake_trajectory_idx = 0
    while i < len(df_copy):

        if i != 0 and (i-last_fake_trajectory_idx) % fake_trajectory_every == 0:
            last_fake_trajectory_idx = i
            trajectory_len = np.random.randint(fake_trajectory_min_len, fake_trajectory_max_len + 1)

            if i + trajectory_len >= len(df_copy):
                break

            # Randomly pick separate snippets for angleD, position, and positionD
            angle_idx = np.random.randint(0, len(df_copy) - trajectory_len)
            angleD_idx = np.random.randint(0, len(df_copy) - trajectory_len)
            position_idx = np.random.randint(0, len(df_copy) - trajectory_len)
            positionD_idx = np.random.randint(0, len(df_copy) - trajectory_len)

            angleD_snippet = df_copy.iloc[angleD_idx:angleD_idx + trajectory_len]['angleD'].values
            position_snippet = df_copy.iloc[position_idx:position_idx + trajectory_len]['position'].values
            positionD_snippet = df_copy.iloc[positionD_idx:positionD_idx + trajectory_len]['positionD'].values

            # Pick a random snippet from 'target_position' to normalize and use as 'angle'
            angle_snippet = fake_angle[angle_idx:angle_idx + trajectory_len]

            # Calculate the cos and sin of the normalized angle
            angle_cos = np.cos(angle_snippet)
            angle_sin = np.sin(angle_snippet)

            # Prepare the data for the fake trajectory
            fake_trajectory = pd.DataFrame({
                'angle': angle_snippet,
                'angle_cos': angle_cos,
                'angle_sin': angle_sin,
                'angleD': angleD_snippet,
                'position': position_snippet,
                'positionD': positionD_snippet,
                'fake_trajectory': 1  # Mark these rows as part of a fake trajectory
            })

            # Add the preserved columns to the fake trajectory
            for col in columns_to_preserve:
                fake_trajectory[col] = df_copy.iloc[i:i + trajectory_len][col].values

            # Insert the fake trajectory into the DataFrame at the current index
            df_copy = pd.concat([df_copy.iloc[:i], fake_trajectory, df_copy.iloc[i + trajectory_len:]]).reset_index(drop=True)

            i += trajectory_len
        else:
            i += 1

    return df_copy
