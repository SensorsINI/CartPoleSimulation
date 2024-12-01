import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from SI_Toolkit.data_preprocessing import transform_dataset

from visualization_pca import visualize_pca, visualize_pca_with_feature
from SI_Toolkit.load_and_normalize import load_data, get_paths_to_datafiles

save_files_to = None

# get_files_from = './Experiment/'
# save_pca_path = './PrecomputedPCA/pca_model.joblib'
# load_pca_path = None

get_files_from = './Experiment/'
save_pca_path = None
# load_pca_path = './PrecomputedPCA/pca_model.joblib'
load_pca_path = None

feature_to_visualize = 'L'
additional_feature = 'L'
# additional_feature = None
data_features_to_filter_out = './AntiReference/uninformed_pca_antiref.csv'
# data_features_to_filter_out = None

def PCA_on_hidden_states(df, save_pca_path=None, load_pca_path=None):
    """
    Perform PCA on GRU hidden states, with options to save or load both a scaler and PCA model.

    Args:
        df (pd.DataFrame): Input DataFrame containing GRU hidden states and other features.
        save_pca_path (str, optional): Path to save the trained scaler and PCA model.
        load_pca_path (str, optional): Path to load the pre-trained scaler and PCA model.

    Returns:
        None
    """
    # Exclude the first 600 rows and explicitly create a copy
    df = df.iloc[600:].copy()
    # df['angle_offset'] = np.rad2deg(np.arctan2(df['angle_offset_sin'], df['angle_offset_cos']))

    # Filter columns matching GRU_H
    columns_to_filter = r'^GRU_H'
    gru_h_columns = df.filter(regex=columns_to_filter)

    if data_features_to_filter_out:
        list_of_paths_to_datafiles = get_paths_to_datafiles([data_features_to_filter_out])
        data_with_features_to_filter_out = load_data(list_of_paths_to_datafiles=list_of_paths_to_datafiles, verbose=False)
        data_with_features_to_filter_out = pd.concat(data_with_features_to_filter_out, ignore_index=True)
        gru_h_columns_antireference = data_with_features_to_filter_out.filter(regex=columns_to_filter)
        gru_h_columns = get_cleaned_data(gru_h_columns_antireference, gru_h_columns)

    # Load or initialize scaler
    if load_pca_path:
        # Load both the scaler and PCA model
        scaler, pca = joblib.load(load_pca_path)
        print(f"Scaler and PCA model loaded from {load_pca_path}")
    else:
        # Create and fit a new scaler
        scaler = StandardScaler()
        gru_h_scaled = scaler.fit_transform(gru_h_columns)

        # Train a new PCA model
        pca = PCA(n_components=2)
        pca.fit(gru_h_scaled)

        # Save both the scaler and PCA model, if required
        if save_pca_path:
            joblib.dump((scaler, pca), save_pca_path)
            print(f"Scaler and PCA model saved to {save_pca_path}")

    # Transform data using the loaded or newly trained scaler and PCA
    gru_h_scaled = scaler.transform(gru_h_columns)
    pca_components = pca.transform(gru_h_scaled)

    # Add PCA components back to the DataFrame
    df['PCA1'] = pca_components[:, 0]
    df['PCA2'] = pca_components[:, 1]

    df.to_csv('PCA_components.csv')

    # visualize_pca(df)
    visualize_pca_with_feature(df, feature_to_visualize, step=10, additional_feature=additional_feature)

    return df


def get_cleaned_data(data_with_features_to_filter_out, target_data_to_clean):

    # Step 1: Fit PCA on dynamic-only data
    pca_dynamic = PCA()
    pca_dynamic.fit(data_with_features_to_filter_out)

    # Step 2: Determine the number of components to retain (k) using variance threshold
    cumulative_variance = np.cumsum(pca_dynamic.explained_variance_ratio_)
    k = np.searchsorted(cumulative_variance, 0.99999972) + 1  # Retain components explaining 95% variance
    k = 125
    # 0.99995
    print(f"Number of components retained: {k}")

    # Step 3: Project combined data onto the retained motion components
    pca_dynamic_k = PCA(n_components=k)
    pca_dynamic_k.fit(data_with_features_to_filter_out)

    # Project combined data into motion PCA space
    motion_projection = pca_dynamic_k.transform(target_data_to_clean)

    # Reconstruct motion-related dynamics
    reconstructed_motion = pca_dynamic_k.inverse_transform(motion_projection)

    # Step 4: Subtract motion components to isolate residuals
    residual_data = target_data_to_clean - reconstructed_motion

    return residual_data

transform_dataset(get_files_from, save_files_to,
                  transformation=PCA_on_hidden_states,
                  save_pca_path=save_pca_path,
                  load_pca_path=load_pca_path,
                  )
