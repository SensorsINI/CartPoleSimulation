import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from SI_Toolkit.data_preprocessing import transform_dataset

from Experiment_Recordings.visualization_pca import visualize_pca, visualize_pca_with_feature


save_files_to = None

# get_files_from = '../PCA/'
# save_pca_path = '../PCA/pca_model.joblib'
# load_pca_path = None

get_files_from = './'
save_pca_path = None
load_pca_path = '../PCA/pca_model.joblib'
# load_pca_path = None

feature_to_visualize = 'angle_offset'


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
    df['angle_offset'] = np.rad2deg(np.arctan2(df['angle_offset_sin'], df['angle_offset_cos']))

    # Filter columns matching GRU_H
    gru_h_columns = df.filter(regex=r'^GRU_H')

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

    # visualize_pca(df)
    visualize_pca_with_feature(df, feature_to_visualize, step=2)

    return df

transform_dataset(get_files_from, save_files_to,
                  transformation=PCA_on_hidden_states,
                  save_pca_path=save_pca_path,
                  load_pca_path=load_pca_path,
                  )
