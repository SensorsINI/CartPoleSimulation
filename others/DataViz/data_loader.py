# data_loader.py

import os
import glob
import pandas as pd
from tqdm import tqdm


class DataLoader:
    """Loads CSV files into a single DataFrame."""
    def __init__(self, config) -> None:
        self.config = config

    def load_data(self) -> pd.DataFrame:
        folder_path = self.config.data_folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")

        df_list = []
        for i, file in enumerate(tqdm(csv_files, desc="Loading CSV files")):
            df_temp = pd.read_csv(file, comment="#")
            df_temp["experiment_id"] = i
            df_list.append(df_temp)

        return pd.concat(df_list, ignore_index=True)
