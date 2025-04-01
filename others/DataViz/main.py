# main.py

import tkinter as tk
from config import Config
from data_loader import DataLoader
from data_processor import DataProcessor
from gui import MainApplication


def main():
    config = Config()
    loader = DataLoader(config)
    df = loader.load_data()

    processor = DataProcessor(config)
    df = processor.compute_errors(df)

    app = MainApplication(config, df)
    app.mainloop()


if __name__ == "__main__":
    main()
