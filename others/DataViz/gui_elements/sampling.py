# gui_elements/sampling.py

import tkinter as tk
from tkinter import filedialog, messagebox


class SamplingFrame(tk.LabelFrame):
    """Frame that has buttons for various sampling strategies (High Error, Low Density, Uniform)."""
    def __init__(self, parent, main_app, sampler, **kwargs):
        super().__init__(parent, text="Sampling", **kwargs)
        self.main_app = main_app

        btn_high_err = tk.Button(self, text="High Error", command=self.sample_high_error)
        btn_high_err.pack(side=tk.LEFT, padx=2, pady=2)

        btn_low_density = tk.Button(self, text="Low Density", command=self.sample_poorly_represented)
        btn_low_density.pack(side=tk.LEFT, padx=2, pady=2)

        btn_uniform = tk.Button(self, text="Uniform", command=self.sample_uniform)
        btn_uniform.pack(side=tk.LEFT, padx=2, pady=2)

        self.sampler = sampler

    def sample_high_error(self):
        sampler = self.sampler(self.main_app.config)
        sampled_df = sampler.sample_high_error(self.main_app.df, top_n=50)
        if sampled_df.empty:
            messagebox.showinfo("No Samples", "No error columns found.")
            return
        self._save_sample_to_csv(sampled_df, "high_error")

    def sample_poorly_represented(self):
        sampler = self.sampler(self.main_app.config)
        sampled_df = sampler.sample_poorly_represented_regions(self.main_app.df, n=50)
        if sampled_df.empty:
            messagebox.showinfo("No Samples", "Dataframe empty.")
            return
        self._save_sample_to_csv(sampled_df, "low_density")

    def sample_uniform(self):
        sampler = self.sampler(self.main_app.config)
        sampled_df = sampler.sample_uniform(self.main_app.df, n=50)
        if sampled_df.empty:
            messagebox.showinfo("No Samples", "Dataframe empty.")
            return
        self._save_sample_to_csv(sampled_df, "uniform")

    def _save_sample_to_csv(self, sampled_df, sample_name="sample"):
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"{sample_name}_sample.csv",
            title="Save sample"
        )
        if filename:
            sampled_df.to_csv(filename, index=False)
            messagebox.showinfo("Saved", f"Sample saved to {filename}")
