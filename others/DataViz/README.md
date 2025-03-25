# CartPoleSimulation Data Visualization & Analysis
#### Generated automatically 25.03.2025

This repository provides a Tkinter-based GUI for loading, analyzing, and visualizing CartPole simulation data stored in CSV files. It supports error computation between teacher (ground-truth) and student (NN-predicted) control signals, basic dimension-reduction (PLS/OLS), sampling, and various plot types (scatter, heatmap, density). 

---

## Table of Contents

1. [Features](#features)  
2. [Architecture Overview](#architecture-overview)  
3. [File-by-File Details](#file-by-file-details)  
4. [Dependencies and Installation](#dependencies-and-installation)  
5. [Usage](#usage)  
6. [Configuration](#configuration)  
7. [How to Extend](#how-to-extend)  
8. [License](#license)  

---

## Features

- **CSV Auto-Loading**: Merges multiple CSV files into one DataFrame, automatically tagging them with `experiment_id`.  
- **State/Control Error Computation**: Computes absolute and relative errors between teacher and student controls.  
- **Dimension Reduction**: Optionally uses Partial Least Squares (PLS) or OLS (linear regression) to reduce data to 2D for plotting.  
- **Sampling**: Quickly sample high-error rows, low-density regions, or random uniform subsets.  
- **GUI-based Filtering**: Filter data by control signals, additional features, or custom ranges.  
- **Multiple Plot Types**: Scatter or heatmap modes, with color-coded errors, controls, or density.  
- **Normalization**: Load global min/max from file or compute locally, scaling data to [-1, 1].  

---

## Architecture Overview

```
CartPoleSimulation/
├── config.py
│    Holds global configuration (folder paths, column names, etc.).
│
├── data_loader.py
│    Defines DataLoader:
│       - Scans the specified folder for CSV files.
│       - Loads them into a single pandas DataFrame with an `experiment_id`.
│
├── data_processor.py
│    Defines DataProcessor:
│       - Computes absolute and relative errors (with optional capping).
│       - Creates clipped versions of student controls.
│
├── sampler.py
│    Defines Sampler:
│       - Offers methods to sample rows based on error magnitude, density, etc.
│
├── main_app.py
│    Defines MainApplication (Tkinter GUI):
│       - Renders the interface (comboboxes, filters, plots).
│       - Handles data normalization, filtering logic, sampling actions.
│       - Integrates with matplotlib for visualization.
│       - Applies PLS or OLS dimension-reduction if requested.
│
└── main.py
     Main entry point:
        - Initializes Config and DataLoader.
        - Computes initial errors via DataProcessor.
        - Launches the GUI in MainApplication.
```

---

## File-by-File Details

1. **`config.py`**  
   - **Class**: `Config`  
   - **Role**: Manages user-defined settings like data paths, column names, and error-capping.  

2. **`data_loader.py`**  
   - **Class**: `DataLoader`  
   - **Role**: Loads all CSV files from a specified folder and concatenates them into one DataFrame with a unique `experiment_id`.  

3. **`data_processor.py`**  
   - **Class**: `DataProcessor`  
   - **Role**: Computes and attaches error columns (absolute and relative) between teacher and student. Supports clipped values.  

4. **`sampler.py`**  
   - **Class**: `Sampler`  
   - **Role**: Provides convenience methods to extract subsets of rows: e.g., high-error rows, low-density, random uniform.  

5. **`main_app.py`**  
   - **Class**: `MainApplication`  
   - **Role**:  
     - Implements the core Tkinter-based GUI.  
     - Manages user-interaction (plot options, filters, dimension-reduction toggles).  
     - Integrates with matplotlib to display scatter/heatmap.  
     - Applies normalization when requested.  
     - Handles data re-loading, sampling, and saving samples.  

6. **`main.py`**  
   - **Function**: `main()`  
   - **Role**:  
     - Instantiates `Config` and `DataLoader`.  
     - Precomputes errors via `DataProcessor`.  
     - Starts the Tkinter application (`MainApplication`).  

---

## Dependencies and Installation

- **Python** 3.7+ (for typing, f-strings, etc.)
- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  
- [tqdm](https://github.com/tqdm/tqdm)  
- [matplotlib](https://matplotlib.org/)  
- [scikit-learn](https://scikit-learn.org/) (PLS, OLS)  
- [tkinter](https://docs.python.org/3/library/tkinter.html) (commonly included with Python)

### Install Dependencies
```
pip install pandas numpy tqdm matplotlib scikit-learn
```
*(If tkinter is not preinstalled, consult your OS documentation.)*

---

## Usage

1. **Adjust Configuration**:  
   - Open `config.py`, set your `data_folder` path, column lists, etc.  
   - (Optional) Provide a `norm_file_path` if you have a global min/max CSV for normalization.  

2. **Run**:
   ```bash
   python main.py
   ```
3. **GUI Interaction**:  
   - **Load**: On startup, it automatically loads CSV files from `config.data_folder`.  
   - **Plot**: Select X/Y columns, or enable PLS/OLS.  
   - **Filter**: Adjust filtering by control signals or by an additional feature.  
   - **Sample**: Export subsets (e.g. high-error) via “Sampling” buttons.  
   - **Normalization**: Check the “Normalize [-1..1]” box to rescale data.  
   - **Relative Error Cap**: Adjust to clamp outlier error values.  

---

## Configuration

- **`data_folder`**: Directory with CSV files.  
- **`state_columns`**: Which columns represent system state features.  
- **`teacher_control_columns`** / **`student_control_columns`**: Columns for teacher/student controls.  
- **`relative_error_cap`**: Clip relative errors above this threshold.  
- **`norm_file_path`**: CSV with rows labeled “min” and “max” for each column (if available). Otherwise, local min/max is used.  

---

## How to Extend

- Add new plots, sampling methods, GUI tweaks, or modular updates by implementing small, focused methods/functions in the appropriate files (`main_app.py`, `sampler.py`, etc.).
- If collaborating with AI assistants like ChatGPT, provide short architecture overviews and relevant code snippets to clearly communicate your intentions and maintain productivity.

---

**Enjoy exploring CartPole simulation data with this interactive GUI!**
