import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def visualize_pca(df):
    # Visualization
    time = df['time']
    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    scatter1 = axs.scatter(df['PCA1'], df['PCA2'], c=time, cmap='coolwarm', alpha=0.7)
    axs.set_title('PCA1 vs PCA2')
    axs.set_xlabel('PCA1')
    axs.set_ylabel('PCA2')
    fig.colorbar(scatter1, ax=axs, label='time [s]')
    plt.tight_layout()
    plt.show()


def visualize_pca_with_feature(df, feature_to_visualize, step=1):
    """
    Creates an animated visualization of PCA components over time alongside the changes in a specified feature.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'time', 'PCA1', 'PCA2', and the specified feature column.
    feature_to_visualize (str): The name of the feature column to visualize over time.
    step (int, optional): Interval for frame sampling to speed up the animation. Defaults to 1 (every row).
    """
    # Validate 'step' parameter
    if step < 1:
        raise ValueError("Parameter 'step' must be a positive integer greater than or equal to 1.")

    # Check if required columns exist in the DataFrame
    required_columns = {'time', 'PCA1', 'PCA2', feature_to_visualize}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"The DataFrame is missing the following required columns: {missing}")

    # Ensure the dataframe is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Sample the DataFrame based on 'step' to speed up the animation
    if step > 1:
        df = df.iloc[::step].reset_index(drop=True)

    time = df['time'].values
    pca1 = df['PCA1'].values
    pca2 = df['PCA2'].values
    feature = df[feature_to_visualize].values

    # Normalize time for colormap
    norm = plt.Normalize(time.min(), time.max())
    cmap = plt.cm.coolwarm

    # Setup the figure and axes
    fig, (ax_pca, ax_feature) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Initialize PCA scatter plot
    scatter = ax_pca.scatter([], [], c=[], cmap=cmap, alpha=0.7)
    ax_pca.set_title('Two highest variance PCA components of GRU memory cells\nStabilization with changing vertical angle')
    ax_pca.set_xlabel('PCA1')
    ax_pca.set_ylabel('PCA2')
    ax_pca.grid(True)

    # Set limits for PCA plots
    padding_pca1 = 0.1 * (pca1.max() - pca1.min()) if pca1.max() != pca1.min() else 1
    padding_pca2 = 0.1 * (pca2.max() - pca2.min()) if pca2.max() != pca2.min() else 1
    ax_pca.set_xlim(pca1.min() - padding_pca1, pca1.max() + padding_pca1)
    ax_pca.set_ylim(pca2.min() - padding_pca2, pca2.max() + padding_pca2)

    # Add vertical colorbar as a time reference
    cbar = fig.colorbar(scatter, ax=ax_pca, orientation='vertical', pad=0.02)
    cbar.set_label('Time [s]')

    # Ensure colorbar reflects actual time range
    try:
        cbar.set_clim(time.min(), time.max())
    except AttributeError:
        scatter.set_norm(norm)

    # Initialize feature plot
    line, = ax_feature.plot([], [], color='blue')
    # ax_feature.set_title(f'{feature_to_visualize} Over Time')
    ax_feature.set_xlabel('Time [s]')
    if feature_to_visualize == 'angle_offset':
        ax_feature.set_ylabel('Angle offset [deg]')
    else:
        ax_feature.set_ylabel(feature_to_visualize)
    ax_feature.grid(True)

    # Set limits for feature plot
    ax_feature.set_xlim(time.min(), time.max())
    if feature.min() != 0:
        feature_min = feature.min() - 0.1 * abs(feature.min())
    else:
        feature_min = feature.min() - 1
    if feature.max() != 0:
        feature_max = feature.max() + 0.1 * abs(feature.max())
    else:
        feature_max = feature.max() + 1
    ax_feature.set_ylim(feature_min, feature_max)

    # Add a progress bar overlay on the colorbar
    cbar_ax = cbar.ax  # The Axes object of the colorbar

    # Initialize the progress bar as fully filled
    progress_bar = Rectangle((0, 0), 1, 1, transform=cbar_ax.transAxes,
                             color='white')
    cbar_ax.add_patch(progress_bar)

    # Number of frames
    num_frames = len(df)

    def init():
        """Initialize the animation."""
        scatter.set_offsets(np.empty((0, 2)))  # Initialize as a 2D empty array
        scatter.set_array(np.array([]))        # Initialize with an empty color array
        line.set_data([], [])
        progress_bar.set_height(1)             # Start with full height
        progress_bar.set_y(0)                  # Start at the bottom
        return scatter, line, progress_bar

    def update(frame):
        """Update the animation by frame."""
        # Update PCA scatter plot
        current_pca1 = pca1[:frame + 1]
        current_pca2 = pca2[:frame + 1]
        current_time = time[:frame + 1]
        scatter.set_offsets(np.c_[current_pca1, current_pca2])
        scatter.set_array(current_time)  # Associate colors with actual time values

        # Update feature plot
        current_feature = feature[:frame + 1]
        current_time_feature = time[:frame + 1]
        line.set_data(current_time_feature, current_feature)

        # Calculate progress (from 0 to 1)
        progress = (frame + 1) / num_frames

        # Update progress bar:
        # - Height decreases from 1 to 0
        # - Y-position increases from 0 to 1
        progress_bar.set_height(1 - progress)
        progress_bar.set_y(progress)

        return scatter, line, progress_bar

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=num_frames, interval=10, blit=True, repeat=False
    )

    plt.tight_layout()
    plt.show()

    # Optional: To save the animation, uncomment the following line
    anim.save('pca_with_feature_animation.mp4', writer='ffmpeg', fps=20)


