import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
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



def visualize_pca_with_feature(df, feature_to_visualize, step=1, additional_feature=None):
    """
    Creates an animated visualization of PCA components over time alongside the changes
    in a specified feature and an optional additional feature.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'time', 'PCA1', 'PCA2', the specified feature column,
                       and optionally the additional feature column.
    feature_to_visualize (str): The name of the primary feature column to visualize over time.
    step (int, optional): Interval for frame sampling to speed up the animation. Defaults to 1 (every row).
    additional_feature (str, optional): The name of an additional feature column to visualize over time.
    """
    # Validate 'step' parameter
    if step < 1:
        raise ValueError("Parameter 'step' must be a positive integer greater than or equal to 1.")

    # Define required columns
    required_columns = {'time', 'PCA1', 'PCA2', feature_to_visualize}
    if additional_feature:
        required_columns.add(additional_feature)

    # Check if required columns exist in the DataFrame
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"The DataFrame is missing the following required columns: {missing_columns}")

    # Prepare the DataFrame
    df = df.sort_values('time').reset_index(drop=True)
    if step > 1:
        df = df.iloc[::step].reset_index(drop=True)

    # Extract data
    time = df['time'].values
    pca1 = df['PCA1'].values
    pca2 = df['PCA2'].values
    feature = df[feature_to_visualize].values
    additional = df[additional_feature].values if additional_feature else None

    # Determine color mapping
    color_values = additional if additional_feature else time
    cmap = plt.get_cmap('viridis' if additional_feature else 'coolwarm')
    norm = Normalize(vmin=color_values.min(), vmax=color_values.max())

    # Setup the figure and axes
    num_subplots = 3 if additional_feature else 2
    height_ratios = [3, 0.5, 0.5] if additional_feature else [3, 1]
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 10),
                             gridspec_kw={'height_ratios': height_ratios})

    ax_pca = axes[0]
    ax_feature = axes[1]
    ax_additional = axes[2] if additional_feature else None

    # Initialize PCA scatter plot
    scatter = ax_pca.scatter([], [], c=[], cmap=cmap, alpha=0.7, norm=norm)
    ax_pca.set_title('Two highest variance PCA Components of GRU memory units \nStabilization at 3 target positions with changing vertical angle')
    ax_pca.set_xlabel('PCA1')
    ax_pca.set_ylabel('PCA2')
    ax_pca.grid(True)

    # Set PCA plot limits
    def set_plot_limits(ax, x_data, y_data, padding=0.1):
        x_range = x_data.max() - x_data.min() or 1
        y_range = y_data.max() - y_data.min() or 1
        ax.set_xlim(x_data.min() - padding * x_range, x_data.max() + padding * x_range)
        ax.set_ylim(y_data.min() - padding * y_range, y_data.max() + padding * y_range)

    set_plot_limits(ax_pca, pca1, pca2)

    # Create colorbar
    divider = make_axes_locatable(ax_pca)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    # Keep the custom image generation as specified
    if additional_feature:
        cbar_image = ((color_values - color_values.min()) / (color_values.max() - color_values.min())).reshape(-1, 1)
    else:
        cbar_image = np.linspace(0, 1, 256).reshape(-1, 1)

    cax.imshow(cbar_image, aspect='auto', extent=[0, 1, time.min(), time.max()],
               cmap=cmap, origin='lower')
    cax.set_ylabel('Time [s]')
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    cax.xaxis.set_visible(False)

    # Add progress bar overlay
    progress_bar = Rectangle((0, 0), 1, 0, transform=cax.transAxes,
                             color='white', zorder=2)
    cax.add_patch(progress_bar)

    # Initialize primary feature plot
    line_feature, = ax_feature.plot([], [], color='blue')
    ax_feature.set_xlabel('Time [s]')
    ax_feature.set_ylabel(get_label(feature_to_visualize))
    ax_feature.grid(True)
    set_plot_limits(ax_feature, time, feature, padding=0.1)

    # Initialize additional feature plot if provided
    if additional_feature:
        line_additional = LineCollection([], cmap=cmap, norm=norm, linewidths=4)
        ax_additional.add_collection(line_additional)
        # line_additional, = ax_additional.plot([], [], color='green')
        ax_additional.set_xlabel('Time [s]')
        ax_additional.set_ylabel(get_label(additional_feature))
        ax_additional.grid(True)
        set_plot_limits(ax_additional, time, additional, padding=0.1)

    num_frames = len(df)

    def init():
        """Initialize the animation."""
        # Ensure the scatter plot is initialized with no points
        scatter.set_offsets(np.empty((0, 2)))  # Correctly initialize as a 2D empty array
        scatter.set_array([])  # No color data initially

        # Clear the primary feature line
        line_feature.set_data([], [])

        # Initialize additional feature plot if applicable
        if additional_feature:
            line_additional.set_segments([])  # No line segments at the start
            line_additional.set_array([])  # No color data initially

        # Initialize the progress bar
        progress_bar.set_height(1)  # Full height
        progress_bar.set_y(0)  # Positioned at the bottom

        # Return all artists for blitting
        artists = [scatter, line_feature, progress_bar]
        if additional_feature:
            artists.append(line_additional)
        return artists

    def update(frame):
        """Update the animation by frame."""
        idx = frame + 1
        # Update scatter
        scatter.set_offsets(np.c_[pca1[:idx], pca2[:idx]])
        scatter.set_array(color_values[:idx])

        # Update primary feature plot
        line_feature.set_data(time[:idx], feature[:idx])

        # Update additional feature plot
        if additional_feature:
            # current_additional = df[additional_feature].values[:frame + 1]
            # line_additional.set_data(current_time_feature, current_additional)
            segments = [((time[i], additional[i]), (time[i+1], additional[i+1])) for i in range(idx-1)]
            line_additional.set_segments(segments)
            line_additional.set_array(color_values[:idx-1])

        # Update progress bar
        progress = idx / num_frames
        progress_bar.set_height(1 - progress)
        progress_bar.set_y(progress)

        artists = [scatter, line_feature, progress_bar]
        if additional_feature:
            artists.append(line_additional)
        return artists

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=num_frames, interval=10, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

    # Optional: To save the animation, uncomment the following line
    # anim.save('pca_with_feature_animation.mp4', writer='ffmpeg', fps=20)


def get_label(feature_name):
    if feature_name == 'position':
        return 'Cart Position [m]'
    elif feature_name == 'angle':
        return 'Pole Angle [rad]'
    elif feature_name == 'target_position':
        return 'Target Position [m]'
    elif feature_name == 'angle_offset':
        return 'Pole Angle Offset [deg]'
    elif feature_name == 'm_pole':
        return 'Pole Mass [kg]'
    elif feature_name == 'L':
        return 'Pole Length [m]'
