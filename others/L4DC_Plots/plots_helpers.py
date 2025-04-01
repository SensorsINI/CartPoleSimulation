import numpy as np


def label_target_position_and_position(ax, position_color, target_position_color, fontsize):
    ax.text(-0.070, 0.5, 'Target Position', color=target_position_color, fontsize=fontsize,
             transform=ax.transAxes, rotation=90, va='center', ha='right')
    # Add blue "Position (cm)" text
    ax.text(-0.065, 0.075, '&', color='black', fontsize=fontsize,
             transform=ax.transAxes, rotation=90, va='center', ha='left')
    ax.text(-0.065, 0.41, 'Position', color=position_color, fontsize=fontsize,
             transform=ax.transAxes, rotation=90, va='center', ha='left')
    ax.text(-0.065, 0.85, '(cm)', color='black', fontsize=fontsize,
             transform=ax.transAxes, rotation=90, va='center', ha='left')


def break_line_on_jump(x, y, threshold=90.0, z=None):

    # Containers for the modified data
    x_modified = []
    y_modified = []
    z_modified = []

    # Loop through the data and insert np.nan where the difference exceeds the threshold
    for i in range(1, len(y)):
        x_modified.append(x[i - 1])
        y_modified.append(y[i - 1])
        if z is not None:
            z_modified.append(z[i - 1])

        if np.abs(y[i] - y[i - 1]) > threshold:
            # Insert np.nan to break the line
            x_modified.append(np.nan)
            y_modified.append(np.nan)
            if z is not None:
                z_modified.append(np.nan)

    # Don't forget to add the last point
    x_modified.append(x[-1])
    y_modified.append(y[-1])
    if z is not None:
        z_modified.append(z[-1])

    # Convert to numpy arrays
    x_modified = np.array(x_modified)
    y_modified = np.array(y_modified)
    z_modified = np.array(z_modified)

    if z is not None:
        return x_modified, y_modified, z_modified
    else:
        return x_modified, y_modified
