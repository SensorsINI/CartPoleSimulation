import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Change backend for matplotlib to plot interactively in pycharm
# This import must go before pyplot
from matplotlib import use
# use('TkAgg')
use('macOSX')

# Wraps the angle into range [-π, π]
def wrap_angle_rad_inplace(angle: np.ndarray) -> None:
    Modulo = np.fmod(angle, 2 * np.pi)  # positive modulo
    neg_wrap, pos_wrap = Modulo < -np.pi, Modulo > np.pi
    angle[neg_wrap] = Modulo[neg_wrap] + 2 * np.pi
    angle[pos_wrap] = Modulo[pos_wrap] - 2 * np.pi
    angle[~(neg_wrap | pos_wrap)] = Modulo[~(neg_wrap | pos_wrap)]

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# file_path = './others/physical_cart_acceleration/cartpole-2021-03-29-14-10-20-step-responses-with-friction-slowdown-handcorrected.csv'
file_path = './others/physical_cart_acceleration/cartpole-2021-03-24-09-29-52-step-responses-motor.csv'
converted_dataset_name = 'cart_acceleration.csv'



data: pd.DataFrame = pd.read_csv(file_path, comment='#')


converted_dataset = pd.DataFrame(columns=['time',
                                          'angle',
                                          'angleD',
                                          'angle_cos',
                                          'angle_sin',
                                          'position',
                                          'positionD',
                                          'Q',
                                          'target_position'])

# Save time
converted_dataset['time'] = data['time']

angle0 = 1400.0 # The position with angle pointing dawnwords
angle = (data['angle']-angle0) # make zero downwords
angle = angle * (2*np.pi/4096.0) # Convert to radians
# angle = angle + np.pi # make the 0 up
wrap_angle_rad_inplace(angle)
angle = angle  # +/- Switch convention if necessary to mach simulation

converted_dataset['angle'] = angle
converted_dataset['angle_cos'] = np.cos(angle)
converted_dataset['angle_sin'] = np.sin(angle)

# This is not working well due to wrap of angle...
converted_dataset['angleD'] = np.gradient(converted_dataset['angle'], converted_dataset['time'])
# We use this instead:
angles = converted_dataset['angle'].to_numpy()
converted_dataset['angle'] = angles
DiffAngles = np.zeros_like(angles)
DiffAngles[0] = np.nan
for i in range(1, len(angles)):
    DiffAngle = angles[i]-angles[i-1]
    # This if-statement works independent if positive is clockwise or anticlockwise
    if DiffAngle>np.pi:
        DiffAngle = DiffAngle-2.0*np.pi
    elif DiffAngle<-np.pi:
        DiffAngle = 2.0 * np.pi-DiffAngle
    else:
        pass
    DiffAngles[i] = DiffAngle

converted_dataset['angleD'] = \
    DiffAngles/(converted_dataset['time']-converted_dataset['time'].shift(1))
dt_filter = 5.0*15
dt_sampling = 5.0
# converted_dataset['angleD'] = smooth(smooth(converted_dataset['angleD'], 10),10)
converted_dataset['angleD'] = smooth(converted_dataset['angleD'], 20)
# Extrapolate
converted_dataset['angleD'] = converted_dataset['angleD'] + \
                              (converted_dataset['angleD']-converted_dataset['angleD'].shift(-1))*(dt_filter/dt_sampling)
# Smooth again
converted_dataset['angleD'] = smooth(converted_dataset['angleD'], 10)

converted_dataset['position'] = data['position']*(0.198/2330.0)

# We use the direct calculation which is less precise however realistic for realt time scenario
# converted_dataset['positionD'] = np.gradient(converted_dataset['position'], converted_dataset['time'])
converted_dataset['positionD'] = \
    (converted_dataset['position']-converted_dataset['position'].shift(1))/(converted_dataset['time']-converted_dataset['time'].shift(1))


converted_dataset['Q'] = data['actualMotorCmd']/8192.0
converted_dataset['target_position'] = data['positionTarget']*(0.198/2330.0)



converted_dataset = converted_dataset.iloc[16:-16,:]

converted_dataset.to_csv(converted_dataset_name, index=False)


fontsize_labels = 10
fontsize_ticks = 8

fig, axs = plt.subplots(6, 1, figsize=(16, 9), sharex=True)


axs[0].set_ylabel("Position (m)", fontsize=fontsize_labels)
axs[0].plot(converted_dataset['time'], converted_dataset['position'],
            'b', markersize=12, label='Position')
axs[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

axs[1].set_ylabel("Speed (m/s)", fontsize=fontsize_labels)
axs[1].plot(converted_dataset['time'], converted_dataset['positionD'],
            'g', markersize=12, label='Velocity')
axs[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)


axs[2].set_ylabel("Angle (rad)", fontsize=fontsize_labels)
axs[2].plot(converted_dataset['time'], converted_dataset['angle'],
            'b', markersize=12, label='Angle')
axs[2].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

axs[3].set_ylabel("Angular speed (rad/s)", fontsize=fontsize_labels)
axs[3].plot(converted_dataset['time'], converted_dataset['angleD'],
            'g', markersize=12, label='Angular speed')
axs[3].tick_params(axis='both', which='major', labelsize=fontsize_ticks)


axs[4].set_ylabel("Motor Input Q (-)", fontsize=fontsize_labels)
axs[4].plot(converted_dataset['time'], converted_dataset['Q'],
            'r', markersize=12, label='Q')
axs[4].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

axs[5].set_ylabel("Target position (m)", fontsize=fontsize_labels)
axs[5].plot(converted_dataset['time'], converted_dataset['target_position'],
            'b', markersize=12, label='Target position')
axs[5].tick_params(axis='both', which='major', labelsize=fontsize_ticks)


axs[5].set_xlabel('Time (s)', fontsize=fontsize_labels)

fig.align_ylabels()

plt.tight_layout()

plt.show()


