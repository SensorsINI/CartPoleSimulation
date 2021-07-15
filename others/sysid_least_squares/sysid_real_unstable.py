import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy import signal
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from numba import jit

def read_data(file):
    data = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0] == 'time':
                n_to_c = dict((name, index) for index, name in enumerate(row))
                break
        for row in reader:
            row_data = []
            for x in row[:-1]:
                try:
                    row_data.append(float(x))
                except ValueError as e:
                    if x == 'False':
                        row_data.append(0)
                    elif x == 'True':
                        row_data.append(1)
                    else:
                        raise e
            data.append(row_data)
    y = np.array(data)
    return y, n_to_c

def filt_data(file):
    b, a = signal.butter(1, 0.032)
 
    y[:, n_to_c['position']] = signal.medfilt(y[:, n_to_c['position']], kernel_size=5)
    y[:, n_to_c['position']] = signal.filtfilt(b, a, y[:, n_to_c['position']], padlen=150)

    y[:, n_to_c['positionD']] = np.gradient(y[:, n_to_c['position']]) / np.gradient(y[:, n_to_c['time']])
    #y[:, n_to_c['positionD']] = signal.filtfilt(b, a, y[:, n_to_c['positionD']], padlen=150)

    sin_a = np.sin(y[:, n_to_c['angle']])
    cos_a = np.cos(y[:, n_to_c['angle']])

    sin_a_med = signal.medfilt(sin_a, kernel_size=5)
    cos_a_med = signal.medfilt(cos_a, kernel_size=5)
    sin_a_filt = signal.filtfilt(b, a, sin_a_med, padlen=150)
    cos_a_filt = signal.filtfilt(b, a, cos_a_med, padlen=150)

    angles = np.arctan2(sin_a_filt, cos_a_filt)

    y[:, n_to_c['angle']] = angles

    # Recalculate angle derivative
    da_list = []
    for i, _ in enumerate(angles):
        if i == 0:
            da_list.append(0)
        else:
            da = angle_distance(angles[i], angles[i-1]) / (y[i, n_to_c['time']] - y[i-1, n_to_c['time']])
            da_list.append(da)

    da_array = np.array(da_list)
    y[:, n_to_c['angleD']] = da_array #signal.filtfilt(b, a, da_array, padlen=150)

    y[:, n_to_c['Q']] = signal.medfilt(y[:, n_to_c['Q']], kernel_size=5)
    y[:, n_to_c['Q']] = signal.filtfilt(b, a, y[:, n_to_c['Q']], padlen=150)


# copied from _CartPole_mathematical_helpers.py
@jit(nopython=True)
def wrap_angle_rad_inplace(angle):
    Modulo = np.fmod(angle, 2 * np.pi)  # positive modulo
    neg_wrap, pos_wrap = Modulo < -np.pi, Modulo > np.pi
    angle[neg_wrap] = Modulo[neg_wrap] + 2 * np.pi
    angle[pos_wrap] = Modulo[pos_wrap] - 2 * np.pi
    angle[~(neg_wrap | pos_wrap)] = Modulo[~(neg_wrap | pos_wrap)]

@jit(nopython=True)
def angle_distance(x, y):
    dist = x - y
    
    if np.abs(dist) > np.pi:
        dist = -(np.sign(dist)*2*np.pi - dist)
        if dist > np.pi:
            raise Exception('NO')
    return dist

# copied from cartpole_model.py
@jit(nopython=True)
def f_cartpole(x, u, p):
    #m = 0.087 #p[0]
    #M = 0.230 #p[1]
    #L = 0.395/2.0 #p[2]
    m = p[0]
    M = p[1]
    L = p[2]

    M_fric = p[3]
    J_fric = p[4]
    u_scale = 1.0 #p[2]
    g = 9.81

    ca = np.cos(-x[2]) # This is how it is done in cartpole_model.py
    sa = np.sin(-x[2])

    k = 1.3333333333333333

    positionD = x[1]
    angleD = x[3]

    # Clockwise rotation is defined as negative
    # force and cart movement to the right are defined as positive
    # g (gravitational acceleration) is positive (absolute value)
    # Checked independently by Marcin and Krishna

    A = m * (ca ** 2) - (k + 1) * (M + m)

    positionDD = (
        (
            + m * g * sa * ca  # Movement of the cart due to gravity
            - ((J_fric * (-angleD) * ca) / L)  # Movement of the cart due to pend' s friction in the joint
            - (k + 1) * (
                + (m * L * (angleD ** 2) * sa)  # Keeps the Cart-Pole center of mass fixed when pole rotates
                - M_fric * positionD  # Braking of the cart due its friction
                + u*u_scale  # Effect of force applied to cart
            )
        ) / A
    )

    # Making m go to 0 and setting J_fric=0 (fine for pole without mass)
    # positionDD = (u_max/M)*Q-(M_fric/M)*positionD
    # Compare this with positionDD = a*Q-b*positionD
    # u_max = M*a = 0.230*19.6 = 4.5, 0.317*19.6 = 6.21, (Second option is if I account for pole mass)
    # M_fric = M*b = 0.230*20 = 4.6, 0.317*20 = 6.34
    # From experiment b = 20, a = 28

    angleDD = (
        (
            g * sa - positionDD * ca - (J_fric * (-angleD)) / (m * L) 
        ) / ((k + 1) * L)
    ) * (-1.0)


    x_dot = np.array([x[1], positionDD, x[3], angleDD])
    return x_dot


def solve_cartpole(p, x_0, u, t_span, t_eval):
    sol = solve_ivp(lambda t, x: f_cartpole(x, u(t), p), t_span, x_0, t_eval=t_eval)
    wrap_angle_rad_inplace(sol.y[2, :])
    return sol

def solve_cartpole_auto_ic(y, p, start_idx=0, end_idx=-1):
    t_span = (y[start_idx, n_to_c['time']], y[end_idx-1, n_to_c['time']])
    t_eval = y[start_idx:end_idx, n_to_c['time']]
    u = interp1d(y[:, n_to_c['time']], y[:, n_to_c['Q']], kind='linear')
    x_0 = np.array([y[start_idx, n_to_c['position']],
                    y[start_idx, n_to_c['positionD']],
                    y[start_idx, n_to_c['angle']],
                    y[start_idx, n_to_c['angleD']]])
    sol = solve_cartpole(p, x_0, u, t_span, t_eval)
    return sol

def solve_cartpole_reinit(y, p, start_idx=0, end_idx=-1, steps=20):
    if end_idx == -1:
        end_idx = len(y)

    plot_vars = []

    for i in range(start_idx, end_idx-steps, steps+1):
        x_0 = np.array([y[i, position_idx],
                        y[i, positionD_idx],
                        y[i, angle_idx],
                        y[i, angleD_idx]])
        plot_vars.append(x_0)

        t_span = (y[i, n_to_c['time']], y[i+steps, n_to_c['time']])
        t_eval = y[i:i+steps, n_to_c['time']]

        u = interp1d(y[:, n_to_c['time']], y[:, n_to_c['Q']], kind='linear')
        sol = solve_cartpole(p, x_0, u, t_span, t_eval)

        for r in sol.y.transpose():
            plot_vars.append(r.tolist())

    return np.array(plot_vars)


# Save off parameters for plotting evolution
p_saved = []
@jit(nopython=True)
def cartpole_residuals_multi_step_numba(y, p, start_idx=0, end_idx=-1, steps=20):
    #p_saved.append(p)
    residuals = []
    if end_idx == -1:
        end_idx = len(y)

    for i in range(start_idx, end_idx-steps, int((steps+1)/1)):
        x_0 = np.array([y[i, position_idx],
                        y[i, positionD_idx],
                        y[i, angle_idx],
                        y[i, angleD_idx]])

        #print('start_point') 
        for j in range(1, steps+1):
            u = y[i+j, Q_idx]
            dx_dt = f_cartpole(x_0, u, p)
            
            dt = y[i+j, time_idx] - y[i+j-1, time_idx]
            x_1_obsv = np.array([y[i+j, position_idx],
                                 y[i+j, positionD_idx],
                                 y[i+j, angle_idx],
                                 y[i+j, angleD_idx]])
            x_1_pred = x_0 + dt * dx_dt
            wrap_angle_rad_inplace(x_1_pred[2:3])

            residual = x_1_pred - x_1_obsv
            residual[2] = angle_distance(x_1_pred[2], x_1_obsv[2])
            # These are written this way so individual residuals can be commented out
            residuals.append(file_residual_weight[0]*residual[0])
            residuals.append(file_residual_weight[1]*residual[1])
            residuals.append(file_residual_weight[2]*residual[2])
            residuals.append(file_residual_weight[3]*residual[3])
            x_0 = x_1_pred

    return np.array(residuals)

# Plot the results with an animation or as a final figure
def plot_progression(save=False):
    # Find the parameters that had a large change in norm
    p_saved_diff = np.array(p_saved[1:]) - np.array(p_saved[:-1])
    p_saved_diff_norm = np.linalg.norm(p_saved_diff, axis=1)
    p_saved_changed = [p for p, norm_diff in zip(p_saved, p_saved_diff_norm) if norm_diff > 1E-4]

    fig, axis = plt.subplots(2, 2)
    x_datas = []
    y_datas = []
    lines = []
    axis = axis.flatten()

    for ax in axis: 
        ax.grid()
        x_datas.append([])
        y_datas.append([])

        line1, = ax.plot([], [], lw=2)
        line2, = ax.plot([], [], lw=2)
        lines.append((line1, line2))


    i_to_c = dict((index, n_to_c[name]) for index, name in enumerate(['position', 'positionD', 'angle', 'angleD']))

    def run(p):
        # update the data
        sol = solve_cartpole_auto_ic(y, p, start_idx=file_start_index, end_idx=file_end_index)
        t_span = (y[file_start_index, n_to_c['time']], y[file_end_index-1, n_to_c['time']])


        for i, (ax, line) in enumerate(zip(axis, lines)):
            line[0].set_data(sol.t, sol.y[i, :])
            line[1].set_data(y[:, n_to_c['time']], y[:, i_to_c[i]])

            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(t_span)
            ax.figure.canvas.draw()

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

        ani = animation.FuncAnimation(fig, run, p_saved_changed, interval=10)
        ani.save('im.mp4', writer=writer)
    else:
        ani = animation.FuncAnimation(fig, run, p_saved_changed, interval=10)
        plt.show()

# Plot the results with an animation or as a final figure
def plot_progression_multistep(save=False):
    # Find the parameters that had a large change in norm
    p_saved_diff = np.array(p_saved[1:]) - np.array(p_saved[:-1])
    p_saved_diff_norm = np.linalg.norm(p_saved_diff, axis=1)
    p_saved_changed = [(i, p) for i, (p, norm_diff) in enumerate(zip(p_saved, p_saved_diff_norm)) if norm_diff > 1E-4]

    fig, axis = plt.subplots(2, 2)
    x_datas = []
    y_datas = []
    lines = []
    axis = axis.flatten()

    for ax in axis:
        ax.grid()
        x_datas.append([])
        y_datas.append([])

        line1, = ax.plot([], [], lw=2)
        line2, = ax.plot([], [], lw=2)
        lines.append((line1, line2))


    i_to_c = dict((index, n_to_c[name]) for index, name in enumerate(['position', 'positionD', 'angle', 'angleD']))

    def run(index_and_p):
        j, p = index_and_p
        # update the data
        residuals, data = cartpole_residuals_multi_step(y, p, file_start_index, file_end_index, return_plot=True)
        t_span = (y[file_start_index, n_to_c['time']], y[file_end_index-1, n_to_c['time']])


        for i, (ax, line) in enumerate(zip(axis, lines)):
            data_end_index = file_start_index+data.shape[0]
            line[0].set_data(y[file_start_index:data_end_index, n_to_c['time']], data[:, i])
            line[1].set_data(y[:, n_to_c['time']], y[:, i_to_c[i]])

            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(t_span)
            ax.set_title(j)
            ax.figure.canvas.draw()

            if i == 3:
                ax.set_ylim([-20, 20])

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

        ani = animation.FuncAnimation(fig, run, p_saved_changed, interval=10)
        ani.save('im.mp4', writer=writer)
    else:
        ani = animation.FuncAnimation(fig, run, p_saved_changed, interval=10)
        plt.show()

def plot_final(p):
    file_end_index = 2000

    sol = solve_cartpole_auto_ic(y, p, start_idx=file_start_index, end_idx=file_end_index)

    t_span = (y[file_start_index, n_to_c['time']], y[file_end_index-1, n_to_c['time']])


    plt.figure()
    plt.subplot(321)
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['position']])
    plt.plot(sol.t, sol.y[0, :])
    plt.grid()
    plt.title('position')
    plt.legend(['true', 'pred'])
    plt.xlim(t_span)

    plt.subplot(322)
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['positionD']])
    plt.plot(sol.t, sol.y[1, :])
    plt.title('positionD')
    plt.grid()
    plt.legend(['true', 'pred'])
    plt.xlim(t_span)

    plt.subplot(323)
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['angle']])
    plt.plot(sol.t, sol.y[2, :])
    plt.title('angle')
    plt.grid()
    plt.legend(['true', 'pred'])
    plt.xlim(t_span)

    plt.subplot(324)
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['angleD']])
    plt.plot(sol.t, sol.y[3, :])
    plt.title('angleD')
    plt.grid()
    plt.legend(['true', 'pred'])
    plt.xlim(t_span)
    plt.ylim([-1, 1])

    plt.subplot(325)
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['Q']])
    plt.xlim(t_span)
    plt.grid()

    plt.show()

def plot_final_multistep(p):
    data = solve_cartpole_reinit(y, p, file_start_index, file_end_plot_index, steps=file_plot_steps)
    t_span = (y[file_start_index, n_to_c['time']], y[file_end_plot_index-1, n_to_c['time']])
    data_end_index = file_start_index+data.shape[0]


    plt.figure()
    plt.subplot(321)
    #plt.plot(y_unfilt[:, n_to_c['time']], y_unfilt[:, n_to_c['position']])
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['position']])
    plt.plot(y[file_start_index:data_end_index, n_to_c['time']], data[:, 0])
    plt.grid()
    plt.title('position')
    plt.legend(['orig', 'true', 'pred'])
    plt.xlim(t_span)

    plt.subplot(322)
    #plt.plot(y_unfilt[:, n_to_c['time']], y_unfilt[:, n_to_c['positionD']])
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['positionD']])
    plt.plot(y[file_start_index:data_end_index, n_to_c['time']], data[:, 1])
    plt.title('positionD')
    plt.grid()
    plt.legend(['orig', 'true', 'pred'])
    plt.xlim(t_span)

    plt.subplot(323)
    #plt.plot(y_unfilt[:, n_to_c['time']], y_unfilt[:, n_to_c['angle']])
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['angle']])
    plt.plot(y[file_start_index:data_end_index, n_to_c['time']], data[:, 2])
    plt.title('angle')
    plt.grid()
    plt.legend(['orig', 'true', 'pred'])
    plt.xlim(t_span)
    plt.ylim([-0.25, 0.25])

    plt.subplot(324)
    #plt.plot(y_unfilt[:, n_to_c['time']], y_unfilt[:, n_to_c['angleD']])
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['angleD']])
    plt.plot(y[file_start_index:data_end_index, n_to_c['time']], data[:, 3])
    plt.title('angleD')
    plt.grid()
    plt.legend(['orig', 'true', 'pred'])
    plt.xlim(t_span)
    plt.ylim([-1, 1])

    plt.subplot(325)
    #plt.plot(y_unfilt[:, n_to_c['time']], y_unfilt[:, n_to_c['Q']])
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['Q']])
    plt.legend(['orig', 'true', 'pred'])
    plt.xlim(t_span)
    plt.title('Q')
    plt.grid()

    plt.show()


# files
file = 'cartpole-2021-07-13-17-59-28.csv'
file_start_index = 0
file_end_index = 1000 + 50
file_end_plot_index = 4316
file_plot_steps = 150
file_animate = False
file_residual_weight = np.array([0.1, 5.0, 20.0, 0.01])

y, n_to_c = read_data(file)

position_idx = n_to_c['position']
positionD_idx = n_to_c['positionD']
angle_idx = n_to_c['angle']
angleD_idx = n_to_c['angleD']
time_idx = n_to_c['time']
Q_idx = n_to_c['Q']

filt_data(y)

# Solve the problem
m_1_0 = 1.0
m_2_0 = 1.0
l_0   = 1.0
k_1_0 = 1.0
k_2_0 = 1.0
u_scale_0 = 1.0
p_0 = [m_1_0, m_2_0, l_0, k_1_0, k_2_0]
#p_0 =
bounds = ([1E-5]*5, [np.inf]*5)
bounds = ([-1]*5, [1]*5)

jac = '3-point'
method = 'trf'
loss = 'soft_l1'
f_scale = 1.5

res = least_squares(lambda p: cartpole_residuals_multi_step_numba(y, p, file_start_index, file_end_index, steps=2),
                    p_0, jac=jac, method=method, verbose=1, loss=loss, f_scale=f_scale)
print(res.x)

for i in range(3, 115, 1):
    print('steps: {}'.format(i))
    res = least_squares(lambda p: cartpole_residuals_multi_step_numba(y, p, file_start_index, file_end_index, steps=i),
                        res.x, jac=jac, method=method, verbose=1, loss=loss, f_scale=f_scale)
    print(res.x)

print('Results')
print(res.x)


if file_animate:
    #plot_progression()
    plot_progression_multistep()
    #plot_progression(save=True)

#plot_final(res.x)

y_unfilt, n_to_c = read_data(file)
plot_final_multistep(res.x)
