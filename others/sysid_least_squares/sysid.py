import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy import signal
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

def read_data(file):
    data = []
    with open(file, newline='') as csvfile:
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

# copied from _CartPole_mathematical_helpers.py
def wrap_angle_rad_inplace(angle: np.ndarray) -> None:
    Modulo = np.fmod(angle, 2 * np.pi)  # positive modulo
    neg_wrap, pos_wrap = Modulo < -np.pi, Modulo > np.pi
    angle[neg_wrap] = Modulo[neg_wrap] + 2 * np.pi
    angle[pos_wrap] = Modulo[pos_wrap] - 2 * np.pi
    angle[~(neg_wrap | pos_wrap)] = Modulo[~(neg_wrap | pos_wrap)]

def angle_distance(x, y):
    dist = x - y
    if np.abs(dist) > np.pi:
        dist -= np.sign(dist) * np.pi
    return dist


# copied from cartpole_model.py
def f_cartpole(x, u, p):
    m = p[0]
    M = p[1]
    L = p[2]
    M_fric = p[3]
    J_fric = p[4]
    u_scale = p[5]
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

def solve_cartpole_auto_ic(y, p, start_idx=0):
    t_span = (y[start_idx, n_to_c['time']], y[-1, n_to_c['time']])
    t_eval = y[start_idx:, n_to_c['time']]
    u = interp1d(y[:, n_to_c['time']], y[:, n_to_c['Q']], kind='linear')
    x_0 = np.array([y[start_idx, n_to_c['position']],
                    y[start_idx, n_to_c['positionD']],
                    y[start_idx, n_to_c['angle']],
                    y[start_idx, n_to_c['angleD']]])
    sol = solve_cartpole(p, x_0, u, t_span, t_eval)
    return sol

# Save off parameters for plotting evolution
p_saved = []
def cartpole_residuals_multi_step(y, p, start_idx=0, end_idx=-1, steps=25):
    p_saved.append(p)
    residuals = []
    if end_idx == -1:
        end_idx = len(y)

    for i in range(start_idx, end_idx-1-steps, steps):
        x_0 = [y[i, n_to_c['position']],
               y[i, n_to_c['positionD']],
               y[i, n_to_c['angle']],
               y[i, n_to_c['angleD']]]
        #print('start_point') 
        for j in range(0, steps):
            u = y[i+j, n_to_c['u']]
            dx_dt = f_cartpole(x_0, u, p)
            
            dt = y[i+j+1, n_to_c['time']] - y[i+j, n_to_c['time']]
            x_1_obsv = [y[i+j+1, n_to_c['position']],
                        y[i+j+1, n_to_c['positionD']],
                        y[i+j+1, n_to_c['angle']],
                        y[i+j+1, n_to_c['angleD']]]
            x_1_pred = x_0 + dt * dx_dt
            wrap_angle_rad_inplace(x_1_pred[2:3])
            residual = x_1_pred - x_1_obsv
            residual[0] *= 100.0
            residual[2] = angle_distance(x_1_pred[2], x_1_obsv[2])
            #dca = np.cos(x_1_pred[2]) - np.cos(x_1_obsv[2])
            #dsa = np.sin(x_1_pred[2]) - np.sin(x_1_obsv[2])
            #print('hi')
            #print(dx_dt)
            #print(x_1_pred)
            residuals.extend(residual)
            #residuals.append(residual[0])
            #residuals.extend([dca, dsa])
            x_0 = x_1_pred

    residuals = np.array(residuals)

    return residuals

def cartpole_euler(y, p, start_idx=0, end_idx=-1):
    x = []
    if end_idx == -1:
        end_idx = len(y)

    x_0 = [y[start_idx, n_to_c['position']],
           y[start_idx, n_to_c['positionD']],
           y[start_idx, n_to_c['angle']],
           y[start_idx, n_to_c['angleD']]]
    x.append(x_0)

    for i in range(start_idx+1, end_idx-1):
        #print('start_point') 
        u = y[i, n_to_c['u']]
        dx_dt = f_cartpole(x_0, u, p)
        
        dt = y[i+1, n_to_c['time']] - y[i, n_to_c['time']]
        x_1_pred = x_0 + dt * dx_dt
        wrap_angle_rad_inplace(x_1_pred[2:3])
        x.append(x_1_pred)

        x_0 = x_1_pred

    x = np.array(x)
    print(y.shape)
    print(x.shape)

    return x



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
        sol = solve_cartpole_auto_ic(y, p, start_idx=file_start_index)


        for i, (ax, line) in enumerate(zip(axis, lines)):
            line[0].set_data(sol.t, sol.y[i, :])
            line[1].set_data(y[:, n_to_c['time']], y[:, i_to_c[i]])

            ax.relim()
            ax.autoscale_view()
            ax.figure.canvas.draw()

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

        ani = animation.FuncAnimation(fig, run, p_saved_changed, interval=10)
        ani.save('im.mp4', writer=writer)
    else:
        ani = animation.FuncAnimation(fig, run, p_saved_changed, interval=10)
        plt.show()

def plot_final():
    #sol = solve_cartpole_auto_ic(y, res.x, start_idx=file_start_index)
    x = cartpole_euler(y, res.x, start_idx=file_start_index, end_idx=file_end_index)
    t_span = (y[file_start_index, n_to_c['time']], y[file_end_index-1, n_to_c['time']])

    plt.figure()
    plt.subplot(321)
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['position']])
    #plt.plot(sol.t, sol.y[0, :])
    plt.plot(y[file_start_index:file_end_index, n_to_c['time']], x[:, 0])
    plt.grid()
    plt.title('position')
    plt.legend(['true', 'pred'])
    plt.xlim(t_span)

    plt.subplot(322)
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['positionD']])
    #plt.plot(sol.t, sol.y[1, :])
    plt.plot(y[file_start_index:file_end_index, n_to_c['time']], x[:, 1])
    plt.title('positionD')
    plt.grid()
    plt.legend(['true', 'pred'])
    plt.xlim(t_span)

    plt.subplot(323)
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['angle']])
    #plt.plot(sol.t, sol.y[2, :])
    plt.plot(y[file_start_index:file_end_index, n_to_c['time']], x[:, 2])
    plt.title('angle')
    plt.grid()
    plt.legend(['true', 'pred'])
    plt.xlim(t_span)

    plt.subplot(324)
    plt.plot(y[:, n_to_c['time']], y[:, n_to_c['angleD']])
    #plt.plot(sol.t, sol.y[3, :])
    plt.plot(y[file_start_index:file_end_index, n_to_c['time']], x[:, 3])
    plt.title('angleD')
    plt.grid()
    plt.legend(['true', 'pred'])
    plt.xlim(t_span)

    plt.subplot(325)
    plt.plot(y[:, n_to_c['time']], res.x[5]*y[:, n_to_c['Q']])
    #plt.plot(y[:, n_to_c['time']], y[:, n_to_c['u']])
    plt.legend(['Q*scale', 'u'])
    plt.xlim(t_span)
    plt.grid()

    plt.show()


# files
#file = 'save.csv'
#file_start_index = 95
#file_end_index = -1
#file_animate = False # solve_ivp gets stuck animating this parameter progression

file = 'save2.csv'
file_start_index = 0
file_end_index = -1
file_animate = False

y, n_to_c = read_data(file)

# Solve the problem
m_1_0 = 1.0
m_2_0 = 1.0
l_0   = 1.0
k_1_0 = 0.0
k_2_0 = 0.0
u_scale_0 = 1.0
p_0 = [m_1_0, m_2_0, l_0, k_1_0, k_2_0, u_scale_0]
bounds = ([0.001, 0.001, 0.001, 0.0, 0.0, 0.001], [1E5, 1E5, 1E5, 1E5, 1E5, 1E5])

#res = least_squares(lambda p: cartpole_residuals_one_step(y, p), p_0, bounds=bounds, verbose=2)
res = least_squares(lambda p: cartpole_residuals_multi_step(y, p, file_start_index, file_end_index), p_0, bounds=bounds, verbose=2)
print('Results')
print(res.x)

# True parameters for comparison
print('True')
p_true = [0.087, 0.23, 0.1975, 6.34, 0.00025, 1.0]
print(p_true)
print('Percent error')
print(100 * (res.x / p_true - 1))

if file_animate:
    plot_progression()
    #plot_progression(save=True)
plot_final()
