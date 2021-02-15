from CartPoleGUI._CartPoleGUI_worker_template import Worker
from CartPoleGUI._CartPoleGUI_summary_window import SummaryWindow
from CartPoleGUI.loop_timer import loop_timer

from time import sleep

# Actions to be taken when start/stop button is clicked
def start_stop_button(self):
    # Either launch a replay function (immediately in a new thread)...
    if self.load_recording:
        worker_replay = Worker(self.thread_replay)
        self.threadpool.start(worker_replay)
    # Or start/stop a new experiment
    else:
        self.play()


def play(self):

    # Terminate experiment if it was running
    if self.run_experiment_thread == 1:
        self.run_experiment_thread = 0
        # If user is saving data wait till data is saved
        if self.save_history:
            while (self.saved == 0):
                sleep(0.001)

        self.CartPoleInstance.use_pregenerated_target_position = False

        # If
        try:
            self.CartPoleInstance.controller.controller_summary()
        except:
            pass

        if self.show_experiment_summary:
            self.CartPoleInstance.summary_plots()
            self.w_summary = SummaryWindow(summary_plots=self.CartPoleInstance.summary_plots)
        # Reset variables and redraw the figures
        self.reset_variables(0)
        # Draw figures
        self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()
        self.cb_load_recorded_data.setEnabled(True)
        self.cb_save_history.setEnabled(True)
        self.cb_show_experiment_summary.setEnabled(True)

    # Start experiment if it was not running
    elif self.run_experiment_thread == 0:
        self.cb_save_history.setEnabled(False)
        self.cb_load_recorded_data.setEnabled(False)
        self.cb_show_experiment_summary.setEnabled(False)
        speedup_updated = self.get_speedup()
        try:
            self.CartPoleInstance.controller.reset()
        except:
            print('Controller reset not done')
        if speedup_updated:
            self.reset_variables(self.reset_mode)
            self.looper.dt_target = self.dt_main_simulation / self.speedup
            self.run_experiment_thread = 1
            # Pass the function to execute
            worker_calculations = Worker(self.experiment_thread)
            # Execute
            self.threadpool.start(worker_calculations)


def thread_replay(self):
    # Check what is in the csv textbox
    csv_name = self.textbox.text()
    history_pd = self.CartPoleInstance.load_history_csv(csv_name=csv_name)
    # TODO: Calculate dt from time here if you want to load data with variable dt

    # Check speedup which user provided with GUI
    self.get_speedup()

    # Define loop timer for now with arbitrary dt
    replay_looper = loop_timer(dt_target=0.0)

    # Start looper
    replay_looper.start_loop()
    for index, row in history_pd.iterrows():
        self.CartPoleInstance.s.position = row['s.position']
        self.CartPoleInstance.s.positionD = row['s.positionD']
        self.CartPoleInstance.s.angle = row['s.angle']
        self.CartPoleInstance.time = row['time']
        self.CartPoleInstance.dt = row['dt']
        self.CartPoleInstance.u = row['u']
        self.CartPoleInstance.Q = row['Q']
        self.CartPoleInstance.target_position = row['target_position']
        self.slider_value = self.CartPoleInstance.target_position

        dt_target = (self.CartPoleInstance.dt / self.speedup)
        replay_looper.dt_target = dt_target

        replay_looper.sleep_leftover_time()

    dict_history = history_pd.to_dict(orient='list')
    self.CartPoleInstance.dict_history = dict_history
    self.CartPoleInstance.summary_plots()
    self.reset_variables(0)