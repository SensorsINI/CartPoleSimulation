from PyQt5.QtWidgets import QApplication
from time import sleep

from CartPoleGUI._CartPoleGUI_worker_template import Worker
from CartPoleGUI._CartPoleGUI_summary_window import SummaryWindow

from CartPoleGUI.gui_default_params import random_length_globals


def generate_random_experiment(self):
    if self.load_recording:
        x = self.load_generate_conflict_msg.exec_()
        return
    self.cb_load_recorded_data.setEnabled(False)
    self.cb_save_history.setEnabled(False)
    if self.textbox_length.text() == '':
        self.CartPoleInstance.random_length = random_length_globals
    else:
        self.CartPoleInstance.random_length = float(self.textbox_length.text())

    turning_points_list = []
    if self.textbox_turning_points.text() != '':
        for turning_point in self.textbox_turning_points.text().split(', '):
            turning_points_list.append(float(turning_point))
    self.CartPoleInstance.turning_points = turning_points_list

    self.CartPoleInstance.Generate_Random_Trace_Function()
    if self.run_experiment_thread == 1:
        print('First reset the previous run')
    else:
        self.CartPoleInstance.use_pregenerated_target_position = True
        self.run_experiment_thread = 1
        # Pass the function to execute
        worker_experiment = Worker(self.experiment_thread)
        # Execute
        self.threadpool.start(worker_experiment)
        while (self.run_experiment_thread == 1):
            QApplication.processEvents()
        # If user is saving data wait till data is saved
        if self.save_history:
            while (self.saved == 0):
                QApplication.processEvents()
                sleep(0.001)

        self.CartPoleInstance.use_pregenerated_target_position = False
        self.CartPoleInstance.summary_plots()
        self.w_summary = SummaryWindow(summary_plots=self.CartPoleInstance.summary_plots)
        # Reset variables and redraw the figures
        self.reset_variables(0)
        # Draw figures
        self.CartPoleInstance.draw_constant_elements(self.fig, self.fig.AxCart, self.fig.AxSlider)
        self.canvas.draw()
        self.cb_load_recorded_data.setEnabled(True)
        self.cb_save_history.setEnabled(True)