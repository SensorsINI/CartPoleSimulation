from PyQt5.QtWidgets import QApplication


# This method initiate calculation of simulation and iterative updates of Cart state
# It also measures time intervals for real time simulation
# implements a termination condition if Pole went out of control
# and initiate saving to a .csv file
def experiment_thread(self):
    self.looper.start_loop()
    while (self.run_experiment_thread):

        # Calculations of the Cart state in the next timestep
        self.CartPoleInstance.update_state()

        # Ensure that the animation drawing function can access CartPoleInstance at this moment
        QApplication.processEvents()

        if self.CartPoleInstance.use_pregenerated_target_position == True and self.CartPoleInstance.time >= self.CartPoleInstance.t_max_pre:
            # print('Terminating!')
            self.run_experiment_thread = 0

        # FIXME: when Speedup empty in GUI I expected inf speedup but got error Loop timer was not initialized properly
        self.looper.sleep_leftover_time()

    # print('Welcome')
    # Save simulation history if user chose to do so at the end of the simulation
    if self.save_history:
        csv_name = self.textbox.text()
        self.CartPoleInstance.save_history_csv(csv_name=csv_name, mode='init')
        self.CartPoleInstance.save_history_csv(csv_name=csv_name, mode='save offline')
        self.saved = 1