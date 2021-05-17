
# region Imports and setting matplotlib backend

# Import functions from PyQt5 module (creating GUI)

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox, \
    QComboBox, QSlider, QFrame, QButtonGroup, QRadioButton
from PyQt5.QtCore import Qt



# Import matplotlib
# This import mus go before pyplot so also before our scripts
from matplotlib import use, get_backend
# Use Agg if not in scientific mode of Pycharm
if get_backend() != 'module://backend_interagg':
    use('Agg')

# Some more functions needed for interaction of matplotlib with PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import colors

# Other imports for GUI
import sys

# endregion

# region Set color map for the plots
cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

# endregion

def run_test_gui(featurs, titles, ground_truth, predictions_list, time_axis):
    # Creat an instance of PyQt5 application
    # Every PyQt5 application has to contain this line
    app = QApplication(sys.argv)
    # Create an instance of the GUI window.
    window = MainWindow(featurs, titles, ground_truth, predictions_list, time_axis)
    window.show()
    # Next line hands the control over to Python GUI
    app.exec_()

# Class implementing the main window of CartPole GUI
class MainWindow(QMainWindow):

    def __init__(self,
                 features,
                 titles,
                 ground_truth,
                 predictions_list,
                 time_axis,
                 *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.features = features
        self.titles = titles
        self.ground_truth = ground_truth
        self.predictions_list = predictions_list
        self.time_axis = time_axis

        self.dataset = predictions_list[0]

        self.max_horizon = self.predictions_list[0].shape[-2]-1
        self.horizon = self.max_horizon//2

        self.show_all = False
        self.downsample = False
        self.current_point_at_timeaxis = (self.time_axis.shape[0]-self.max_horizon)//2
        self.feature_to_display = self.features[0]

        # region - Create container for top level layout
        layout = QVBoxLayout()
        # endregion

        # region - Change geometry of the main window
        self.setGeometry(300, 300, 2500, 1000)
        # endregion

        # region - Matplotlib figures (CartPole drawing and Slider)
        # Draw Figure
        self.fig = Figure(figsize=(25, 10))  # Regulates the size of Figure in inches, before scaling to window size.
        self.canvas = FigureCanvas(self.fig)
        self.fig.Ax = self.canvas.figure.add_subplot(111)
        self.redraw_canvas()

        self.toolbar = NavigationToolbar(self.canvas, self)

        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.toolbar)
        lf.addWidget(self.canvas)
        layout.addLayout(lf)

        # endregion

        l_sl = QHBoxLayout()

        # region - Slider position
        l_sl_p = QVBoxLayout()
        l_sl_p.addWidget(QLabel('"Current" point in time:'))
        self.sl_p = QSlider(Qt.Horizontal)
        self.sl_p.setMinimum(0)
        self.sl_p.setMaximum(self.time_axis.shape[0]-self.max_horizon-1)
        self.sl_p.setValue((self.time_axis.shape[0]-self.max_horizon)//2)
        self.sl_p.setTickPosition(QSlider.TicksBelow)
        # self.sl_p.setTickInterval(5)

        l_sl_p.addWidget(self.sl_p)
        self.sl_p.valueChanged.connect(self.slider_position_f)
        # endregion

        # region - Slider horizon
        l_sl_h = QVBoxLayout()
        l_sl_h.addWidget(QLabel('Prediction horizon:'))
        self.sl_h = QSlider(Qt.Horizontal)
        self.sl_h.setMinimum(0)
        self.sl_h.setMaximum(self.max_horizon)
        self.sl_h.setValue(self.max_horizon//2)
        self.sl_h.setTickPosition(QSlider.TicksBelow)
        # self.sl_h.setTickInterval(5)
        # endregion

        l_sl_h.addWidget(self.sl_h)
        self.sl_h.valueChanged.connect(self.slider_horizon_f)

        separatorLine = QFrame()
        separatorLine.setFrameShape( QFrame.VLine )
        separatorLine.setFrameShadow( QFrame.Raised )

        l_sl.addLayout(l_sl_p)
        l_sl.addWidget(separatorLine)
        l_sl.addLayout(l_sl_h)
        layout.addLayout(l_sl)


        # region - Make strip of layout for checkboxes and compobox
        l_cb = QHBoxLayout()

        # region -- Checkbox: Show all
        self.cb_show_all = QCheckBox('Show all', self)
        if self.show_all:
            self.cb_show_all.toggle()
        self.cb_show_all.toggled.connect(self.cb_show_all_f)
        l_cb.addWidget(self.cb_show_all)
        # endregion

        # region -- Checkbox: Save/don't save experiment recording
        self.cb_downsample = QCheckBox('Downsample predictions (X2)', self)
        if self.downsample:
            self.cb_downsample.toggle()
        self.cb_downsample.toggled.connect(self.cb_downsample_f)
        l_cb.addWidget(self.cb_downsample)
        # endregion

        # region Radio buttons to chose the dataset

        self.rbs_datasets = []

        for title in self.titles:
            self.rbs_datasets.append(QRadioButton(title))

        # Ensures that radio buttons are exclusive
        self.datasets_buttons_group = QButtonGroup()
        for button in self.rbs_datasets:
            self.datasets_buttons_group.addButton(button)

        lr_d = QHBoxLayout()
        lr_d.addStretch(1)
        lr_d.addWidget(QLabel('Dataset:'))
        for rb in self.rbs_datasets:
            rb.clicked.connect(self.RadioButtons_detaset_selection)
            lr_d.addWidget(rb)
        lr_d.addStretch(1)

        self.rbs_datasets[0].setChecked(True)
        # if len(self.predictions_list) < 2:
        #     self.rbs_datasets[1].setEnabled(False)
        #     # self.rbs_datasets[2].setEnabled(False)

        l_cb.addLayout(lr_d)

        # endregion

        # region -- Combobox: Select feature to plot
        l_cb.addWidget(QLabel('Feature to plot:'))
        self.cb_select_feature = QComboBox()
        self.cb_select_feature.addItems(self.features)
        self.cb_select_feature.currentIndexChanged.connect(self.cb_select_feature_f)
        self.cb_select_feature.setCurrentText(self.features[0])
        l_cb.addWidget(self.cb_select_feature)

        # region - Add checkboxes to layout
        layout.addLayout(l_cb)
        # endregion

        # endregion

        # region - QUIT button
        bq = QPushButton("QUIT")
        bq.pressed.connect(self.quit_application)
        lb = QVBoxLayout()  # Layout for buttons
        lb.addWidget(bq)
        layout.addLayout(lb)
        # endregion

        # region - Create an instance of a GUI window
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.show()
        self.setWindowTitle('Testing System Model')

        # endregion


    def slider_position_f(self, value):
        self.current_point_at_timeaxis = int(value)

        self.redraw_canvas()

    def slider_horizon_f(self, value):
        self.horizon = int(value)

        self.redraw_canvas()

    def cb_show_all_f(self, state):
        if state:
            self.show_all = True
        else:
            self.show_all = False

        self.redraw_canvas()

    def cb_downsample_f(self, state):
        if state:
            self.downsample = True
        else:
            self.downsample = False

        self.redraw_canvas()

    def RadioButtons_detaset_selection(self):

        for i in range(len(self.rbs_datasets)):
            if self.rbs_datasets[i].isChecked():
                self.dataset = self.predictions_list[i]

        self.redraw_canvas()

    def cb_select_feature_f(self):
        self.feature_to_display = self.cb_select_feature.currentText()
        self.redraw_canvas()

    # The actions which has to be taken to properly terminate the application
    # The method is evoked after QUIT button is pressed
    # TODO: Can we connect it somehow also the the default cross closing the application?
    #   If you find out, please correct for the same in CartPole simulator
    def quit_application(self):
        # Closes the GUI window
        self.close()
        # The standard command
        # It seems however not to be working by its own
        # I don't know how it works
        QApplication.quit()


    def redraw_canvas(self):

        self.fig.Ax.clear()

        brunton_widget(self.features, self.ground_truth, self.dataset, self.time_axis,
                       axs=self.fig.Ax,
                       current_point_at_timeaxis=self.current_point_at_timeaxis,
                       feature_to_display=self.feature_to_display,
                       max_horizon=self.max_horizon,
                       horizon=self.horizon,
                       show_all=self.show_all,
                       downsample=self.downsample)

        self.canvas.draw()



def brunton_widget(features, ground_truth, predictions_array, time_axis, axs=None,
                   current_point_at_timeaxis=None,
                   feature_to_display=None,
                   max_horizon=10, horizon=None,
                   show_all=True,
                   downsample=False):

    # Start at should be done bu widget (slider)
    if current_point_at_timeaxis is None:
        current_point_at_timeaxis = ground_truth.shape[0]//2

    if feature_to_display is None:
        feature_to_display = features[0]

    if horizon is None:
        horizon = max_horizon

    feature_idx = features.index(feature_to_display)

    # Brunton Plot
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(18, 10), sharex=True)

    axs.plot(time_axis, ground_truth[:, feature_idx], 'k:', markersize=12, label='Ground Truth')
    y_lim = axs.get_ylim()
    prediction_distance = []
    axs.set_ylabel(feature_to_display, fontsize=18)
    axs.set_xlabel('Time [s]', fontsize=18)
    for i in range(horizon):

        if not show_all:
            axs.plot(time_axis[current_point_at_timeaxis], ground_truth[current_point_at_timeaxis, feature_idx],
                     'g.', markersize=16, label='Start')
            prediction_distance.append(predictions_array[current_point_at_timeaxis, i+1, feature_idx])
            if downsample:
                if (i % 2) == 0:
                    continue
            axs.plot(time_axis[current_point_at_timeaxis+i+1], prediction_distance[i],
                        c=cmap(float(i)/max_horizon),
                        marker='.')

        else:
            prediction_distance.append(predictions_array[:-(i+1), i+1, feature_idx])
            if downsample:
                if (i % 2) == 0:
                    continue
            axs.plot(time_axis[i+1:], prediction_distance[i],
                        c=cmap(float(i)/max_horizon),
                        marker='.', linestyle = '')

    # axs.set_ylim(y_lim)

    plt.show()