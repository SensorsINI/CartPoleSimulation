from PyQt5.QtWidgets import QVBoxLayout, QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# Window displaying summary (matplotlib plots) of an experiment with CartPole after clicking Stop button
# (if experiment was previously running)
class SummaryWindow(QWidget):
    def __init__(self, summary_plots=None):
        super(SummaryWindow, self).__init__()

        ## Create GUI Layout
        layout = QVBoxLayout()

        self.fig, self.axs = summary_plots()
        for axis in self.axs:
            axis.tick_params(axis='both', which='major', labelsize=9)
            axis.xaxis.label.set_fontsize(11)
            axis.yaxis.label.set_fontsize(11)
        self.fig.subplots_adjust(bottom=0.11)
        self.fig.subplots_adjust(hspace=0.3)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.toolbar)
        lf.addWidget(self.canvas)
        layout.addLayout(lf)


        # add quit button
        bq = QPushButton("QUIT")
        bq.pressed.connect(self.close)
        lb = QVBoxLayout()  # Layout for buttons
        lb.addWidget(bq)
        layout.addLayout(lb)

        self.setLayout(layout)
        self.setGeometry(100, 150, 1500, 800)
        # self.adjustSize()
        self.show()
        self.setWindowTitle('Last experiment summary')