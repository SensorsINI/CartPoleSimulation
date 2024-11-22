"""
Main window of CartPole GUI
"""

# Import functions from PyQt6 module (creating GUI)
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget

# This import mus go before pyplot so also before our scripts
from matplotlib import use, get_backend

# Use Agg if not in scientific mode of Pycharm
if get_backend() != 'module://backend_interagg':
    use('Agg')

from GUI._CartPoleGUI_GuiActions import CartPole_GuiActions, TrackHalfLength
from GUI._CartPoleGUI_GuiLayout import CartPole_GuiLayout


# Class implementing the main window of CartPole GUI
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # Change geometry of the main window
        self.setGeometry(300, 300, 2500, 1000)

        self.GuiLayout = CartPole_GuiLayout()
        self.GuiActions = CartPole_GuiActions(self.GuiLayout)
        self.GuiLayout.create_layout(self, self.GuiActions, quit_callback=self.quit_application)

        # Create an instance of a GUI window
        w = QWidget()
        w.setLayout(self.GuiLayout.layout)
        self.setCentralWidget(w)
        self.show()
        self.setWindowTitle('CartPole Simulator')

        self.GuiActions.finish_initialization()

    def quit_application(self):
        self.GuiActions.quit_application()
        self.close()
        # The standard command
        # It seems however not to be working by its own
        # I don't know how it works
        QApplication.quit()
