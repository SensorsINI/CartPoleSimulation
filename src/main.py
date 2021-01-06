# Import module to interact with OS
import sys

# Import functions from PyQt5 module (creating GUI)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
# Import custom made elements of GUI
from src.gui import MainWindow


# This piece of code gives a custom ID to our application
# It is essential for packaging
# If you do not create .exe file or installer this snippet is probably redundant
try:
    # Include in try/except block if you're also targeting Mac/Linux
    from PyQt5.QtWinExtras import QtWin
    myappid = 'INI.CART.IT.V1'
    QtWin.setCurrentProcessExplicitAppUserModelID(myappid)    
except ImportError:
    pass

##############################################################################
##############################################################################
##############################################################################
    
# Run only if this file is run as a main file, that is not as an imported module
if __name__ == "__main__":

    # The operations are wrapped into a function run_app
    # This is necessary to make the app instance disappear after closing the window
    # At least while using Spyder 4 IDE,
    # this is the only way allowing program to be restarted without restarting Python kernel
    def run_app():
        # Creat an instance of PyQt5 application
        # Every PyQt5 application has to contain this line
        app = QApplication(sys.argv)
        # Set the default icon to use for all the windows of our application
        app.setWindowIcon(QIcon('./src/myicon.ico'))
        # Create an instance of the GUI window.
        window = MainWindow()
        window.show()
        # Next line hands the control over to Python GUI
        app.exec_()
    
    # Have fun!
    run_app()
