# The following classes WorkerSignals and Worker are a standard tamplete
# used to implement multithreading in PyQt6 context
# This is taking from: https://www.learnpyqt.com/tutorials/multithreading-pyqt-applications-qthreadpool/
# See also: https://realpython.com/python-pyqt-qthread/ to learn more about multithreading in PyQt5

from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot

import traceback

class WorkerSignals(QObject):
    result = pyqtSignal(object)
    finished = pyqtSignal()


class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed a, kwargs.
        '''

        # Retrieve a/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            print(traceback.format_exc())
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            if hasattr(self, 'signals'):
                self.signals.finished.emit()
