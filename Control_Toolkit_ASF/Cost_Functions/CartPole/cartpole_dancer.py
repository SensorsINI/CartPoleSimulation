import csv
from pathlib import Path

from GUI import MainWindow
from others.globals_and_utils import get_logger



# import os

log = get_logger(__name__)
try:
    import winsound
except:
    pass

class cartpole_dancer:
    def __init__(self):
        csv.register_dialect('cartpole-dancer', skipinitialspace=True)
        self.fp = 'Control_Toolkit_ASF/Cost_Functions/CartPole/cartpole_dance.csv'  # os.path.join('Control_Toolkit_ASF','Cost_Functios','cartpole_dance.csv')
        self.fpath = Path(self.fp)
        self.csvfile=None
        self.mtime = self.fpath.stat().st_mtime

        self.DURATION = 'duration'
        self.POLICY = 'policy'
        self.OPTION = 'option'
        self.CARTPOS = 'cartpos'
        self.FREQ = 'freq'
        self.AMP = 'amp'
        self.FREQ2 = 'freq2'
        self.AMP2 = 'amp2'

        self._reset_fields()

    def _reset_fields(self):
        self.time_step_started = float(0)
        self.current_row = None
        self._started = False
        self.duration = None
        self.option = None
        self.policy = None
        self.cartpos = None
        self.freq = None
        self.amp = None
        self.freq2 = None
        self.amp2 = None

    def start(self, time: float):
        """ Starts the dance now at time time"""

        self._reset_fields()
        self.mtime = self.fpath.stat().st_mtime

        if self.csvfile:
            self.csvfile.close()
        self.csvfile = open(self.fp, 'r')
        self.reader = csv.DictReader(filter(lambda row: row[0] != '#', self.csvfile),
                                     dialect='cartpole-dancer')  # https://stackoverflow.com/questions/14158868/python-skip-comment-lines-marked-with-in-csv-dictreader
        self.time_step_started = float(time)
        self._started = True
        self.row_iterator = self.reader.__iter__()

    def step(self, time: float):
        if not self._started:
            raise Exception('cartpole_dancer must be start()ed before calling step()')
        if self.duration is None or time >= self.time_step_started + self.duration:
            self._read_next_step(time)
        MainWindow.set_status_text(self.format_step(time))
        return self.current_row

    def _read_next_step(self, time: float):
        try:
            # check if file modified, if so, restart whole dance
            mtime = self.fpath.stat().st_mtime
            if mtime>self.mtime:
                log.warning('cartpole_dance.csv modified, restarting dance')
                self.start(time)

            self.time_step_started=time
            self.current_row = self.reader.__next__()
            log.debug(f'At time={time:.1f} new dance step is {self.current_row}')
            if winsound:
                winsound.Beep(1000,200) # beep

        except StopIteration:
            self.start(time)
            return self.step(time)
        try:
            self.duration = float(self.current_row[self.DURATION])
            self.policy = self.current_row[self.POLICY]
            self.option = self.current_row[self.OPTION]
            self.cartpos = float(self.current_row[self.CARTPOS])
            self.freq = self.convert_float(self.FREQ)
            self.amp = self.convert_float(self.AMP)
            self.freq2 = self.convert_float(self.FREQ2)
            self.amp2 = self.convert_float(self.AMP2)
        except ValueError as e:
            log.warning(f'could not convert value in row {self.current_row}: {e}')

    def convert_float(self, key):
        if self.current_row[key] and self.current_row[key]!='':
            return float(self.current_row[key])
        else:
            return None

    def format_step(self, time:float) -> str:
        if not self.freq is  None and not self.freq2 is None and not self.amp is None and not self.amp2 is None:
            return f'Dance: {self.policy}/{self.option} t={(time-self.time_step_started):.1f}/{self.duration:.1f}s pos={self.cartpos:.1f}m f0/f1={self.freq:.1f}/{self.freq2:.1f}Hz a0/a1={self.amp:.2f}/{self.amp2:.2f}m'
        elif not self.freq  is None and not self.amp is None:
            return f'Dance: {self.policy}/{self.option} t={(time-self.time_step_started):.1f}/{self.duration:.1f}s pos={self.cartpos:.1f}m freq={self.freq:.1f}Hz amp={self.amp:.2f}m'
        elif not self.freq is None:
            return f'Dance: {self.policy}/{self.option} t={(time-self.time_step_started):.1f}/{self.duration:.1f} pos={self.cartpos:.1f}m freq={self.freq:.1f}Hz'
        else:
            return f'Dance: {self.policy}/{self.option} t={(time-self.time_step_started):.1f}/{self.duration:.1f}s pos={self.cartpos:.1f}m'
