import csv
from pathlib import Path

from others.globals_and_utils import get_logger

# import os

log = get_logger(__name__)


class cartpole_dancer:
    def __init__(self):
        fp = 'Control_Toolkit_ASF/Cost_Functions/CartPole/cartpole_dance.csv'  # os.path.join('Control_Toolkit_ASF','Cost_Functios','cartpole_dance.csv')
        fpath = Path(fp)
        self.mtime = fpath.stat().st_mtime
        self.csvfile = open(fp, 'r')
        csv.register_dialect('cartpole-dancer', skipinitialspace=True)

        self.DURATION = 'duration'
        self.POLICY = 'policy'
        self.OPTION = 'option'
        self.CARTPOS = 'cartpos'
        self.FREQ = 'freq'
        self.AMP = 'amp'

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

    def start(self, time: float):
        """ Starts the dance now at time time"""

        self._reset_fields()
        self.csvfile.seek(0)
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
        return self.current_row

    def _read_next_step(self, time: float):
        try:
            self.time_step_started=time
            self.current_row = self.reader.__next__()
            log.debug(f'At time={time} new dance step is {self.current_row}')
        except StopIteration:
            self.start(time)
            return self.step(time)
        self.duration = float(self.current_row[self.DURATION])
        self.policy = self.current_row[self.POLICY]
        self.option = self.current_row[self.OPTION]
        self.cartpos = float(self.current_row[self.CARTPOS])
        self.freq = float(self.current_row[self.FREQ]) if self.current_row[self.FREQ] else None
        self.amp = float(self.current_row[self.AMP]) if self.current_row[self.AMP] else None
