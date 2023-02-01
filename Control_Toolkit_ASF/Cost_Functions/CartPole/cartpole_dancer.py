import csv
from pathlib import Path
from typing import Dict

from GUI import CartPoleMainWindow
from others.globals_and_utils import get_logger

# SONG='others/Media/01 Sookie_ Sookie.mp3'
# CSV='Control_Toolkit_ASF/Cost_Functions/CartPole/cartpole_dance-sookie-sookie.csv'
# SONG="others/Media/Rolling_Stones_Satisfaction.mp3"
# CSV='Control_Toolkit_ASF/Cost_Functions/CartPole/cartpole_dance-satisfaction.csv'
# SONG_PLAYBACK_RATE=0.55 # rate to match song speed compared to real time simulation rate

import os
try:
    import vlc
except Exception as e:
    raise Exception(f'{e}: pip install python-vlc and install VLC (https://www.videolan.org/vlc/) to your system. Y'
                    f'ou may need to restart your python IDE. See also https://stackoverflow.com/questions/59014318/filenotfounderror-could-not-find-module-libvlc-dll ;'
                    f'you might need 64 bit version of VLC since default installs 32-bit version')
import sys

log = get_logger(__name__)
try:
    import winsound
except:
    log.warning('winsound is not available - will not play beeps when dance step changes')
    pass

UPDATE_INTERVAL=10 # update interval for status text and song rate update

class cartpole_dancer:
    """ Reads steps from CSV file for cartpole dancer
    """

    def __init__(self):
        """ Constructs the dance reader, opening the CSV file

        """
        self.song_file_name = None
        self.song_player = None
        self.row_iterator = None
        self.reader = None
        self.csvfile = None
        csv.register_dialect('cartpole-dancer', skipinitialspace=True)

        self.ENDTIME = 'endtime'
        self.POLICY = 'policy'
        self.OPTION = 'option'
        self.CARTPOS = 'cartpos'
        self.FREQ = 'freq'
        self.AMP = 'amp'
        self.FREQ2 = 'freq2'
        self.AMP2 = 'amp2'

        self.first_step_run=False  # used to avoid starting playing music when dance is configured in config_cost_functions but simulation not running
        self.state:str='running' # set by signals emitted by CartPoleMainWindow, set to running initially so that we read the first row of CSV to compute trajectory for init
        if CartPoleMainWindow.CartPoleMainWindowInstance and CartPoleMainWindow.CartPoleMainWindowInstance.CartPoleSimulationStateSignal:
            CartPoleMainWindow.CartPoleMainWindowInstance.CartPoleSimulationStateSignal.connect(self.process_signal)

        self.csv_file_name = None
        self.csv_file_path = None
        self.mtime = None
        self.reset_fields()
        self.paused=False
        self.time_step_started = float(0)
        self.time_dance_started = float(0)

        self.cartpole_trajectory_generator = None # set in start()
        self.iteration_counter=0 # to reduce some updates

    def reload_csv_and_song(self):
        self.csv_file_name = bytes.decode(self.cartpole_trajectory_generator.cost_function.dance_csv_file.numpy())
        if not os.path.exists(self.csv_file_name):
            raise FileNotFoundError(f'CSV file for dance {self.cartpole_trajectory_generator.cost_function.dance_song_file} not found, cannot play music')
        self.csv_file_path = Path(self.csv_file_name)
        self.mtime = self.csv_file_path.stat().st_mtime

        self.song_file_name=bytes.decode(self.cartpole_trajectory_generator.cost_function.dance_song_file.numpy())
        if not os.path.exists(self.song_file_name):
            raise FileNotFoundError(f'mp3/wave file for dance {self.song_file_name} not found, cannot play music')
        self.song_player:vlc.MediaPlayer = vlc.MediaPlayer(self.song_file_name)

        if self.song_player.set_rate(self.cartpole_trajectory_generator.cost_function.dance_song_playback_rate)==-1:
            log.warning('could not set playback rate for song')
        em = self.song_player.event_manager()
        em.event_attach(vlc.EventType.MediaPlayerEndReached, self.start) # restart dance when song restarts
        self.mtime = self.csv_file_path.stat().st_mtime
        if self.csvfile:
            self.csvfile.close()
        self.csvfile = open(self.csv_file_name, 'r')
        self.reader = csv.DictReader(filter(lambda row: row[0] != '#', self.csvfile),
                                     dialect='cartpole-dancer')  # https://stackoverflow.com/questions/14158868/python-skip-comment-lines-marked-with-in-csv-dictreader
        self.row_iterator = self.reader.__iter__()

    def reset_fields(self) -> None:
        self.time_dance_started = float(0)
        self.time_step_started = float(0)
        self.current_row = None
        self.started = False
        self.endtime = None
        self.option = None
        self.policy = None
        self.policy_number = None
        self.cartpos = None
        self.freq = None
        self.amp = None
        self.freq2 = None
        self.amp2 = None

    def start(self, time: float=0.0, cartpole_trajectory_generator=None) -> None:
        """ Starts the dance now at time in seconds

        :param time: the current simulation time in seconds, 0 by default
        :param cartpole_trajectory_generator: the trajectory generator, if not None it sets it for use by cartpole_dancer use
        """
        if cartpole_trajectory_generator:
            self.cartpole_trajectory_generator=cartpole_trajectory_generator
        self.reset_fields()
        self.reload_csv_and_song()
        self.time_step_started = float(time)
        self.time_dance_started = float(time)
        self.started = True
        self.paused=False
        if CartPoleMainWindow.CartPoleGuiSimulationState=='running':
            self.song_player.play() # only play if simulator running

    def stop(self)->None:
        self.song_player.stop()
        self.started=False

    def pause(self)->None:
        self.paused=True
        self.song_player.pause()

    def process_signal(self, signal:str):
        """ Process signal from CartPoleMainWindow

        :param signal: the string state, 'running', 'paused', 'stopped'
        """
        log.debug(f'got signal "{signal}"')
        self.state=signal
        if self.state=='paused':
            self.pause()
        elif self.state=='stopped':
            self.stop()
        elif self.state=='running':
            if self.started:
                self.start(0) # only start (song) if dance has been started, don't play if e.g. policy is balance

    def step(self, time: float)-> Dict:
        """ Does next time step, reading the csv for next step if the current one's duration has timed out

        :param time: the current simulation time in seconds

        :returns: the current step row of cartpole_dance.csv as dict of column name -> value
        """

        if self.state=='running':
            if not self.started:
                log.warning('cartpole_dancer must be start()ed before calling step() - calling start now')
                self.start(time)
            if self.endtime is None or time >=  self.time_dance_started+self.endtime:
                self._read_next_step(time)
        if self.iteration_counter % UPDATE_INTERVAL== 0:
            # setting rate dynamically produces very choppy sound; we must ensure that we run fast enough for real time by optimizing code....
            # if not CartPoleMainWindow.CartPoleMainWindowInstance.speedup_measured is None:
            #     rate = CartPoleMainWindow.CartPoleMainWindowInstance.speedup_measured
            #     self.song_player.set_rate(rate)  # TODO check effect on performance in vlc
            CartPoleMainWindow.set_status_text(self.format_step(time))
        self.iteration_counter+=1
        return self.current_row

    def _read_next_step(self, time: float) ->None:
        """ Reads next step from CSV file, reloading file from start if it has been modified since it was last read.
        Plays a beep when the step changes.

        :param time: the current time in seconds

        :returns: None
        """
        try:
            # check if file modified, if so, restart whole dance
            mtime = self.csv_file_path.stat().st_mtime
            if mtime>self.mtime:
                log.warning('cartpole_dance.csv modified, restarting dance')
                self.start(time)

            self.time_step_started=time
            self.current_row = self.reader.__next__()
            log.debug(f'At time={time:.1f} new dance step is {self.current_row}')
            if 'winsound' in sys.modules:
                winsound.Beep(1000,300) # beep

        except StopIteration:
            log.info(f'restarting dance at time={time}s')
            self.start(time)
            self.step(time)
            return
        try:
            self.endtime = float(self.current_row[self.ENDTIME])
            self.policy = self.current_row[self.POLICY][:-1] # the dqnce step 'policy" column has entries like balance1, spin2, etc
            self.policy_number=int(self.current_row[self.POLICY][-1]) # todo not sufficient for XLA / TF compiled code to see this variable change
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
        """ Formats the current step for display in GUI main window status line

        :param time: the current time in seconds

        :returns: the step string
        """
        duration=self.endtime-self.time_step_started
        timestr=f't={(time):.1f}/{self.endtime:.1f}s (dur={duration:.1f}s)'
        if not self.freq is  None and not self.freq2 is None and not self.amp is None and not self.amp2 is None:
            return f'Dance: {self.policy}/{self.option} {timestr} pos={self.cartpos:.1f}m f0/f1={self.freq:.1f}/{self.freq2:.1f}Hz a0/a1={self.amp:.2f}/{self.amp2:.2f}m'
        elif not self.freq  is None and not self.amp is None:
            return f'Dance: {self.policy}/{self.option} {timestr} pos={self.cartpos:.1f}m freq={self.freq:.1f}Hz amp={self.amp:.2f}m'
        elif not self.freq is None:
            return f'Dance: {self.policy}/{self.option} {timestr} pos={self.cartpos:.1f}m freq={self.freq:.1f}Hz'
        else:
            return f'Dance: {self.policy}/{self.option} {timestr} pos={self.cartpos:.1f}m'
