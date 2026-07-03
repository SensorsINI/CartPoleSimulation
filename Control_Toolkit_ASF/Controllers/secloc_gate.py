import os
import atexit
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX
from SI_Toolkit.load_and_normalize import load_yaml


SECLOC_CONFIG_PATH = os.path.join("Control_Toolkit_ASF", "config_secloc.yml")


class SeclocConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, secloc):
        self.secloc = secloc
        self.config_path = os.path.abspath(secloc.config_path)

    def on_modified(self, event):
        if os.path.abspath(event.src_path) == self.config_path:
            self.secloc.reload_config_from_file_flag = True


class SeclocGate:
    def __init__(self, log_base, ref_period, dead_ang, dead_pos):
        self.log_base = log_base
        self.ref_period = ref_period
        self.dead_ang = dead_ang
        self.dead_pos = dead_pos
        self.reset()

    @classmethod
    def from_config(cls, config_controller):
        return cls(
            log_base=config_controller["log_base"],
            ref_period=config_controller["ref_period"],
            dead_ang=config_controller["dead_ang"],
            dead_pos=config_controller["dead_pos"],
        )

    @classmethod
    def from_config_file(cls, config_name="default"):
        config_secloc, _ = load_yaml(SECLOC_CONFIG_PATH, return_path=True)
        return cls.from_config(dict(config_secloc[config_name]))

    def update_from_config(self, config_controller):
        self.log_base = config_controller["log_base"]
        self.ref_period = config_controller["ref_period"]
        self.dead_ang = config_controller["dead_ang"]
        self.dead_pos = config_controller["dead_pos"]

    def update_ref_period_from_config(self, config_controller):
        self.ref_period = config_controller["ref_period"]

    def start_config_watcher(
        self,
        config_name="default",
    ):
        if hasattr(self, "config_observer"):
            return

        self.config_name = config_name
        self.reload_config_from_file_flag = False
        _, config_path = load_yaml(SECLOC_CONFIG_PATH, return_path=True)
        self.config_path = config_path
        self.config_observer = Observer()
        self.config_handler = SeclocConfigChangeHandler(self)
        self.config_observer.schedule(
            self.config_handler,
            os.path.dirname(os.path.abspath(self.config_path)),
            recursive=False,
        )
        atexit.register(self.stop_config_watcher)
        self.config_observer.start()

    def stop_config_watcher(self):
        if not hasattr(self, "config_observer"):
            return

        if self.config_observer is not None and self.config_observer.is_alive():
            self.config_observer.stop()
            self.config_observer.join()
        self.config_observer = None

    def update_from_config_file_if_needed(self):
        if not getattr(self, "reload_config_from_file_flag", False):
            return

        self.reload_config_from_file_flag = False
        try:
            config_secloc, config_path = load_yaml(SECLOC_CONFIG_PATH, return_path=True)
            self.config_path = config_path
            self.update_from_config(dict(config_secloc[self.config_name]))
            print(
                "Secloc config reloaded: "
                f"log_base={self.log_base}, ref_period={self.ref_period}, "
                f"dead_ang={self.dead_ang}, dead_pos={self.dead_pos}"
            )
        except Exception as exc:
            print(f"Secloc config reload failed, keeping previous parameters: {exc}")

    def __del__(self):
        self.stop_config_watcher()

    def reset(self):
        self.ang_last_shift = 0.0001
        self.pos_last_shift = 0.0001
        self.time_last = None

    def time_difference(self, time=None):
        if self.time_last is None:
            return self.ref_period
        return time - self.time_last

    def should_sample(self, s, target_position, time=None, time_difference=None):
        if time_difference is None:
            time_difference = self.time_difference(time)

        if time_difference + self.ref_period / 20 < self.ref_period:
            return False

        spike = False
        ang_shift = s[ANGLE_IDX]
        pos_shift = s[POSITION_IDX] - target_position

        if ang_shift < 0:
            ang_shift = -ang_shift
        if pos_shift < 0:
            pos_shift = -pos_shift

        if (ang_shift > self.dead_ang) and (self.ang_last_shift != 0):
            ang_ratio_inc = ang_shift / self.ang_last_shift
            ang_ratio_dec = 1.0 / ang_ratio_inc
            if (ang_ratio_inc >= self.log_base) or (ang_ratio_dec >= self.log_base):
                self.ang_last_shift = ang_shift
                spike = True
        elif (pos_shift > self.dead_pos) and (self.pos_last_shift != 0):
            pos_ratio_inc = pos_shift / self.pos_last_shift
            pos_ratio_dec = 1.0 / pos_ratio_inc
            if (pos_ratio_inc >= self.log_base) or (pos_ratio_dec >= self.log_base):
                self.pos_last_shift = pos_shift
                spike = True

        if spike:
            self.time_last = time

        return spike
