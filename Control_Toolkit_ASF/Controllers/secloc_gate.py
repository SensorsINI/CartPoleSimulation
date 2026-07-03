import os
import atexit
from collections import deque
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
    def __init__(self, log_base, ref_period, dead_ang, dead_pos, status_window_size=100):
        self.log_base = log_base
        self.ref_period = ref_period
        self.dead_ang = dead_ang
        self.dead_pos = dead_pos
        self.status_window_size = status_window_size
        self.reset()

    @classmethod
    def from_config(cls, config_controller):
        return cls(
            log_base=config_controller["log_base"],
            ref_period=config_controller["ref_period"],
            dead_ang=config_controller["dead_ang"],
            dead_pos=config_controller["dead_pos"],
            status_window_size=config_controller.get("status_window_size", 100),
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
        self.set_status_window_size(config_controller.get("status_window_size", self.status_window_size))

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
        self.total_decisions = 0
        self.skipped_decisions = 0
        self.update_decisions = 0
        self.last_did_update = False
        self.recent_decisions = deque(maxlen=max(1, int(self.status_window_size)))

    def set_status_window_size(self, status_window_size):
        status_window_size = max(1, int(status_window_size))
        if status_window_size == self.status_window_size:
            return

        self.status_window_size = status_window_size
        existing_decisions = list(getattr(self, "recent_decisions", []))[-self.status_window_size:]
        self.recent_decisions = deque(existing_decisions, maxlen=self.status_window_size)

    def time_difference(self, time=None):
        if self.time_last is None:
            return self.ref_period
        return time - self.time_last

    def should_sample(self, s, target_position, time=None, time_difference=None):
        if time_difference is None:
            time_difference = self.time_difference(time)

        if time_difference + self.ref_period / 20 < self.ref_period:
            self.record_decision(False)
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

        self.record_decision(spike)
        return spike

    def record_decision(self, did_update):
        self.last_did_update = bool(did_update)
        self.total_decisions += 1
        self.recent_decisions.append(self.last_did_update)
        if did_update:
            self.update_decisions += 1
        else:
            self.skipped_decisions += 1

    @property
    def skipped_update_percentage(self):
        if self.total_decisions == 0:
            return 0.0
        return 100.0 * self.skipped_decisions / self.total_decisions

    @property
    def recent_total_decisions(self):
        return len(self.recent_decisions)

    @property
    def recent_update_decisions(self):
        return sum(self.recent_decisions)

    @property
    def recent_skipped_decisions(self):
        return self.recent_total_decisions - self.recent_update_decisions

    @property
    def recent_skipped_update_percentage(self):
        if self.recent_total_decisions == 0:
            return 0.0
        return 100.0 * self.recent_skipped_decisions / self.recent_total_decisions

    def get_status(self):
        return (
            f"Secloc skipped {self.recent_skipped_update_percentage:.1f}% of controller updates "
            f"in the last {self.recent_total_decisions}/{self.status_window_size} decisions "
            f"({self.recent_skipped_decisions}/{self.recent_total_decisions}; "
            f"updates: {self.recent_update_decisions})"
        )

    def get_csv_data(self):
        return {
            "secloc_skipped_update": lambda: int(not self.last_did_update),
        }
