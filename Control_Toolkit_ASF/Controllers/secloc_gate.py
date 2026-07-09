import math
import os
import atexit
from collections import deque, namedtuple
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX
from SI_Toolkit.load_and_normalize import load_yaml


PollStat = namedtuple("PollStat", ("time", "ang_unchanged", "pos_unchanged", "skipped"))


SECLOC_CONFIG_PATH = os.path.join("Control_Toolkit_ASF", "config_secloc.yml")


class SeclocConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, secloc):
        self.secloc = secloc
        self.config_path = os.path.abspath(secloc.config_path)

    def on_modified(self, event):
        if os.path.abspath(event.src_path) == self.config_path:
            self.secloc.reload_config_from_file_flag = True


class SeclocLogic:
    def __init__(self, log_base, ref_period, dead_ang, dead_pos):
        self.log_base = log_base
        self.ref_period = ref_period
        self.dead_ang = dead_ang
        self.dead_pos = dead_pos
        # Time quantum (s) of the incoming timestamps (chip polling period or
        # simulation dt). When set, the ref_period check runs in integer ticks
        # instead of comparing float seconds with a tolerance.
        self.time_quantum = None
        self.reset()

    def set_time_quantum(self, time_quantum_s):
        self.time_quantum = float(time_quantum_s) if time_quantum_s else None

    @property
    def ref_period_ticks(self):
        return max(1, int(round(self.ref_period / self.time_quantum)))

    def _tick(self, time):
        return int(round(time / self.time_quantum))

    @classmethod
    def from_config(cls, config_controller):
        return cls(
            log_base=config_controller["log_base"],
            ref_period=config_controller["ref_period"],
            dead_ang=config_controller["dead_ang"],
            dead_pos=config_controller["dead_pos"],
        )

    def update_from_config(self, config_controller):
        self.log_base = config_controller["log_base"]
        self.ref_period = config_controller["ref_period"]
        self.dead_ang = config_controller["dead_ang"]
        self.dead_pos = config_controller["dead_pos"]

    def update_ref_period_from_config(self, config_controller):
        self.ref_period = config_controller["ref_period"]

    def reset(self):
        self.ang_last_shift = 0.0001
        self.pos_last_shift = 0.0001
        self.time_last = None
        self.tick_last = None

    def time_difference(self, time=None):
        if self.time_last is None:
            return self.ref_period
        return time - self.time_last

    def period_elapsed(self, time=None, time_difference=None):
        """True when at least ref_period has passed since the last accepted update.

        With a time quantum set, this is exact integer arithmetic on tick counts;
        otherwise the legacy float comparison with a ref_period/20 tolerance.
        """
        if self.time_quantum is not None and time is not None:
            if self.tick_last is None:
                return True
            return self._tick(time) - self.tick_last >= self.ref_period_ticks

        if time_difference is None:
            time_difference = self.time_difference(time)
        return time_difference + self.ref_period / 20 >= self.ref_period

    def should_sample(self, s, target_position, time=None, time_difference=None,
                      target_equilibrium=1.0):
        if not self.period_elapsed(time=time, time_difference=time_difference):
            return False

        ang_spike, pos_spike, ang_shift, pos_shift = self._evaluate_spike(
            s, target_position, target_equilibrium
        )
        spike = ang_spike or pos_spike
        if spike:
            if ang_spike:
                self.ang_last_shift = ang_shift
            if pos_spike:
                self.pos_last_shift = pos_shift
            self.time_last = time
            if self.time_quantum is not None and time is not None:
                self.tick_last = self._tick(time)
        return spike

    def peek_should_sample(self, s, target_position, time=None, time_difference=None,
                           target_equilibrium=1.0):
        """Return whether the gate would update, without changing internal state."""
        if not self.period_elapsed(time=time, time_difference=time_difference):
            return False

        ang_spike, pos_spike, _, _ = self._evaluate_spike(
            s, target_position, target_equilibrium
        )
        return ang_spike or pos_spike

    @staticmethod
    def angle_shift_from_target(angle, target_equilibrium):
        """Angle distance from the active target equilibrium.

        Target up (equilibrium > 0): distance from upright, |angle|.
        Target down: distance from hanging, pi - |angle| (angle wraps at +/-pi,
        so hanging is angle = +/-pi and this is the wrap-aware distance).
        Without this, the multiplicative log_base criterion is evaluated around
        |angle| ~ pi during down phases, where a factor of e.g. 1.05 needs a
        ~0.16 rad excursion before an update fires - the gate goes nearly blind
        exactly at the equilibrium it is supposed to track.
        """
        shift = abs(angle)
        if target_equilibrium < 0:
            shift = math.pi - shift
        return shift

    def _evaluate_spike(self, s, target_position, target_equilibrium=1.0):
        """Independent per-axis event checks.

        Angle and position are evaluated independently (an update fires when
        either exceeds the log_base ratio); on update, only the reference of
        the axis (or axes) that fired is refreshed.
        """
        ang_shift = self.angle_shift_from_target(s[ANGLE_IDX], target_equilibrium)
        pos_shift = s[POSITION_IDX] - target_position
        if pos_shift < 0:
            pos_shift = -pos_shift

        ang_spike = False
        if (ang_shift > self.dead_ang) and (self.ang_last_shift != 0):
            ang_ratio_inc = ang_shift / self.ang_last_shift
            ang_ratio_dec = 1.0 / ang_ratio_inc
            ang_spike = bool(
                (ang_ratio_inc >= self.log_base) or (ang_ratio_dec >= self.log_base)
            )

        pos_spike = False
        if (pos_shift > self.dead_pos) and (self.pos_last_shift != 0):
            pos_ratio_inc = pos_shift / self.pos_last_shift
            pos_ratio_dec = 1.0 / pos_ratio_inc
            pos_spike = bool(
                (pos_ratio_inc >= self.log_base) or (pos_ratio_dec >= self.log_base)
            )

        return ang_spike, pos_spike, ang_shift, pos_shift


class SeclocGate:
    def __init__(
        self,
        log_base,
        ref_period,
        dead_ang,
        dead_pos,
        status_window_size=100,
        poll_stats_window_s=5.0,
    ):
        self.logic = SeclocLogic(log_base, ref_period, dead_ang, dead_pos)
        self.status_window_size = status_window_size
        self.poll_stats_window_s = float(poll_stats_window_s)
        self.reset_statistics()

    @classmethod
    def from_config(cls, config_controller):
        return cls(
            log_base=config_controller["log_base"],
            ref_period=config_controller["ref_period"],
            dead_ang=config_controller["dead_ang"],
            dead_pos=config_controller["dead_pos"],
            status_window_size=config_controller.get("status_window_size", 100),
            poll_stats_window_s=config_controller.get("poll_stats_window_s", 5.0),
        )

    @classmethod
    def from_config_file(cls, config_name="default"):
        config_secloc, _ = load_yaml(SECLOC_CONFIG_PATH, return_path=True)
        return cls.from_config(dict(config_secloc[config_name]))

    @property
    def log_base(self):
        return self.logic.log_base

    @log_base.setter
    def log_base(self, value):
        self.logic.log_base = value

    @property
    def ref_period(self):
        return self.logic.ref_period

    @ref_period.setter
    def ref_period(self, value):
        self.logic.ref_period = value

    @property
    def dead_ang(self):
        return self.logic.dead_ang

    @dead_ang.setter
    def dead_ang(self, value):
        self.logic.dead_ang = value

    @property
    def dead_pos(self):
        return self.logic.dead_pos

    @dead_pos.setter
    def dead_pos(self, value):
        self.logic.dead_pos = value

    @property
    def ang_last_shift(self):
        return self.logic.ang_last_shift

    @ang_last_shift.setter
    def ang_last_shift(self, value):
        self.logic.ang_last_shift = value

    @property
    def pos_last_shift(self):
        return self.logic.pos_last_shift

    @pos_last_shift.setter
    def pos_last_shift(self, value):
        self.logic.pos_last_shift = value

    @property
    def time_last(self):
        return self.logic.time_last

    @time_last.setter
    def time_last(self, value):
        self.logic.time_last = value

    @property
    def time_quantum(self):
        return self.logic.time_quantum

    def set_time_quantum(self, time_quantum_s):
        """Timestamp granularity (chip polling period / sim dt); enables exact
        tick-based ref_period decisions instead of float-seconds comparisons."""
        self.logic.set_time_quantum(time_quantum_s)

    def update_from_config(self, config_controller):
        self.logic.update_from_config(config_controller)
        self.set_status_window_size(config_controller.get("status_window_size", self.status_window_size))
        self.set_poll_stats_window_s(
            config_controller.get("poll_stats_window_s", self.poll_stats_window_s)
        )

    def update_ref_period_from_config(self, config_controller):
        self.logic.update_ref_period_from_config(config_controller)

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
        self.logic.reset()
        self.reset_statistics()

    def reset_statistics(self):
        self.total_decisions = 0
        self.skipped_decisions = 0
        self.update_decisions = 0
        self.last_did_update = False
        self.recent_decisions = deque(maxlen=max(1, int(self.status_window_size)))
        self.recent_poll_stats = deque()
        self._prev_poll_ang_shift = None
        self._prev_poll_pos_shift = None
        self._last_poll_stat_time = None
        self.last_gate_evaluated = False
        self.last_gate_would_update = False

    def set_status_window_size(self, status_window_size):
        status_window_size = max(1, int(status_window_size))
        if status_window_size == self.status_window_size:
            return

        self.status_window_size = status_window_size
        existing_decisions = list(getattr(self, "recent_decisions", []))[-self.status_window_size:]
        self.recent_decisions = deque(existing_decisions, maxlen=self.status_window_size)

    def set_poll_stats_window_s(self, poll_stats_window_s):
        poll_stats_window_s = max(0.1, float(poll_stats_window_s))
        if poll_stats_window_s == self.poll_stats_window_s:
            return

        self.poll_stats_window_s = poll_stats_window_s
        if self._last_poll_stat_time is not None:
            self._prune_poll_stats(self._last_poll_stat_time)

    @staticmethod
    def _poll_shifts(s, target_position, target_equilibrium=1.0):
        ang_shift = SeclocLogic.angle_shift_from_target(s[ANGLE_IDX], target_equilibrium)
        pos_shift = abs(s[POSITION_IDX] - target_position)
        return ang_shift, pos_shift

    def _gate_evaluated(self, time, time_difference):
        if time is None and time_difference is None:
            return True
        return self.logic.period_elapsed(time=time, time_difference=time_difference)

    def _record_poll_stat(self, ang_unchanged, pos_unchanged, skipped, time):
        poll_time = 0.0 if time is None else float(time)
        self._last_poll_stat_time = poll_time
        self.recent_poll_stats.append(
            PollStat(
                time=poll_time,
                ang_unchanged=ang_unchanged,
                pos_unchanged=pos_unchanged,
                skipped=skipped,
            )
        )
        self._prune_poll_stats(poll_time)

    def _prune_poll_stats(self, current_time):
        cutoff = current_time - self.poll_stats_window_s
        while self.recent_poll_stats and self.recent_poll_stats[0].time < cutoff:
            self.recent_poll_stats.popleft()

    def _active_poll_stats(self):
        if self._last_poll_stat_time is None:
            return []
        cutoff = self._last_poll_stat_time - self.poll_stats_window_s
        return [stat for stat in self.recent_poll_stats if stat.time >= cutoff]

    def _poll_stat_percentage(self, predicate):
        active_stats = self._active_poll_stats()
        if not active_stats:
            return 0.0
        return 100.0 * sum(predicate(stat) for stat in active_stats) / len(active_stats)

    def time_difference(self, time=None):
        return self.logic.time_difference(time)

    def peek_would_update(self, s, target_position, time=None, time_difference=None,
                          target_equilibrium=1.0):
        """Evaluate the gate without recording stats or mutating gate state."""
        if time_difference is None:
            time_difference = self.time_difference(time)

        gate_evaluated = self._gate_evaluated(time, time_difference)
        would_update = False
        if gate_evaluated:
            would_update = self.logic.peek_should_sample(
                s,
                target_position,
                time=time,
                time_difference=time_difference,
                target_equilibrium=target_equilibrium,
            )
        self.last_gate_evaluated = gate_evaluated
        self.last_gate_would_update = would_update
        return would_update

    def should_sample(self, s, target_position, time=None, time_difference=None,
                      target_equilibrium=1.0):
        if time_difference is None:
            time_difference = self.time_difference(time)

        ang_shift, pos_shift = self._poll_shifts(s, target_position, target_equilibrium)
        gate_evaluated = self._gate_evaluated(time, time_difference)

        spike = self.logic.should_sample(
            s,
            target_position,
            time=time,
            time_difference=time_difference,
            target_equilibrium=target_equilibrium,
        )

        self.last_gate_evaluated = gate_evaluated
        self.last_gate_would_update = spike

        if gate_evaluated:
            if self._prev_poll_ang_shift is not None:
                self._record_poll_stat(
                    ang_unchanged=ang_shift == self._prev_poll_ang_shift,
                    pos_unchanged=pos_shift == self._prev_poll_pos_shift,
                    skipped=not spike,
                    time=time,
                )
            self._prev_poll_ang_shift = ang_shift
            self._prev_poll_pos_shift = pos_shift

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
        active_stats = self._active_poll_stats()
        poll_count = len(active_stats)
        ang_flat_pct = self._poll_stat_percentage(lambda stat: stat.ang_unchanged)
        pos_flat_pct = self._poll_stat_percentage(lambda stat: stat.pos_unchanged)
        skip_or_changed_pct = self._poll_stat_percentage(
            lambda stat: (not stat.ang_unchanged or not stat.pos_unchanged) and stat.skipped
        )
        skip_and_changed_pct = self._poll_stat_percentage(
            lambda stat: (not stat.ang_unchanged and not stat.pos_unchanged) and stat.skipped
        )
        return (
            f"Secloc: angle not changed {ang_flat_pct:.1f}% | "
            f"position not changed {pos_flat_pct:.1f}% | "
            f"skipped (angle or position changed) {skip_or_changed_pct:.1f}% | "
            f"skipped (angle and position changed) {skip_and_changed_pct:.1f}% "
            f"[{poll_count} gate polls, {self.poll_stats_window_s:.0f}s window]"
        )

    def get_csv_data(self):
        return {
            "secloc_skipped_update": lambda: int(not self.last_did_update),
            "secloc_gate_skipped": lambda: int(
                self.last_gate_evaluated and not self.last_gate_would_update
            ),
        }
