"""Pure SecLoc gate decision logic.

C counterpart: Firmware/Src/General/secloc_logic.c (physical-cartpole repo);
the two are kept in lockstep by tests/test_secloc_c_python_parity.py, which
compiles the C file on the PC and asserts identical update/skip decisions.
Platform plumbing (config yaml, live reload, poll statistics) lives in
secloc_gate.py; keep this module free of such dependencies.
"""
import math

from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX


class SeclocLogic:
    def __init__(self, log_base, ref_period_ticks, dead_ang, dead_pos):
        self.log_base = log_base
        # Throttle in integer control loop iterations (ticks of the time
        # quantum): after an accepted update the gate is next consulted
        # ref_period_ticks iterations later. 0 and 1 both mean the gate is
        # consulted every iteration.
        self.ref_period_ticks = int(ref_period_ticks)
        self.dead_ang = dead_ang
        self.dead_pos = dead_pos
        # Time quantum (s) of the incoming timestamps (chip polling period or
        # simulation dt); maps timestamps to tick indices. Required whenever
        # ref_period_ticks > 0.
        self.time_quantum = None
        self.reset()

    def set_time_quantum(self, time_quantum_s):
        self.time_quantum = float(time_quantum_s) if time_quantum_s else None

    def _tick(self, time):
        return int(round(time / self.time_quantum))

    @classmethod
    def from_config(cls, config_controller):
        return cls(
            log_base=config_controller["log_base"],
            ref_period_ticks=config_controller["ref_period_ticks"],
            dead_ang=config_controller["dead_ang"],
            dead_pos=config_controller["dead_pos"],
        )

    def update_from_config(self, config_controller):
        self.log_base = config_controller["log_base"]
        self.ref_period_ticks = int(config_controller["ref_period_ticks"])
        self.dead_ang = config_controller["dead_ang"]
        self.dead_pos = config_controller["dead_pos"]

    def update_ref_period_from_config(self, config_controller):
        self.ref_period_ticks = int(config_controller["ref_period_ticks"])

    def reset(self):
        self.ang_last_shift = 0.0001
        self.pos_last_shift = 0.0001
        self.time_last = None
        self.tick_last = None

    def period_elapsed(self, time=None):
        """True when at least ref_period_ticks control loop iterations have
        passed since the last accepted update.

        Exact integer arithmetic on tick counts of the time quantum; a positive
        ref_period_ticks requires the quantum (and timestamps) to be set.
        """
        if self.ref_period_ticks <= 0:
            return True
        if self.time_quantum is None:
            raise ValueError(
                "Secloc ref_period_ticks > 0 requires a time quantum: call "
                "set_time_quantum() with the polling period / simulation dt."
            )
        if self.tick_last is None:
            return True
        if time is None:
            raise ValueError("Secloc ref_period_ticks > 0 requires timestamps (time=None)")
        return self._tick(time) - self.tick_last >= self.ref_period_ticks

    def should_sample(self, s, target_position, time=None, target_equilibrium=1.0):
        if not self.period_elapsed(time=time):
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

    def peek_should_sample(self, s, target_position, time=None, target_equilibrium=1.0):
        """Return whether the gate would update, without changing internal state."""
        if not self.period_elapsed(time=time):
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
