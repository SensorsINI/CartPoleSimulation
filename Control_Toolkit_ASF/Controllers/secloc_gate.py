from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX


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

    def reset(self):
        self.ang_last_shift = 0.0001
        self.pos_last_shift = 0.0001
        self.time_last = None

    def should_sample(self, s, target_position, time=None, time_difference=None):
        if time_difference is None:
            if self.time_last is None:
                time_difference = self.ref_period
            else:
                time_difference = time - self.time_last

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
