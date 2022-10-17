from Control_Toolkit.others.environment import EnvironmentBatched


class cost_function_default:
    """Uses as cost function the get_reward method of environment provided."""

    def __init__(self, env) -> None:
        self.env: EnvironmentBatched = env

    def get_terminal_cost(self, s_hor):
        return 0.0

    def get_stage_cost(self, s, u, u_prev):
        return -self.env.get_reward(s, u)

    def get_trajectory_cost(self, s_hor, u, u_prev=None):
        return self.env.lib.sum(
            self.get_stage_cost(s_hor[:, :-1, :], u, None), 1
        ) + self.get_terminal_cost(s_hor)
