from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class RollingSuccessCountCallback(BaseCallback):
    """
    After each episode, logs:
      - roll/episode                         : total number of episodes so far
      - roll/last_N_successes_by_episode     : successes in last N episodes (indexed by episode count)
      - roll/last_N_successes_by_timestep    : same successes (indexed by total timesteps)
      - roll/total_timesteps                 : cumulative number of environment steps so far
    """

    def __init__(self, n_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.n_episodes    = n_episodes
        # deque automatically drops oldest entries once full
        self._window       = deque(maxlen=n_episodes)
        # count of completed episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Called at every environment step. We only act when an env reports done=True.
        """
        # ─── detect which sub-envs just ended ────────────────────────────────────
        # `dones` is a list/array of bools for each parallel env
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for idx, done in enumerate(dones):
            if not done:
                continue  # nothing to do until this env finishes an episode

            self.episode_count += 1

            info = infos[idx]
            # ⮕ Gymnasium will set this when the episode ends due to hitting the time limit
            truncated = info.get("TimeLimit.truncated", False)
            # ⮕ we treat a truncation by timeout as a “success”
            success = int(truncated)

            self._window.append(success)

            # ─── rolling sum over last N episodes ───────────────────────────────
            count = sum(self._window)
            print(f"[Callback] Last {self.n_episodes} successes: {count}")

            # ─── log under the episode-indexed tag ───────────────────────────────
            self.logger.record("roll/last_N_successes_by_episode", count)
            # use episode_count as the step for this series
            self.logger.dump(self.episode_count)

            # ─── log under the timestep-indexed tag ──────────────────────────────
            self.logger.record("roll/last_N_successes_by_timestep", count)
            self.logger.record("roll/total_timesteps", self.num_timesteps)
            # use num_timesteps as the step for this series
            self.logger.dump(self.num_timesteps)

            # (Optional) also log the raw episode index
            self.logger.record("roll/episode", self.episode_count)
            # it will show up on the next dump

        return True
