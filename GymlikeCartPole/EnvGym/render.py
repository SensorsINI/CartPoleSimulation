# render.py

import numpy as np
from GymlikeCartPole.EnvGym.state_utils import POSITION_IDX, ANGLE_IDX


# ─── render.py ───────────────────────────────────────────────────────────────
class PygameViewer:
    """Stateless(ish) drawer so env logic never touches pygame directly."""
    def __init__(self, width: int, height: int, fps: int):
        import pygame
        from pygame import gfxdraw
        self.pg, self.gfx = pygame, gfxdraw

        pygame.init()
        self.clock   = pygame.time.Clock()
        self.canvas  = pygame.Surface((width, height))
        self.window  = pygame.display.set_mode((width, height))
        self.width   = width
        self.height  = height
        self.fps     = fps

    # --------------------------------------------------------------------- #
    # All geometry maths lives here; the env just passes state & physics.   #
    # --------------------------------------------------------------------- #
    def draw(self, state, physics, *, target_pos=None) -> np.ndarray:
        pg, gfx = self.pg, self.gfx
        self.canvas.fill((255, 255, 255))

        # ----- constants --------------------------------------------------
        pole_w   = 10.0
        cart_w   = 50.0
        cart_h   = 30.0
        axle_off = cart_h / 4.0
        scale    = self.width / (2 * physics.x_limit)
        pole_len = scale * physics.pole_length

        # ----- cart -------------------------------------------------------
        cx = state[POSITION_IDX] * scale + self.width / 2
        cy = 100
        l, r, t, b = -cart_w/2, cart_w/2,  cart_h/2, -cart_h/2
        cart = [(l, b), (l, t), (r, t), (r, b)]
        cart = [(x+cx, y+cy) for x, y in cart]
        gfx.filled_polygon(self.canvas, cart, (0, 0, 0))

        # ----- pole -------------------------------------------------------
        l, r, t, b = -pole_w/2, pole_w/2, pole_len - pole_w/2, -pole_w/2
        pole = [(l, b), (l, t), (r, t), (r, b)]
        pole = [pg.math.Vector2(p).rotate_rad(state[ANGLE_IDX])
                for p in pole]
        pole = [(x+cx, y+cy+axle_off) for x, y in pole]
        gfx.filled_polygon(self.canvas, pole, (202, 152, 101))

        # axle
        gfx.filled_circle(self.canvas, int(cx), int(cy+axle_off),
                          int(pole_w/2), (129, 132, 203))

        # target marker (optional)
        if target_pos is not None:
            tx = target_pos * scale + self.width / 2
            gfx.filled_circle(self.canvas, int(tx), int(cy), 10, (231, 76, 60))

        # floor line
        gfx.hline(self.canvas, 0, self.width, cy, (0, 0, 0))

        # pygame’s y-axis is top-down → flip for usual math coords
        self.canvas = pg.transform.flip(self.canvas, False, True)
        self.window.blit(self.canvas, (0, 0))
        pg.display.flip()
        self.clock.tick(self.fps)

        # for rgb_array mode
        return np.transpose(pg.surfarray.pixels3d(self.window), (1, 0, 2)).copy()

    def close(self):
        """
        Tear down all pygame subsystems associated with this viewer.
        Idempotent: safe to call even if already closed.
        """
        # Shut down the display module, then quit pygame entirely.
        self.pg.display.quit()
        self.pg.quit()
