import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from CartPole.cartpole_parameters import TrackHalfLength


class TargetSlider:
    def __init__(self):

        # Variable relevant for interactive use of slider
        self.slider_max = 1.0
        self.value = 0.0

        # Graphics
        # Depending on mode, slider may be displayed either as bar or as an arrow
        self.Slider_Bar = None
        self.Slider_Arrow = None
        self.init_graphical_elements()

    # This method accepts the mouse position and updated the slider value accordingly
    # The mouse position has to be captured by a function not included in this class
    def update_slider(self, mouse_position):
        # The if statement formulates a saturation condition

        if mouse_position > self.slider_max:
            self.value = self.slider_max
        elif mouse_position < -self.slider_max:
            self.value = -self.slider_max
        else:
            self.value = mouse_position

    def init_graphical_elements(self):
        self.Slider_Arrow = FancyArrowPatch((self.value, 0), (self.value, 0),
                                            arrowstyle='fancy', mutation_scale=50)
        self.Slider_Bar = Rectangle((0.0, 0.0), self.value, 1.0)

    def draw_constant_elements(self, AxSlider, controller_name):
        # Set y limits
        AxSlider.set(xlim=(-1.1 * self.slider_max, self.slider_max * 1.1))
        # Remove ticks on the y-axes
        AxSlider.yaxis.set_major_locator(plt.NullLocator())  # NullLocator is used to disable ticks on the Figures

        if controller_name == 'manual-stabilization':
            pass
        else:
            locs = np.array([-50.0, -37.5, -25.0, -12.5, - 0.0, 12.5, 25.0, 37.5, 50.0]) / 50.0
            labels = [str(np.around(np.array(x * TrackHalfLength), 3)) for x in locs]
            AxSlider.xaxis.set_major_locator(plt.FixedLocator(locs))
            AxSlider.xaxis.set_major_formatter(plt.FixedFormatter(labels))

        # Apply scaling
        AxSlider.set_aspect("auto")

    def update_drawing(self, controller_name):
        if controller_name == 'manual-stabilization':
            self.Slider_Bar.set_width(self.value)
        else:
            self.Slider_Arrow.set_positions((self.value, 0), (self.value, 1.0))
