import numpy as np

# region Graphics imports
import matplotlib.pyplot as plt
# rc sets global parameters for matplotlib; transforms is used to rotate the Mast
from matplotlib import animation, rc, transforms
# Shapes used to draw a Cart
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)

from CartPole.cartpole_equations import ANGLE_CONVENTION
from CartPole.cartpole_parameters import TrackHalfLength
from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX

from CartPole.cartpole_target_slider import TargetSlider

# Set the font parameters for matplotlib figures
font = {'size': 22}
rc('font', **font)


class CartPoleDrawer:

    def __init__(self, simulator, target_slider):
        
        self.cp = simulator

        # region Variables initialization for drawing/animating a CartPole
        # DIMENSIONS OF THE DRAWING ONLY!!!
        # NOTHING TO DO WITH THE SIMULATION AND NOT INTENDED TO BE MANIPULATED BY USER !!!

        self.physical_to_graphics = None
        self.graphics_to_physical = None

        # Parameters needed to display CartPole in GUI
        # They are assigned with values in self.init_elements()
        self.CartLength = None
        self.WheelRadius = None
        self.WheelToMiddle = None
        self.y_plane = None
        self.y_wheel = None
        self.mast_height_maximal_drawing_units = None  # For drawing only. For calculation see L
        self.max_height_maximal_physical_units = None
        self.mast_height_current_drawing_units = None
        self.zero_angle_tick_height_current_drawing_units = None
        self.MastThickness = None
        self.ZeroAngleTickThickness = None
        self.TrackHalfLengthGraphics = None  # Length of the track

        # Elements of the drawing
        self.Mast = None
        self.Chassis = None
        self.WheelLeft = None
        self.WheelRight = None

        self.ZeroAngleTick = None

        # Arrow indicating acceleration (=motor power)
        self.Acceleration_Arrow = None

        self.y_acceleration_arrow = None
        self.scaling_dx_acceleration_arrow = None
        self.x_acceleration_arrow = None

        self.t2 = None  # An abstract container for the transform rotating the mast
        self.t_zero_angle = None  # An abstract container for the transform rotating the zero angle tick

        self.slider = target_slider

        self.init_graphical_elements(self.cp.L_updater.init_value, self.cp.L_updater.range_random)  # Assign proper object to the above variables
        # endregion


    # region 5. Methods needed to display CartPole in GUI
    """
    This section contains methods related to displaying CartPole in GUI of the simulator.
    One could think of moving these function outside of CartPole class and connecting them rather more tightly
    with GUI of the simulator.
    We leave them however as a part of CartPole class as they rely on variables of the CartPole.
    """

    # This method initializes CartPole elements to be plotted in CartPole GUI
    def init_graphical_elements(self, L_initial, L_range):

        L = self.cp.current_L()

        self.CartLength = 10.0
        self.WheelRadius = 0.5
        self.WheelToMiddle = 4.0
        self.y_plane = 0.0
        self.y_wheel = self.y_plane + self.WheelRadius

        self.mast_height_maximal_drawing_units = 10.0
        self.max_height_maximal_physical_units = np.max([L_initial, *L_range])
        self.mast_height_current_drawing_units = self.mast_height_maximal_drawing_units * (
                    float(L) / self.max_height_maximal_physical_units)

        self.zero_angle_tick_height_current_drawing_units = 1.0

        self.MastThickness = 0.05
        self.ZeroAngleTickThickness = 0.01
        self.TrackHalfLengthGraphics = 50.0  # Full Length of the track

        self.physical_to_graphics = (
                                                self.TrackHalfLengthGraphics - self.WheelToMiddle) / TrackHalfLength  # TrackHalfLength is the effective length of track
        self.graphics_to_physical = 1.0 / self.physical_to_graphics

        self.y_acceleration_arrow = 1.5 * self.WheelRadius
        self.scaling_dx_acceleration_arrow = 20.0
        self.x_acceleration_arrow = (
                self.cp.s[POSITION_IDX] * self.physical_to_graphics +
                # np.sign(self.cp.Q) * (self.CartLength / 2.0) +
                self.scaling_dx_acceleration_arrow * self.cp.Q
        )

        self.slider.init_graphical_elements()

        # Initialize elements of the drawing
        self.Mast = FancyBboxPatch(
            xy=(self.cp.s[POSITION_IDX] * self.physical_to_graphics - (self.MastThickness / 2.0), 1.25 * self.WheelRadius),
            width=self.MastThickness,
            height=self.mast_height_current_drawing_units,
            fc='g')

        self.ZeroAngleTick = FancyBboxPatch(xy=(
        self.cp.s[POSITION_IDX] * self.physical_to_graphics - (self.ZeroAngleTickThickness / 2.0),
        1.2 * self.mast_height_current_drawing_units),
                                            width=self.ZeroAngleTickThickness,
                                            height=self.zero_angle_tick_height_current_drawing_units,
                                            fc='yellow')

        self.Chassis = FancyBboxPatch(
            (self.cp.s[POSITION_IDX] * self.physical_to_graphics - (self.CartLength / 2.0), self.WheelRadius),
            self.CartLength,
            1 * self.WheelRadius,
            fc='r')

        self.WheelLeft = Circle((self.cp.s[POSITION_IDX] * self.physical_to_graphics - self.WheelToMiddle, self.y_wheel),
                                radius=self.WheelRadius,
                                fc='y',
                                ec='k',
                                lw=5)

        self.WheelRight = Circle((self.cp.s[POSITION_IDX] * self.physical_to_graphics + self.WheelToMiddle, self.y_wheel),
                                 radius=self.WheelRadius,
                                 fc='y',
                                 ec='k',
                                 lw=5)

        self.Acceleration_Arrow = FancyArrowPatch(
            (self.cp.s[POSITION_IDX] * self.physical_to_graphics, self.y_acceleration_arrow),
            (self.x_acceleration_arrow, self.y_acceleration_arrow),
            arrowstyle='simple', mutation_scale=10,
            facecolor='gold', edgecolor='orange')

        self.t2 = transforms.Affine2D().rotate(0.0)  # An abstract container for the transform rotating the mast
        self.t_zero_angle = transforms.Affine2D().rotate(
            0.0)  # An abstract container for the transform rotating the mast

    # This method draws elements and set properties of the CartPole figure
    # which do not change at every frame of the animation
    def draw_constant_elements(self, fig, AxCart, AxSlider):

        ## Upper chart with Cart Picture
        # Set x and y limits
        AxCart.set_xlim((-self.TrackHalfLengthGraphics * 1.1, self.TrackHalfLengthGraphics * 1.1))

        # Remove ticks on the y-axes
        AxCart.yaxis.set_major_locator(plt.NullLocator())  # NullLocator is used to disable ticks on the Figures

        locs = [-50.0, -25.0, - 0.0, 25.0, 50.0]
        labels = [str(np.around(np.array(x * self.graphics_to_physical), 3)) for x in locs]
        AxCart.xaxis.set_major_locator(plt.FixedLocator(locs))
        AxCart.xaxis.set_major_formatter(plt.FixedFormatter(labels))

        # Draw track
        Floor = Rectangle((-self.TrackHalfLengthGraphics, -1.0),
                          2 * self.TrackHalfLengthGraphics,
                          1.0,
                          fc='brown')
        AxCart.add_patch(Floor)

        # Draw an invisible point at constant position
        # Thanks to it the axes is drawn high enough for the mast
        InvisiblePointUp = Rectangle((0, self.mast_height_maximal_drawing_units + 2.0),
                                     self.MastThickness,
                                     0.0001,
                                     fc='w',
                                     ec='w')

        AxCart.add_patch(InvisiblePointUp)

        point_y_coordinate = -self.mast_height_maximal_drawing_units - 1.0

        InvisiblePointDown = Rectangle((0, point_y_coordinate),
                                       self.MastThickness,
                                       0.0001,
                                       fc='w',
                                       ec='w')

        AxCart.add_patch(InvisiblePointDown)

        # Apply scaling
        AxCart.axis('scaled')

        ## Lower Chart with Slider
        self.slider.draw_constant_elements(AxSlider, self.cp.controller_name)

        return fig, AxCart, AxSlider

    # This method updates the elements of the Cart Figure which change at every frame.
    # Not that these elements are not ploted directly by this method
    # but rather returned as objects which can be used by another function
    # e.g. animation function from matplotlib package
    def update_drawing(self):

        L = self.cp.current_L()

        self.x_acceleration_arrow = (
                self.cp.s[POSITION_IDX] * self.physical_to_graphics +
                # np.sign(self.cp.Q) * (self.CartLength / 2.0) +
                self.scaling_dx_acceleration_arrow * self.cp.Q
        )

        self.Acceleration_Arrow.set_positions(
            (self.cp.s[POSITION_IDX] * self.physical_to_graphics, self.y_acceleration_arrow),
            (self.x_acceleration_arrow, self.y_acceleration_arrow))

        # Draw mast
        mast_position = (self.cp.s[POSITION_IDX] * self.physical_to_graphics - (self.MastThickness / 2.0))
        self.Mast.set_x(mast_position)
        self.Mast.set_height(
            self.mast_height_maximal_drawing_units * (float(L) / self.max_height_maximal_physical_units))

        zero_tick_position = (self.cp.s[POSITION_IDX] * self.physical_to_graphics - (self.ZeroAngleTickThickness / 2.0))
        self.ZeroAngleTick.set_x(zero_tick_position)

        # Draw rotated mast
        t21 = transforms.Affine2D().translate(-mast_position, -1.25 * self.WheelRadius)
        if ANGLE_CONVENTION == 'CLOCK-NEG':
            t22 = transforms.Affine2D().rotate(self.cp.s[ANGLE_IDX])
        elif ANGLE_CONVENTION == 'CLOCK-POS':
            t22 = transforms.Affine2D().rotate(-self.cp.s[ANGLE_IDX])
        else:
            raise ValueError('Unknown angle convention')
        t23 = transforms.Affine2D().translate(mast_position, 1.25 * self.WheelRadius)
        self.t2 = t21 + t22 + t23

        t21 = transforms.Affine2D().translate(-zero_tick_position, 0.0)
        if ANGLE_CONVENTION == 'CLOCK-NEG':
            t22 = transforms.Affine2D().rotate(-self.cp.vertical_angle_offset)
        elif ANGLE_CONVENTION == 'CLOCK-POS':
            t22 = transforms.Affine2D().rotate(self.cp.vertical_angle_offset)
        else:
            raise ValueError('Unknown angle convention')
        t23 = transforms.Affine2D().translate(zero_tick_position, 0.0)
        self.t_zero_angle = t21 + t22 + t23

        # Draw Chassis
        self.Chassis.set_x(self.cp.s[POSITION_IDX] * self.physical_to_graphics - (self.CartLength / 2.0))
        # Draw Wheels
        self.WheelLeft.center = (self.cp.s[POSITION_IDX] * self.physical_to_graphics - self.WheelToMiddle, self.y_wheel)
        self.WheelRight.center = (self.cp.s[POSITION_IDX] * self.physical_to_graphics + self.WheelToMiddle, self.y_wheel)
        # Draw SLider
        self.slider.update_drawing(self.cp.controller_name)

    # A function redrawing the changing elements of the Figure
    def run_animation(self, fig):
        def init():
            # Adding variable elements to the Figure
            fig.AxCart.add_patch(self.Mast)
            fig.AxCart.add_patch(self.Chassis)
            fig.AxCart.add_patch(self.WheelLeft)
            fig.AxCart.add_patch(self.WheelRight)
            fig.AxCart.add_patch(self.Acceleration_Arrow)
            fig.AxCart.add_patch(self.ZeroAngleTick)
            fig.AxSlider.add_patch(self.slider.Slider_Bar)
            fig.AxSlider.add_patch(self.slider.Slider_Arrow)
            return self.Mast, self.Chassis, self.WheelLeft, self.WheelRight, \
                self.slider.Slider_Bar, self.slider.Slider_Arrow, self.Acceleration_Arrow, self.ZeroAngleTick

        def animationManage(i):
            # Updating variable elements
            self.update_drawing()
            # Special care has to be taken of the mast rotation
            self.t2 = self.t2 + fig.AxCart.transData
            self.t_zero_angle = self.t_zero_angle + fig.AxCart.transData
            self.Mast.set_transform(self.t2)
            self.ZeroAngleTick.set_transform(self.t_zero_angle)
            return self.Mast, self.Chassis, self.WheelLeft, self.WheelRight, \
                self.slider.Slider_Bar, self.slider.Slider_Arrow, self.Acceleration_Arrow, self.ZeroAngleTick

        # Initialize animation object
        anim = animation.FuncAnimation(fig, animationManage,
                                       init_func=init,
                                       frames=300,
                                       # fargs=(CartPoleInstance,), # It was used when this function was a part of GUI class. Now left as an example how to add arguments to FuncAnimation
                                       interval=10,
                                       blit=True,
                                       repeat=True)
        return anim
