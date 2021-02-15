"""
This file contains methods related to displaying CartPole in GUI of the simulator.
One could think of moving these function outside of CartPole class and connecting them rather more tightly
with GUI of the simulator.
We leave them however as a part of CartPole class as they rely on variables of the CartPole.
"""

# Shapes used to draw a Cart and the slider
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
# rc sets global parameters for matlibplot; transforms is used to rotate the Mast
from matplotlib import transforms, rc
from matplotlib import animation

import matplotlib.pyplot as plt

from CartPole.cartpole_model import ANGLE_CONVENTION


# Set the font parameters for matplotlib figures
font = {'size': 22}
rc('font', **font)


# This method initializes CartPole elements to be plotted in CartPole GUI
def init_graphical_elements(self):

    self.CartLength = 10.0
    self.WheelRadius = 0.5
    self.WheelToMiddle = 4.0
    self.y_plane = 0.0
    self.y_wheel = self.y_plane + self.WheelRadius
    self.MastHight = 10.0  # For drawing only. For calculation see L
    self.MastThickness = 0.05
    self.HalfLength = 50.0  # Length of the track

    # Initialize elements of the drawing
    self.Mast = FancyBboxPatch(xy=(self.s.position - (self.MastThickness / 2.0), 1.25 * self.WheelRadius),
                               width=self.MastThickness,
                               height=self.MastHight,
                               fc='g')

    self.Chassis = FancyBboxPatch((self.s.position - (self.CartLength / 2.0), self.WheelRadius),
                                  self.CartLength,
                                  1 * self.WheelRadius,
                                  fc='r')

    self.WheelLeft = Circle((self.s.position - self.WheelToMiddle, self.y_wheel),
                            radius=self.WheelRadius,
                            fc='y',
                            ec='k',
                            lw=5)

    self.WheelRight = Circle((self.s.position + self.WheelToMiddle, self.y_wheel),
                             radius=self.WheelRadius,
                             fc='y',
                             ec='k',
                             lw=5)

    self.Slider = Rectangle((0.0, 0.0), self.slider_value, 1.0)
    self.t2 = transforms.Affine2D().rotate(0.0)  # An abstract container for the transform rotating the mast


# This method accepts the mouse position and updated the slider value accordingly
# The mouse position has to be captured by a function not included in this class
def update_slider(self, mouse_position):
    # The if statement formulates a saturation condition
    if mouse_position > self.slider_max:
        self.slider_value = self.slider_max
    elif mouse_position < -self.slider_max:
        self.slider_value = -self.slider_max
    else:
        self.slider_value = mouse_position


# This method draws elements and set properties of the CartPole figure
# which do not change at every frame of the animation
def draw_constant_elements(self, fig, AxCart, AxSlider):
    # Delete all elements of the Figure
    AxCart.clear()
    AxSlider.clear()

    ## Upper chart with Cart Picture
    # Set x and y limits
    AxCart.set_xlim((-self.HalfLength * 1.1, self.HalfLength * 1.1))
    AxCart.set_ylim((-1.0, 15.0))
    # Remove ticks on the y-axes
    AxCart.yaxis.set_major_locator(plt.NullLocator())  # NullLocator is used to disable ticks on the Figures

    # Draw track
    Floor = Rectangle((-self.HalfLength, -1.0),
                      2 * self.HalfLength,
                      1.0,
                      fc='brown')
    AxCart.add_patch(Floor)

    # Draw an invisible point at constant position
    # Thanks to it the axes is drawn high enough for the mast
    InvisiblePointUp = Rectangle((0, self.MastHight + 2.0),
                                 self.MastThickness,
                                 0.0001,
                                 fc='w',
                                 ec='w')

    AxCart.add_patch(InvisiblePointUp)
    # Apply scaling
    AxCart.axis('scaled')

    ## Lower Chart with Slider
    # Set y limits
    AxSlider.set(xlim=(-1.1 * self.slider_max, self.slider_max * 1.1))
    # Remove ticks on the y-axes
    AxSlider.yaxis.set_major_locator(plt.NullLocator())  # NullLocator is used to disable ticks on the Figures
    # Apply scaling
    AxSlider.set_aspect("auto")

    return fig, AxCart, AxSlider


# This method updates the elements of the Cart Figure which change at every frame.
# Not that these elements are not ploted directly by this method
# but rather returned as objects which can be used by another function
# e.g. animation function from matplotlib package
def update_drawing(self):

    # Draw mast
    mast_position = (self.s.position - (self.MastThickness / 2.0))
    self.Mast.set_x(mast_position)
    # Draw rotated mast
    t21 = transforms.Affine2D().translate(-mast_position, -1.25 * self.WheelRadius)
    if ANGLE_CONVENTION == 'CLOCK-NEG':
        t22 = transforms.Affine2D().rotate(self.s.angle)
    elif ANGLE_CONVENTION == 'CLOCK-POS':
        t22 = transforms.Affine2D().rotate(-self.s.angle)
    else:
        raise ValueError('Unknown angle convention')
    t23 = transforms.Affine2D().translate(mast_position, 1.25 * self.WheelRadius)
    self.t2 = t21 + t22 + t23
    # Draw Chassis
    self.Chassis.set_x(self.s.position - (self.CartLength / 2.0))
    # Draw Wheels
    self.WheelLeft.center = (self.s.position - self.WheelToMiddle, self.y_wheel)
    self.WheelRight.center = (self.s.position + self.WheelToMiddle, self.y_wheel)
    # Draw SLider
    self.Slider.set_width(self.slider_value)

    return self.Mast, self.t2, self.Chassis, self.WheelRight, self.WheelLeft, self.Slider


# A function redrawing the changing elements of the Figure
def run_animation(self, fig):
    def init():
        # Adding variable elements to the Figure
        fig.AxCart.add_patch(self.Mast)
        fig.AxCart.add_patch(self.Chassis)
        fig.AxCart.add_patch(self.WheelLeft)
        fig.AxCart.add_patch(self.WheelRight)
        fig.AxSlider.add_patch(self.Slider)
        return self.Mast, self.Chassis, self.WheelLeft, self.WheelRight, self.Slider

    def animationManage(i):
        # Updating variable elements
        self.update_drawing()
        # Special care has to be taken of the mast rotation
        self.t2 = self.t2 + fig.AxCart.transData
        self.Mast.set_transform(self.t2)
        return self.Mast, self.Chassis, self.WheelLeft, self.WheelRight, self.Slider

    # Initialize animation object
    anim = animation.FuncAnimation(fig, animationManage,
                                        init_func=init,
                                        frames=300,
                                        # fargs=(CartPoleInstance,), # It was used when this function was a part of GUI class. Now left as an example how to add arguments to FuncAnimation
                                        interval=10,
                                        blit=True,
                                        repeat=True)
    return anim
