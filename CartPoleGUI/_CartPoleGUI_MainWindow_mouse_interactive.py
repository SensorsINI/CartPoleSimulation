"""
These are some methods GUI uses to capture mouse effect while hoovering or clicking over/on the charts
"""

# Function evoked at a mouse movement
# If the mouse cursor is over the lower chart it reads the corresponding value
# and updates the slider
def on_mouse_movement(self, event):
    if event.xdata == None or event.ydata == None:
        pass
    else:
        if event.inaxes == self.fig.AxSlider:
            self.slider_value = event.xdata
            if not self.slider_on_click:
                self.CartPoleInstance.update_slider(mouse_position=event.xdata)

# Function evoked at a mouse click
# If the mouse cursor is over the lower chart it reads the corresponding value
# and updates the slider
def on_mouse_click(self, event):
    if event.xdata == None or event.ydata == None:
        pass
    else:
        if event.inaxes == self.fig.AxSlider:
            self.CartPoleInstance.update_slider(mouse_position=event.xdata)