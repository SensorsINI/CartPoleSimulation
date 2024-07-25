"""
This file runs Live Plotter (real time data visualization) in standalone mode or GUI mode.
It is the server (receiver) side of the live plotter.
The client side must be run by the program sending the data to be visualized.
"""

GUI = True

KEEP_SAMPLES_DEFAULT = 500  # at least 10
DEFAULT_ADDRESS = ('0.0.0.0', 6000)

if GUI:
    from SI_Toolkit.LivePlotter.live_plotter_GUI import run_live_plotter_gui
    run_live_plotter_gui(address=DEFAULT_ADDRESS, keep_samples=KEEP_SAMPLES_DEFAULT)

else:
    from SI_Toolkit.LivePlotter.live_plotter import LivePlotter
    plotter = LivePlotter(address=DEFAULT_ADDRESS, keep_samples=KEEP_SAMPLES_DEFAULT)
    plotter.run_standalone()
