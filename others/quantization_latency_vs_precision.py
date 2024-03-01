"""
This script plots
the precision of derivative measurement vs. the delay with which the information about der. is obtained.
While calculating derivatives the precision of the result is determined by the precision of the measured angle/position
AND! by the length of the time interval between the measurements.
smallest measurable derviative "bin" = precision of the angle or position measurement / time_interval
For cartpole basic setup with angle and positon measurement being around 4000 units per 360 degree or track length
This cause visible problems while controlling at 1kHz frequency.
For this reason we are calculating derivatives less frequently than control rate.
"""

import numpy as np
import plotly.graph_objects as go
from CartPole.cartpole_parameters import TrackHalfLength

# Initialize precision values
encoder_precision = 2*TrackHalfLength/4705.0
ADC_precision = (1.0 / 4096.0)

# Generate measurement intervals and calculate latencies
measurement_interval = np.linspace(0.001, 0.1, 1000)
latency = measurement_interval / 2

# Calculate precision for ADC and Encoder
precision_ADC = ADC_precision / measurement_interval
precision_encoder = encoder_precision / measurement_interval

# Encoder Precision Plot
fig_encoder = go.Figure()
fig_encoder.add_trace(go.Scatter(x=latency*1000, y=precision_encoder*100, mode='lines', name='Encoder'))
fig_encoder.update_layout(
    title='Encoder Precision',
    xaxis_title='Latency (ms)',
    yaxis_title='Precision (cm/s)',
    yaxis_type='log',  # Set y-axis to logarithmic
    xaxis_type='log',  # Set x-axis to logarithmic
    xaxis=dict(exponentformat='none'),
    yaxis=dict(exponentformat='none', showgrid=True),  # Disable scientific notation and show grid
)
fig_encoder.show()

# ADC Precision Plot
fig_adc = go.Figure()
fig_adc.add_trace(go.Scatter(x=latency*1000, y=precision_ADC*360, mode='lines', name='ADC'))
fig_adc.update_layout(
    title='ADC Precision',
    xaxis_title='Latency (ms)',
    yaxis_title='Precision (deg/s)',
    yaxis_type='log',  # Set y-axis to logarithmic
    xaxis_type='log',  # Set x-axis to logarithmic
    xaxis=dict(exponentformat='none'),
    yaxis=dict(exponentformat='none', showgrid=True),  # Disable scientific notation and show grid
)
fig_adc.show()

fig_encoder.write_html('encoder_precision.html')
fig_adc.write_html('adc_precision.html')
