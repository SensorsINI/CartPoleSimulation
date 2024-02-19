"""
This script starts CartPole GUI from main folder
"""

# import os
# print('GPU Disabled')
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TF: If uncommented, only uses CPU
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from GUI import run_gui

run_gui()