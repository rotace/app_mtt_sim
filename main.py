""" MTT SIMULATION """

"""
    This program is made for learning, testing and comparing
    various algorithm of MTT.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import utils
import models
import tracks
import sensors
import trackers
from notebook import IRSTexample

def main():
    """
    Main Function
    """
    gnn = IRSTexample.generate_irst_example_p878(PD=0.7, PFA=1e-6)
    gnn.plot_position(n_scan=50, is_all_obs_displayed=True)
    # result = gnn.estimate_track_statistics(n_scan=10, n_run=10)
    # print(result["Tc"][0])

if __name__ == '__main__':
    if (sys.flags.interactive != 1):
        main()