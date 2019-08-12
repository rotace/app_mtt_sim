""" MTT SIMULATION """

"""
    This program is made for learning, testing and comparing
    various algorithm of MTT.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


targets = [
    np.array([0.0,  0.0, 1.0, 1.0]),
    np.array([0.0,10.0, 1.0, -1.0]),
]

for k in range(25):
    if k > 0:
        
        for t in targets:
            t[0:2] += t[2:]

    if k == 5:
        targets.append(np.array([5.0, 5.0, 1.0, 0.0]))

    plt.plot(
        [t[0] for t in targets],
        [t[1] for t in targets],
        marker="D", color="y", alpha=.5, linestyle="None"
    )

plt.show()