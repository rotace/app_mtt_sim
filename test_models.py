import copy
import unittest
import numpy as np
import matplotlib.pyplot as plt

import models

np.set_printoptions(suppress=True)

class TestModels(unittest.TestCase):

    def test_LinearKalmanModel(self):
        """Linear Kalman Filter

            ref) Design and Analysis of Modern Tracking Systems
                        3.3.3 Example of a Two-State Kalman Filter
        """

        PD=0.5
        sig_dv = 1
        sig_x0 = 10
        sig_vx0 = 5
        sig_o = 5
        T = 1

        x = np.array([0,0])
        F = np.array(
            [
                [1,T],
                [0,1]
            ]
        )
        H = np.array(
            [
                [1,0]
            ]
        )
        P = np.array(
            [
                [sig_x0**2, 0],
                [0, sig_vx0**2]
            ]
        )
        Q = np.array(
            [
                [0,0],
                [0,sig_dv**2]
            ]
        )

        model = models.KalmanModel(x,F,H,P,Q,False)
        mdl_list = []

        for y in np.random.normal(loc=0, scale=sig_o, size=36):
            
            if np.random.choice([True, False], p=[PD, 1-PD]):
                model.update(models.Obs(y, np.eye(1)*sig_o**2))
                mdl_list.append(copy.deepcopy(model))
            else:
                model.update(None)
                mdl_list.append(None)

        if False:
            plt.plot(
                [i for i in range(len(mdl_list))],
                [ mdl.K[0] if mdl is not None else None for mdl in mdl_list ],
                marker="D", color="g", alpha=.5, linestyle="None"
            )
            plt.plot(
                [i for i in range(len(mdl_list))],
                [ mdl.K[1] if mdl is not None else None for mdl in mdl_list ],
                marker="D", color="r", alpha=.5, linestyle="None"
            )
            plt.show()