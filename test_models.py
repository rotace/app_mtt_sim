import copy
import unittest
import numpy as np
import matplotlib.pyplot as plt

import models
import sensors

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


    def test_SignerModelFactory(self):
        """Singer Model Linear Kalman Filter

            ref) Design and Analysis of Modern Tracking Systems
                        4.2.1 Singer Acceleration Model
        """
        mf = models.SingerModelFactory(
            model = models.KalmanModel,
            tm = 1.0,
            sm = 1.0,
            SD = 1
        )

        md = mf.create(
            models.Obs(
                y=np.array([1.0]),
                R=np.zeros((1,1))+0.1,
                sensor=sensors.BaseSensor()
            )
        )
        np.testing.assert_equal(md.x.shape, (3,))
        np.testing.assert_equal(md.F.shape, (3,3))
        np.testing.assert_equal(md.H.shape, (1,3))
        np.testing.assert_equal(md.P.shape, (3,3))
        np.testing.assert_equal(md.Q.shape, (3,3))

        expected = [1.0, 0.0, 0.0]

        np.testing.assert_almost_equal(md.x, expected)

        expected = [
            [1, 1, np.exp(-1)],
            [0, 1, 1-np.exp(-1)],
            [0, 0, np.exp(-1)]
        ]
        
        np.testing.assert_almost_equal(md.F, expected)

        expected = [
            [1, 0, 0]
        ]

        np.testing.assert_almost_equal(md.H, expected)

        q11 = 1 - np.exp(-2) + 2 + 2/3 - 2 -4*np.exp(-1)
        q12 = np.exp(-2) + 1 -2*np.exp(-1) + 2*np.exp(-1) -2 + 1
        q13 = 1 - np.exp(-2) - 2*np.exp(-1)
        q22 = 4*np.exp(-1) -3 -np.exp(-2) + 2
        q23 = np.exp(-2) +1 -2*np.exp(-1)
        q33 = 1 - np.exp(-2)

        expected = [
            [q11, q12, q13],
            [q12, q22, q23],
            [q13, q23, q33]
        ]

        np.testing.assert_almost_equal(md.Q, expected)
        