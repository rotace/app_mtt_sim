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

    
    def test_ValueType(self):
        expected = [
            models.ValueType.POSIT_DIST,
            models.ValueType.POSIT_AZIM,
            models.ValueType.VELOC_DIST,
            models.ValueType.VELOC_AZIM,
        ]
        
        actual = models.ValueType.generate_value_type(
            crd_type=models.CoordType.POLAR,
            SD=2,
            RD=2,
            is_vel_measure_enabled=False
        )

        np.testing.assert_equal(actual, expected)


    def test_SignerModelFactory(self):
        """Singer Model Linear Kalman Filter

            ref) Design and Analysis of Modern Tracking Systems
                        4.2.1 Singer Acceleration Model
        """
        # simple check
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

        # check @ beta*dT -> 0
        sm = 7
        tm = 1.e+3
        dT = 2
        mf = models.SingerModelFactory(
            model = models.KalmanModel,
            tm = tm,
            sm = sm,
            SD = 1
        )

        md = mf.create(
            models.Obs(
                y=np.array([1.0]),
                R=np.zeros((1,1))+0.1,
                sensor=sensors.BaseSensor(dT=dT)
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
            [1, dT, dT**2/2],
            [0, 1., dT],
            [0, 0., 1]
        ]
        
        np.testing.assert_almost_equal(md.F, expected, decimal=2)

        expected = [
            [1, 0, 0]
        ]

        np.testing.assert_almost_equal(md.H, expected)

        expected = np.array([
            [dT**5/20, dT**4/8, dT**3/6],
            [dT**4/8,  dT**3/3, dT**2/2],
            [dT**3/6,  dT**2/2, dT]
        ])

        np.testing.assert_almost_equal(md.Q, expected*2*sm**2/tm, decimal=2)


    def test_Target(self):
        pass
        # target = models.SimpleTarget(
        #     x0=np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.]),
        #     x0_type=models.ModelType(
        #         crd_type = models.CoordType.CART,
        #         val_type = [
        #             models.ValueType.POSIT_D,
        #             models.ValueType.ACCEL_N,
        #             models.ValueType.POSIT_E,
        #             models.ValueType.VELOC_E,
        #             models.ValueType.POSIT_N,
        #             models.ValueType.VELOC_D,
        #             models.ValueType.ACCEL_E,
        #             models.ValueType.ACCEL_D,
        #             models.ValueType.VELOC_N,
        #         ]
        #     ),
        #     SD=3
        # )

        # expected = np.array([5., 3., 1., 9., 4., 6., 2., 7., 8.])
        # np.testing.assert_almost_equal(target.x, expected)

        # target.update(1.0)
        # expected = np.array([15., 10.5, 11., 11., 11., 14., 2., 7., 8.])
        # np.testing.assert_almost_equal(target.x, expected)


    def test_ModelEvaluator(self):

        eval = models.ModelEvaluator(
            models.SingerModelFactory(
                models.KalmanModel,
                tm=10.0,
                sm=15.0,
                SD=1,
                is_vel_measure_enabled=False
            ),
            models.SingerTarget(
                tm=10.0,
                sm=15.0,
                SD=1
            ),
            R=np.diag([150**2]),
            sensor=sensors.BaseSensor(dT=1)
        )

        RMSE = eval.estimate_prediction_error()
        np.testing.assert_almost_equal(RMSE.shape, (3,))

        eval = models.ModelEvaluator(
            models.SingerModelFactory(
                models.KalmanModel,
                tm=10.0,
                sm=15.0,
                SD=1,
                is_vel_measure_enabled=True
            ),
            models.SingerTarget(
                tm=10.0,
                sm=15.0,
                SD=1
            ),
            R=np.diag([150**2, 5.**2]),
            sensor=sensors.BaseSensor(dT=1)
        )

        RMSE = eval.estimate_prediction_error()
        np.testing.assert_almost_equal(RMSE.shape, (3,))


