import unittest
import numpy as np

import models
import tracks
import sensors
import trackers

np.set_printoptions(suppress=True)

class TestTracker(unittest.TestCase):

    def test_JPDA(self):
        """Calculate Association of Observations by JPDA Method

            ref) Design and Analysis of Modern Tracking Systems
                        6.6.2 Extension to JPDA
        """

        tracker = trackers.JPDA(
            sensor=sensors.BaseSensor(
                dT=1.0,
                PD=0.7,
                VC=1.0,
                PFA=1e-6,
                BNT=0.03
            ),
            model_factory=models.Simple2DModelFactory(
                model=models.PDAKalmanModel,
                q=0.0,
                pv=0.0
            ),
            track_factory=tracks.BaseTrackFactory(
                track=tracks.PDALLRTrack,
                gate=8
            )
        )

        a=2
        x1=(a**2+2-2.5)/2/a
        x2=(a**2+4-3)/2/a

        obs_list = [
            models.Obs(y, np.eye(2) * 0.5) for y in [
                np.array([0,0]),
                np.array([a,0]),
            ]
        ]

        tracker.register_scan(obs_list)

        obs_list = [
            models.Obs(y, np.eye(2) * 0.5) for y in [
                np.array([-1,0]),
                np.array([x1, np.sqrt(2 - x1**2)]),
                np.array([x2, np.sqrt(4 - x2**2)]),
            ]
        ]

        expected = np.array([
            1.0,
            9.0,
            2.0,
            2.5,
            4.0,
            3.0
        ])
        actual = np.array([
            dy @ np.linalg.inv(S) @ dy
            for dy, S in  [
                trk.model.residual(obs)
                for obs in obs_list for trk in tracker.trk_list
            ]
        ])
        np.testing.assert_almost_equal(actual, expected)

        tracker.register_scan(obs_list)

        expected = np.array([
            6.47e-5,
            5.04e-5,
            3.06e-5,
            1.44e-5,
            1.82e-5,
            1.11e-5,
            8.60e-6,
            6.70e-6,
            4.10e-6,
            2.40e-6,
        ])
        actual = tracker.hyp_score_list
        np.testing.assert_almost_equal(actual, expected)

        # targets = [
        #     np.array([-1,1]),
        #     np.array([20,36]),
        # ]

        # tracker.register_scan(
        #     [models.Obs(y, np.eye(2) * 5) for y in targets]
        # )

        # targets = [
        #     np.array([0,0]),
        #     np.array([25,44]),
        #     np.array([40,50]),
        # ]

        # tracker.register_scan(
        #     [models.Obs(y, np.eye(2) * 5) for y in targets]
        # )

        # targets = [
        #     np.array([0,0]),
        #     np.array([30,52]),
        #     np.array([40,50]),
        # ]

        # tracker.register_scan(
        #     [models.Obs(y, np.eye(2) * 5) for y in targets]
        # )

        # import matplotlib.pyplot as plt
        # for trk in tracker.trk_list:
        #     plt.plot(trk.cnt_list, trk.scr_list)

        # plt.show()

        # self.assertEqual(1, 2, "test")
