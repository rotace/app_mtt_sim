import unittest
import numpy as np

import models
import tracks
import sensors
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

class MockTracker():
    def __init__(self, sensor, model_factory):
        self.sensor = sensor
        self.model_factory = model_factory
        self.count = 0

class TestTracks(unittest.TestCase):
    
    def test_DistTrack(self):
        tracker = MockTracker(
            sensor=sensors.BaseSensor(
                dT=1.0,
                PD=0.7,
                VC=1.0,
                PFA=1e-6,
                BNT=0.03
            ),
            model_factory=models.Simple2DModelFactory(
                model=models.KalmanModel,
                q=0.001,
                pv=0.1
            )
        )

        sim_targets = [
            np.array([0.0,  0.0, 1.0, 1.0]),
        ]

        sim_tracks = []

        for k in range(100):

            tracker.count = k

            if k==0:
                sim_tracks.extend(
                    [
                        tracks.DistTrack(
                            models.Obs(tgt[:2], np.eye(2), tracker.sensor ) ,
                            tracker
                        ) for tgt in sim_targets
                    ]
                )

            if k > 0:
                for tgt in sim_targets:
                    tgt[0:2] += tgt[2:]
                
                for trk, tgt in zip(sim_tracks, sim_targets):
                    trk.assign(models.Obs(
                        # tgt[:2],  np.eye(2), tracker.sensor
                        np.random.multivariate_normal(tgt[:2], np.eye(2)), np.eye(2), tracker.sensor
                    ))

            # plt.plot(
            #     [tgt[0] for tgt in sim_targets],
            #     [tgt[1] for tgt in sim_targets],
            #     marker="D", color="y", alpha=.5, linestyle="None"
            # )

            # plt.plot(
            #     [trk.model.x[0] for trk in sim_tracks],
            #     [trk.model.x[1] for trk in sim_tracks],
            #     marker="D", color="r", alpha=.5, linestyle="None"
            # )

            # plt.plot(
            #     k,
            #     sim_targets[-1][0],
            #     marker="D", color="y", alpha=.5, linestyle="None"
            # )

            # plt.plot(
            #     k,
            #     sim_tracks[-1].model.x[0],
            #     marker="D", color="r", alpha=.5, linestyle="None"
            # )

        # after loop
        if False:
            for trk in sim_tracks:
                # trk.plot_obs_list()
                # trk.plot_mdl_list()
                trk.plot_gate()        
            plt.show()
