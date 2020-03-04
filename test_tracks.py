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
        
        ND=3
        
        tracker = MockTracker(
            sensor=sensors.BaseSensor(
                dT=1.0,
                PD=0.7,
                VC=1.0,
                PFA=1e-6,
                BNT=0.03
            ),
            model_factory=models.SimpleModelFactory(
                model=models.KalmanModel,
                q=0.001
            )
        )

        tgt = np.array([0.0,  0.0, 1.0, 1.0])

        for k in range(200):

            tracker.count = k

            if k==0:
                trk = tracks.DistTrack(
                    models.Obs(tgt[:2], np.eye(2), tracker.sensor ) ,
                    tracker,
                    ND=ND
                )
                
            if 0 < k <= 100:
                tgt[0:2] += tgt[2:]
                
                trk.assign(models.Obs(
                    # tgt[:2],  np.eye(2), tracker.sensor
                    np.random.multivariate_normal(tgt[:2], np.eye(2)), np.eye(2), tracker.sensor
                ))

            if 100 < k:
                trk.unassign()


            # judge_deletion test
            if 0 <= k < 100+ND:
                np.testing.assert_equal(trk.judge_deletion(), False)
                pass

            elif 100+ND <= k:
                np.testing.assert_equal(trk.judge_deletion(), True)
                pass

        # after loop
        if False:
            # trk.plot_obs_list()
            # trk.plot_mdl_list()
            trk.plot_gate()        
            plt.show()


    def test_LLRTrack(self):
        """Track Score Function

            ref) Design and Analysis of Modern Tracking Systems
                        6.2 Track Score Function
        """
        PD=0.7
        PFD=1.e-3
        NFC=4
        NFA=1000

        tracker = MockTracker(
            sensor=sensors.BaseSensor(
                dT=1.0,
                PD=PD,
                VC=1.0,
                PFA=1e-6,
                BNT=0.03
            ),
            model_factory=models.SimpleModelFactory(
                model=models.KalmanModel,
                q=0.001
            )
        )

        tgt = np.array([0.0,  0.0, 1.0, 1.0])
        com_list = []
        del_list = []

        for k in range(200):

            tracker.count = k

            if k==0:
                trk = tracks.LLRTrack(
                    models.Obs(tgt[:2], np.eye(2), tracker.sensor ) ,
                    tracker,
                    PFD=PFD,
                    alpha=NFC/3600/NFA,
                    beta=0.1
                )
                
            if 0 < k <= 100:
                tgt[0:2] += tgt[2:]

                obs = models.Obs(
                    # tgt[:2],  np.eye(2), tracker.sensor
                    np.random.multivariate_normal(tgt[:2], np.eye(2)), np.eye(2), tracker.sensor
                )
                
                if np.random.choice([True, False], p=[PD, 1-PD]):
                    trk.assign(obs)
                else:
                    trk.unassign()

            if 100 < k:
                trk.unassign()

            com_list.append(trk.judge_confirmation())
            del_list.append(trk.judge_deletion())

        # after loop
        if False:
            # trk.plot_obs_list()
            # trk.plot_mdl_list()
            # trk.plot_gate()
            # trk.plot_scr_list()
            plt.plot([i for i in range(200)], np.array(com_list, int) - np.array(del_list, int))
            plt.show()