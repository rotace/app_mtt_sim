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

    def test_counter(self):
        class InheritBaseTrack(tracks.BaseTrack):
            pass
        tracks.BaseTrack.initialize()
        np.testing.assert_equal(InheritBaseTrack._generate_id(), 1)
        np.testing.assert_equal(tracks.BaseTrack._generate_id(), 2)
        InheritBaseTrack.initialize()
        np.testing.assert_equal(tracks.BaseTrack._generate_id(), 1)
        np.testing.assert_equal(InheritBaseTrack._generate_id(), 2)


    def test_SimpleManagedTrack(self):
        
        ND=3
        
        tracker = MockTracker(
            sensor=sensors.BaseSensor(
                PD=0.7,
                VC=1.0,
                PFA=1e-6,
                BNT=0.03
            ),
            model_factory=models.SimpleModelFactory(
                model=models.KalmanModel,
                dT=1.0,
                q=0.001
            )
        )

        tgt = np.array([0.0,  0.0, 1.0, 1.0])

        for k in range(200):

            tracker.count = k

            if k==0:
                obs = models.Obs(tgt[:2], np.eye(2), tracker.sensor )
                trk = tracks.SimpleManagedTrack(
                    obs,
                    tracker.model_factory.create(obs),
                    ND=ND
                )
                
            if 0 < k <= 100:
                tgt[0:2] += tgt[2:]
                
                trk.assign(models.Obs(
                    # tgt[:2],  np.eye(2), tracker.sensor
                    np.random.multivariate_normal(tgt[:2], np.eye(2)), np.eye(2), tracker.sensor
                ))

            if 100 < k:
                trk.unassign(tracker.sensor)


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
            plt.show()


    def test_ScoreManagedTrack(self):
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
                PD=PD,
                VC=1.0,
                PFA=1e-6,
                BNT=0.03
            ),
            model_factory=models.SimpleModelFactory(
                model=models.KalmanModel,
                dT=1.0,
                q=0.001
            )
        )

        tgt = np.array([0.0,  0.0, 1.0, 1.0])
        com_list = []
        del_list = []

        for k in range(200):

            tracker.count = k

            if k==0:
                obs = models.Obs(tgt[:2], np.eye(2), tracker.sensor )
                trk = tracks.ScoreManagedTrack(
                    obs,
                    tracker.model_factory.create(obs),
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
                    trk.unassign(tracker.sensor)

            if 100 < k:
                trk.unassign(tracker.sensor)

            com_list.append(trk.judge_confirmation())
            del_list.append(trk.judge_deletion())

        # after loop
        if False:
            # trk.plot_obs_list()
            # trk.plot_mdl_list()
            # trk.plot_scr_list()
            plt.plot([i for i in range(200)], np.array(com_list, int) - np.array(del_list, int))
            plt.show()


    def test_TrackEvaluator(self):

        scan_time = 1.0
        sigma_o   = 1.0
        time_m    = 2.0
        sigma_mx  = 4.0
        sigma_my  = 1.0
        sigma_vx  = 18.0
        sigma_vy  =  4.0
        vx0 = np.random.normal(0.0, sigma_vx)
        vy0 = np.random.normal(0.0, sigma_vy)

        eval = tracks.TrackEvaluator(
            sensor=sensors.BaseSensor(
                PD=0.7,
                VC=1.0,
                PFA=1e-6,
                BNT=0.03
            ),
            model_factory=models.SingerModelFactory(
                model=models.KalmanModel,
                dT=1.0,
                tm=time_m,
                sm=[sigma_mx, sigma_my],
                SD=2,
                P0=np.diag([sigma_o**2, sigma_o**2, sigma_vx**2, sigma_vy**2])
            ),
            track_factory=tracks.BaseTrackFactory(
                track=tracks.ScoreManagedTrack
            ),
            target=models.SimpleTarget(
                x0=[0.0, 0.0, vx0, vy0],
                SD=2
            ),
            R=np.diag([sigma_o**2, sigma_o**2])
        )

        # eval.plot_score()