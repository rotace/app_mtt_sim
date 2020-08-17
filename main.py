""" MTT SIMULATION """

"""
    This program is made for learning, testing and comparing
    various algorithm of MTT.
"""

import os
import sys
import cmath
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as ani

import utils
import models
import tracks
import sensors
import trackers
import analyzers
from notebook import IRSTexample

def main():
    """
    Main Function

    In this function, you can select sample function.
    """

    # sample_IRSTexample372()
    # sample_IRSTexample878()
    sample_MultiSensorGNN()
    # sample_JPDA()

def sample_IRSTexample372():
    """
    This program calculate IRST(literature p.372) simulation with GNN.
    In addition, plot, animate, and analysis them.
    """
    gnn = IRSTexample.generate_irst_example_p372(PD=0.7, PFA=6e-5)
    gnn.animate_position(n_scan=50, is_all_obs_displayed=True)
    # result = gnn.estimate_track_statistics(n_scan=65, n_run=50)
    # plt.plot(result["Na"][0,:], label="Na")
    # plt.plot(result["Nc"][0,:], label="Nc")
    # plt.plot(result["Nm"][0,:], label="Nm")
    # plt.legend()
    # plt.show()

def sample_IRSTexample878():
    """
    This program calculate IRST(literature p.878) simulation with GNN.
    In addition, plot, animate, and analysis them.
    """
    gnn = IRSTexample.generate_irst_example_p878(PD=0.7, PFA=6e-7)
    gnn.plot_position(n_scan=50, is_all_obs_displayed=True)
    # result = gnn.estimate_track_statistics(n_scan=10, n_run=10)
    # print(result["Tc"][0])

def sample_MultiSensorGNN():
    """
    This program calculate two target / two sensor case with GNN, and animate it.
    """
    PD = 0.99
    PFA = 1e-7
    scan_time = 0.5
    sigma_o   = 1.0
    time_m    = 2.0
    sigma_mx  = 4.0
    sigma_my  = 1.0
    sigma_vx  = 18.0
    sigma_vy  =  4.0

    class Radar2DSensor(sensors.Polar2DSensor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def is_trk_in_range(self, trk):
            dist, azim = cmath.polar( (trk.model.x[0]-self.x[0]) + 1j*(trk.model.x[1]-self.x[1]) )
            return self.range_min < dist < self.range_max and abs(azim-self.angle) < self.width/2

        def is_tgt_in_range(self, tgt):
            dist, azim = cmath.polar( (tgt.x[0]-self.x[0]) + 1j*(tgt.x[1]-self.x[1]) )
            return self.range_min < dist < self.range_max and abs(azim-self.angle) < self.width/2

    sen_list = [
        Radar2DSensor(
            PD=PD,
            VC=1.0,
            PFA=PFA,
            BNT=0.03,
            x0=np.array([0.0, 10.0, 1.0, 0.0]),
            angle0=15/180*cmath.pi,
            DETECT_RANGE_MAX=40,
            DETECT_RANGE_MIN=10,
            DETECT_WIDTH=120/180*cmath.pi
        ),
        Radar2DSensor(
            PD=PD,
            VC=1.0,
            PFA=PFA,
            BNT=0.03,
            x0=np.array([50.0, 10.0, 1.0, 0.0]),
            angle0=-15/180*cmath.pi,
            DETECT_RANGE_MAX=40,
            DETECT_RANGE_MIN=10,
            DETECT_WIDTH=120/180*cmath.pi
        )
    ]

    tgt_list = [
        models.SimpleTarget(SD=2, x0=[100., 00.,-1.,+0.3]),
        models.SimpleTarget(SD=2, x0=[100., 10.,-1.,-0.1]),
        models.SimpleTarget(SD=2, x0=[100., 20.,-1.,-0.2]),
    ]

    tracker = trackers.GNN(
        sen_list=sen_list,
        model_factory=models.SingerModelFactory(
            model=models.KalmanModel,
            dT=scan_time,
            tm=time_m,
            sm=[sigma_mx, sigma_my],
            SD=2,
            P0=np.diag([sigma_o**2, sigma_o**2, sigma_vx**2, sigma_vy**2])
        ),
        track_factory=tracks.BaseTrackFactory(
            track=tracks.ScoreManagedTrack
        )
    )

    obs_df = pd.DataFrame()
    tgt_df = pd.DataFrame()
    trk_df = pd.DataFrame()
    sen_df = pd.DataFrame()

    for i_scan in range(100):
        timestamp = pd.Timestamp(i_scan, unit="s")

        if not i_scan % 5:
            # scan by sensor0 (once in 5 times)
            sensor = sen_list[0]
            R = np.eye(2) * 0.01
            obs_list = [models.Obs(np.random.multivariate_normal(tgt.x[:2], R), R) for tgt in tgt_list if sensor.is_tgt_in_range(tgt)]
            trk_list = tracker.register_scan(obs_list, sensor=sensor)
        else:
            # scan by sensor1 (everytime except sensor0 turn)
            sensor = sen_list[1]
            R = np.eye(2) * 0.01
            obs_list = [models.Obs(np.random.multivariate_normal(tgt.x[:2], R), R) for tgt in tgt_list if sensor.is_tgt_in_range(tgt)]
            trk_list = tracker.register_scan(obs_list, sensor=sensor)

        # tgt_list update
        [ tgt.update(tracker._dT()) for tgt in tgt_list ]

        # save as dataframe
        obs_df = obs_df.append( [ obs.to_record(timestamp, i_scan) for obs in obs_list ], ignore_index=True )
        trk_df = trk_df.append( [ trk.to_record(timestamp, i_scan) for trk in trk_list ], ignore_index=True )
        tgt_df = tgt_df.append( [ tgt.to_record(timestamp, i_scan) for tgt in tgt_list ], ignore_index=True)
        sen_df = sen_df.append( [ sen.to_record(timestamp, i_scan) for sen in sen_list ], ignore_index=True )

    # export
    anal = analyzers.BaseAnalyzer.import_df(tracker, obs_df, trk_df, sen_df, tgt_df)
    anal.export_csv()
    anal.export_db()

    # analyse
    analyzers.main()

def sample_JPDA():
    """
    This program calculate two target case with JPDA, and animate it.
    But there are some problems that I still not solved.

    1. Track Confirmation and Deletion
        In this case, I implemented IPDA method for track confirmation and deletion,
        but the judgement argorithm is alternative one I made due to lack of knowledge.
        Temporally, I set the threshold of Pt (deletion at under 0.4, confirmation at over 0.95).
        However it seems not to be good. It's needed to search more literature about IPDA.

    2. Presentation Logic
        JPDA has many unlikely tracks, so it should be to implement presentation logic, which
        select likely track and display them. But not implemented yet.
    """
    PD = 0.99
    PFA = 1e-7
    scan_time = 0.5
    sigma_o   = 1.0
    time_m    = 2.0
    sigma_mx  = 4.0
    sigma_my  = 1.0
    sigma_vx  = 18.0
    sigma_vy  =  4.0

    tracker = trackers.JPDA(
        sensor=sensors.BaseSensor(
            PD=PD,
            VC=1.0,
            PFA=PFA,
            BNT=0.03
        ),
        model_factory=models.SingerModelFactory(
            model=models.PDAKalmanModel,
            dT=scan_time,
            tm=time_m,
            sm=[sigma_mx, sigma_my],
            SD=2,
            P0=np.diag([sigma_o**2, sigma_o**2, sigma_vx**2, sigma_vy**2])
        ),
        track_factory=tracks.BaseTrackFactory(
            track=tracks.PDATrack
        )
    )

    tgt_list = [
        np.array([0.,0.,1.,1.]),
        np.array([100.,100.,-1.,-1.])
    ]

    art_list =[]
    fig = plt.figure()
    plt.axis("equal")
    plt.grid()

    for i_scan in range(10):

        trk_list = tracker.register_scan(
            [models.Obs(tgt[:2], np.eye(2) * 5) for tgt in tgt_list]
        )

        tgt_art = plt.plot(
                [tgt[0] for tgt in tgt_list ],
                [tgt[1] for tgt in tgt_list ],
                marker="D", color="b", alpha=.5, linestyle="None", label="tgt"
            )

        trk_art = plt.plot(
            [trk.model.x[0] if trk is not None else None for trk in trk_list ],
            [trk.model.x[1] if trk is not None else None for trk in trk_list ],
            marker="D", color="r", alpha=.5, linestyle="None", label="trk"
        )

        ax_pos = plt.gca().get_position()
        count = fig.text( ax_pos.x1-0.1, ax_pos.y1-0.05, "count:" + str(i_scan), size = 10 )

        art_list.append( trk_art + tgt_art + [count] )

        for tgt in tgt_list:
            tgt[:2] += tgt[2:]*scan_time

    _ = ani.ArtistAnimation(fig, art_list, interval=1000)
    plt.show()


""" Execute Section """
if __name__ == '__main__':
    if (sys.flags.interactive != 1):
        main()