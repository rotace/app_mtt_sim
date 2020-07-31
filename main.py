""" MTT SIMULATION """

"""
    This program is made for learning, testing and comparing
    various algorithm of MTT.
"""

import os
import sys
import cmath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as ani

import utils
import models
import tracks
import sensors
import trackers
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

    class Radar2DSensor(sensors.BaseSensor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.px = kwargs["INIT_POSITION"][0]
            self.py = kwargs["INIT_POSITION"][1]
            self.vx = kwargs["CRUISE_SPEED"][0]
            self.vy = kwargs["CRUISE_SPEED"][1]
            self.range = kwargs["DETECT_RANGE"]
            self.width = kwargs["DETECT_WIDTH"]
        def update(self, dT, *args, **kwargs):
            super().update(dT, *args, **kwargs)
            self.px += self.vx*dT
            self.py += self.vy*dT
        def is_trk_in_range(self, trk):
            dist, azim = cmath.polar( (trk.model.x[0]-self.px) + 1j*(trk.model.x[1]-self.py) )
            return dist < self.range and abs(azim) < self.width/2
        def is_tgt_in_range(self, tgt):
            dist, azim = cmath.polar( (tgt[0]-self.px) + 1j*(tgt[1]-self.py) )
            return dist < self.range and abs(azim) < self.width/2

    sen_list = [
        Radar2DSensor(
            PD=PD,
            VC=1.0,
            PFA=PFA,
            BNT=0.03,
            INIT_POSITION=(0.0, 10.0),
            CRUISE_SPEED=(1.0, 0.0),
            DETECT_RANGE=40,
            DETECT_WIDTH=120/180*cmath.pi
        ),
        Radar2DSensor(
            PD=PD,
            VC=1.0,
            PFA=PFA,
            BNT=0.03,
            INIT_POSITION=(50.0, 10.0),
            CRUISE_SPEED=(1.0, 0.0),
            DETECT_RANGE=40,
            DETECT_WIDTH=120/180*cmath.pi
        )
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

    tgt_list = [
        np.array([100., 0.,-1.,+0.3]),
        np.array([100.,10.,-1.,-0.1]),
        np.array([100.,20.,-1.,-0.2])
    ]

    art_list =[]
    fig = plt.figure()
    plt.axis("equal")
    plt.grid()

    for i_scan in range(100):

        if not i_scan % 5:
            # scan by sensor0 (once in 5 times)
            sensor = sen_list[0]
            R = np.eye(2) * 0.01
            trk_list = tracker.register_scan(
                [models.Obs(np.random.multivariate_normal(tgt[:2], R), R) for tgt in tgt_list if sensor.is_tgt_in_range(tgt)], sensor=sensor
            )
        else:
            # scan by sensor1 (everytime except sensor0 turn)
            sensor = sen_list[1]
            R = np.eye(2) * 0.01
            trk_list = tracker.register_scan(
                [models.Obs(np.random.multivariate_normal(tgt[:2], R), R) for tgt in tgt_list if sensor.is_tgt_in_range(tgt)], sensor=sensor
            )

        for tgt in tgt_list:
            tgt[:2] += tgt[2:]*scan_time

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

        sen_art = plt.plot(
            [sen.px for sen in sen_list],
            [sen.py for sen in sen_list],
            marker="D", color="g", alpha=.5, linestyle="None", label="trk"
        )

        pat_art = [plt.gca().add_patch(pat.Wedge(
            center=(sen.px, sen.py),
            r=sen.range,
            theta1=-sen.width*180/cmath.pi/2,
            theta2=+sen.width*180/cmath.pi/2,
            color="g",
            alpha=0.2
        )) for sen in sen_list]

        ax_pos = plt.gca().get_position()
        count = fig.text( ax_pos.x1-0.1, ax_pos.y1-0.05, "count:" + str(i_scan), size = 10 )

        art_list.append( trk_art + tgt_art + sen_art + pat_art + [count] )


    _ = ani.ArtistAnimation(fig, art_list, interval=200)
    plt.show()


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