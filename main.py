""" MTT SIMULATION """

"""
    This program is made for learning, testing and comparing
    various algorithm of MTT.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import utils
import models
import tracks
import sensors
import trackers

class SimSample:
    """ Simulation Samples
    """
    class SinusoidTarget(models.Target):
        def __init__(self, A, Tp):
            super().__init__()
            self.A = A
            self.omega = 2*np.pi/Tp
        
        def update_x(self, T, dT):
            self.x = np.array([
                +self.A * np.sin(self.omega * T),
                +0.0,
                +self.A * np.cos(self.omega * T) * self.omega,
                +0.0,
                -self.A * np.sin(self.omega * T) * self.omega**2,
                +0.0
            ])

    @staticmethod
    def generate_irst_example_p878():
        """ IRST example of p.878

        unit is pixcel (=70urad)

        ref) Design and Analysis of Modern Tracking Systems
        13.3.5 IRST Example
        """

        scan_time = 1.0
        sigma_o   = 1.0
        time_m    = 2.0
        sigma_mx  = 4.0
        sigma_my  = 1.0

        vx0 = np.random.normal(0.0, 18.0)
        vy0 = np.random.normal(0.0,  4.0)


        eval = trackers.TrackerEvaluator(
            trackers.GNN(
                sensor=sensors.BaseSensor(
                    dT=scan_time,
                    PD=0.7,
                    VC=1.0,
                    PFA=1e-6,
                    BNT=0.03
                ),
                model_factory=models.SingerModelFactory(
                    model=models.KalmanModel,
                    tm=time_m,
                    sm=[sigma_mx, sigma_my],
                    SD=2
                ),
                track_factory=tracks.BaseTrackFactory(
                    track=tracks.LLRTrack,
                    gate=None
                )
            ),
            tgt_list=[
                models.SimpleTarget(
                    x0=[0.0, 0.0, vx0, vy0],
                    SD=2),
            ],
            R=np.diag([sigma_o, sigma_o])
        )

        return eval

    @staticmethod
    def generate_irst_example_p372():
        """ IRST example of p.372

        unit is pixcel (=70urad)

        ref) Design and Analysis of Modern Tracking Systems
        6.8.2 Simulation Study Results
        """


        scan_time = 1.0
        sigma_o   = 1.0
        time_m    = 2.5
        sigma_mx  = 25.
        sigma_my  = 2.0

        eval = trackers.TrackerEvaluator(
            trackers.GNN(
                sensor=sensors.BaseSensor(
                    dT=scan_time,
                    PD=0.7,
                    VC=1.0,
                    PFA=1e-6,
                    BNT=0.03
                ),
                model_factory=models.SingerModelFactory(
                    model=models.KalmanModel,
                    tm=time_m,
                    sm=[sigma_mx, sigma_my],
                    SD=2
                ),
                track_factory=tracks.BaseTrackFactory(
                    track=tracks.LLRTrack,
                    gate=None
                )
            ),
            tgt_list=[
                SimSample.SinusoidTarget(A=9.0, Tp=3.5)
            ],
            R=np.diag([sigma_o, sigma_o])
        )

        return eval

def main():
    """
    Main Function
    """
    eval = SimSample.generate_irst_example_p878()
    eval.plot_tgt_trk(n_scan=10)

if __name__ == '__main__':
    if (sys.flags.interactive != 1):
        main()