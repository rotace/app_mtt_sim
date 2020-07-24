import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
import utils
import models
import tracks
import sensors
import trackers

def generate_irst_example_p878(PD=0.7, PFA=1e-6):
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
    sigma_vx  = 18.0
    sigma_vy  =  4.0
    vx0 = np.random.normal(0.0, sigma_vx)
    vy0 = np.random.normal(0.0, sigma_vy)

    gnn = trackers.TrackerEvaluator(
        tracker=trackers.GNN(
            sensor=sensors.BaseSensor(
                dT=scan_time,
                PD=PD,
                VC=1.0,
                PFA=PFA,
                BNT=0.03,
                # y_mins=[-44880,-250],
                # y_maxs=[+44880,+250],
                y_mins=[-1125,-250], # reduce calculation load
                y_maxs=[+1125,+250], # reduce calculation load
                y_stps=[1,1]
            ),
            model_factory=models.SingerModelFactory(
                model=models.KalmanModel,
                tm=time_m,
                sm=[sigma_mx, sigma_my],
                SD=2,
                P0=np.diag([sigma_o**2, sigma_o**2, sigma_vx**2, sigma_vy**2])
            ),
            track_factory=tracks.BaseTrackFactory(
                track=tracks.ScoreManagedTrack
            )
        ),
        tgt_list=[
            models.SimpleTarget(
                x0=[0.0, 0.0, vx0, vy0],
                SD=2),
        ],
        R=np.diag([sigma_o**2, sigma_o**2])
    )

    # TODO: implement SPRT, SMC(MHT), FMC(MHT), FMC(IPDA)

    # SPRT( Sequential Probability Ratio Test )
    # 13.3.3 SPRT Analysis of Track Confirmation (p874)

    # SMC ( Simplified Monte Carlo Simulation )
    # 13.3.4 Simplified Monte Carlo Simulation (p877)

    # FMC ( Full Monte Carlo Simulation ) (p878)
    # 13.3.5 IRST Example

    return gnn


class TrackerEvaluatorForP372(trackers.TrackerEvaluator):
    def __init__(self, tracker, tgt_list, R, PD, PFA):
        super().__init__(tracker, tgt_list, R)
        self.PD_after_10 = PD
        self.PFA_after_10 = PFA

    def _update_sim_param(self, i_scan):
        if i_scan is not None and i_scan < 10:
            self.PD = 1.0
            self.PFA = 10**-4
        else:
            self.PD = self.PD_after_10
            self.PFA = self.PFA_after_10

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

def generate_irst_example_p372(PD=0.7, PFA=1e-6, is_maneuver_enabled=True):
    """ IRST example of p.372

    unit is pixcel (=70urad)

    ref) Design and Analysis of Modern Tracking Systems
    6.8.2 Simulation Study Results
    """

    scan_time = 1.0
    sigma_o   = 1.0
    time_m    = 2.5

    if is_maneuver_enabled:
        target = SinusoidTarget(A=9.0, Tp=3.5)
        sigma_mx  = 25.0
        sigma_my  = 2.0
    else:
        target = SinusoidTarget(A=0.0, Tp=3.5)
        sigma_mx  = 5.0
        sigma_my  = 2.0

    gnn = TrackerEvaluatorForP372(
        tracker=trackers.GNN(
            sensor=sensors.BaseSensor(
                dT=scan_time,
                PD=PD,
                VC=1.0,
                PFA=PFA,
                BNT=0.03,
                # y_mins=[-44880,-250],
                # y_maxs=[+44880,+250],
                y_mins=[-50,-50], # reduce calculation load
                y_maxs=[+50,+50], # reduce calculation load
                y_stps=[1,1]
            ),
            model_factory=models.SingerModelFactory(
                model=models.KalmanModel,
                tm=time_m,
                sm=[sigma_mx, sigma_my],
                SD=2,
                P0=np.diag([sigma_o**2, sigma_o**2, (target.A*target.omega)**2, (target.A*target.omega)**2])
            ),
            track_factory=tracks.BaseTrackFactory(
                track=tracks.ScoreManagedTrack
            )
        ),
        tgt_list=[target],
        R=np.diag([sigma_o**2, sigma_o**2]),
        PD=PD,
        PFA=PFA
    )

    return gnn
