from scipy import stats

import numpy as np
import models


class BaseSensor():
    """ Sensor Base Class

    ref) Design and Analysis of Modern Tracking Systems
                6.2 Track Score Function
    """
    def __init__(self, **kwargs):
        assert "dT" not in kwargs

        if ("BNT" in kwargs and "PFA" in kwargs and "VC" in kwargs):
            kwargs["PNT"] = kwargs["BNT"] * kwargs["VC"]
            kwargs["BFT"] = kwargs["PFA"] / kwargs["VC"]
            kwargs["BETA"] = kwargs["BNT"] + kwargs["BFT"]
        
        self.param = kwargs
        self.count = 0

    def update(self, *args, **kwargs):
        self.count += 1

    def calc_LLR0(self):
        assert "PNT" in self.param
        assert "PD"  in self.param
        assert "PFA" in self.param
        L0  = np.log( self.param["PNT"] )
        dLk = 0
        dLs = np.log( self.param["PD"] / self.param["PFA"] )
        return L0 + dLk + dLs

    def calc_match_dLLR(self, log_gij):
        assert "VC"  in self.param
        assert "PD"  in self.param
        assert "PFA" in self.param
        dLk = np.log( self.param["VC"] ) + log_gij
        dLs = np.log( self.param["PD"] / self.param["PFA"] )
        return dLk + dLs

    def calc_miss_dLLR(self):
        assert "PD"  in self.param
        dLk = 0
        dLs = np.log( 1.0 - self.param["PD"] )
        return dLk + dLs

    def calc_ellipsoidal_gate(self, detS, M):
        """Calc Ellipsoidal Gate

        dLLR(match) > dLLR(miss) === gate > dist

        ref) Design and Analysis of Modern Tracking Systems
                    6.3.2 Ellipsoidal Gates
        """
        assert "PD"  in self.param
        assert "BFT" in self.param
        PD  = self.param["PD"]
        BFT = self.param["BFT"]
        COEF = (2*np.pi) ** (0.5*M)
        gate = 2*( np.log(PD/(1-PD)/BFT) - np.log(COEF*np.sqrt(detS)) )
        return gate

    def create_obs_list(self, tgt_list, R=None, PD=None, PFA=None):
        assert "R" in self.param
        assert "PD"  in self.param
        assert "PFA" in self.param
        assert "y_mins" in self.param
        assert "y_maxs" in self.param
        assert "y_stps" in self.param

        if R is None:
            R = self.param["R"]
        if PD is None:
            PD = self.param["PD"]
        if PFA is None:
            PFA = self.param["PFA"]

        y_mins = self.param["y_mins"]
        y_maxs = self.param["y_maxs"]
        y_stps = self.param["y_stps"]
        n_mesh = int(np.prod([ abs((y_max-y_min)/y_stp) for y_min, y_max, y_stp in zip(y_mins, y_maxs, y_stps) ]))

        # init observation
        obs_list = []

        # add target observation
        obs_list.extend([
            models.Obs(
                np.random.multivariate_normal(tgt.x[:len(R)], R), # set as real parameter R
                self.param["R"], # set as model parameter R
                self
            )
            for tgt in tgt_list if tgt.is_exist() and np.random.choice([True, False], p=[PD, 1-PD])
        ])
        
        # add false alarm observation
        obs_list.extend([
            models.Obs(
                np.array([ np.random.uniform(y_min, y_max) for y_min, y_max in zip(y_mins, y_maxs) ]),
                self.param["R"], # set as model parameter R
                self
            )
            for k in range(stats.binom.rvs(n=n_mesh, p=PFA))
        ])

        return obs_list

class SimpleRadar2DSensor(BaseSensor):
    """ Range and Azimuth Measured Radar
    
    This is used with models.SimplePolar2D
    """
    pass
