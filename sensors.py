import cmath
import numpy as np
import pandas as pd
from scipy import stats

import models


class BaseSensor():
    """ Sensor Base Class

    ref) Design and Analysis of Modern Tracking Systems
                6.2 Track Score Function
    """
    sen_id_counter = 0
    @classmethod
    def _generate_id(cls):
        cls.sen_id_counter+=1
        return cls.sen_id_counter

    def __init__(self, **kwargs):
        assert "dT" not in kwargs

        if ("BNT" in kwargs and "PFA" in kwargs and "VC" in kwargs):
            kwargs["PNT"] = kwargs["BNT"] * kwargs["VC"]
            kwargs["BFT"] = kwargs["PFA"] / kwargs["VC"]
            kwargs["BETA"] = kwargs["BNT"] + kwargs["BFT"]
        
        self.param = kwargs
        self.count = 0
        self.sen_id = BaseSensor._generate_id()

    def get_id(self):
        return self.sen_id

    def update(self, dT, *args, **kwargs):
        self.count += 1
    
    def is_trk_in_range(self, trk):
        return True

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
                self.param["R"] # set as model parameter R
            )
            for tgt in tgt_list if tgt.is_exist() and np.random.choice([True, False], p=[PD, 1-PD])
        ])
        
        # add false alarm observation
        obs_list.extend([
            models.Obs(
                np.array([ np.random.uniform(y_min, y_max) for y_min, y_max in zip(y_mins, y_maxs) ]),
                self.param["R"] # set as model parameter R
            )
            for k in range(stats.binom.rvs(n=n_mesh, p=PFA))
        ])

        return obs_list


class Polar2DSensor(BaseSensor):
    """ Polar 2D Area Sensor

    Moving Model : Constant Velocity Motion
    Coordinate Type : Cartesian
    Space Dimension : 2
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mdl_type = models.ModelType.generate_model_type(
            crd_type=models.CoordType.CART,
            SD=2,
            RD=2
        )
        self.x = kwargs["x0"]
        self.angle = kwargs["angle0"]
        self.width = kwargs["DETECT_WIDTH"]
        self.range_max = kwargs["DETECT_RANGE_MAX"]
        self.range_min = kwargs["DETECT_RANGE_MIN"]

    def to_series(self, timestamp, scan_id):
        assert isinstance(timestamp, pd.Timestamp), "timestamp is invalid, actual:"+str(timestamp)
        x_lbl = [ v.name for v in self.mdl_type.val_type ]
        value=[scan_id, self.get_id(), Polar2DSensor.__name__]
        label=["SCAN_ID", "SEN_ID", "SEN_TYPE"]
        value+= list(self.x)+[self.angle-self.width/2, self.angle+self.width/2, self.range_min, self.range_max] 
        label+= x_lbl+["THETA_MIN", "THETA_MAX", "RANGE_MIN", "RANGE_MAX"]
        return pd.Series(value, index=label, name=timestamp)

    def update(self, dT, *args, **kwargs):
        super().update(dT, *args, **kwargs)
        self.x[0] += self.x[2]*dT
        self.x[1] += self.x[3]*dT
    
    def is_trk_in_range(self, trk):
        dx, dx_type = models.ModelType.diff_x(self.x, self.mdl_type, trk.model.x, trk.model._x_type)

        if dx_type.crd_type == models.CoordType.CART:
            px = dx[dx_type.val_type.index(models.ValueType.POSIT_X)]
            py = dx[dx_type.val_type.index(models.ValueType.POSIT_Y)]
            dist, azim = cmath.polar(  px + 1j*py )
        
        elif dx_type.crd_type == models.CoordType.POLAR:
            dist = dx[dx_type.val_type.index(models.ValueType.POSIT_DIST)]
            azim = dx[dx_type.val_type.index(models.ValueType.POSIT_AZIM)]
        
        else:
            raise NotImplementedError

        return self.range_min < dist < self.range_max and abs(azim-self.angle) < self.width/2


class SimpleRadar2DSensor(Polar2DSensor):
    """ Range and Azimuth Measured Radar
    
    This is used with models.SimplePolar2D
    """
    pass
