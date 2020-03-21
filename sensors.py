import numpy as np


class BaseSensor():
    """ Sensor Base Class

    ref) Design and Analysis of Modern Tracking Systems
                6.2 Track Score Function
    """
    def __init__(self, **kwargs):
        
        if "dT" not in kwargs:
            kwargs["dT"] = 1.0
        if "PD" not in kwargs:
            kwargs["PD"] = 0.7
        if "PFA" not in kwargs:
            kwargs["PFA"] = 1e-6
        if "VC" not in kwargs:
            kwargs["VC"] = 1.0
        if "BNT" not in kwargs:
            kwargs["BNT"] = 0.03

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


class SimpleRadar2DSensor(BaseSensor):
    """ Range and Azimuth Measured Radar
    
    This is used with models.SimplePolar2D
    """
    pass
