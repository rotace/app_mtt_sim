import numpy as np


class BaseSensor():
    """ Sensor Base Class """

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
        if "BFT" not in kwargs:
            kwargs["BFT"] = kwargs["PFA"]/kwargs["VC"]
        if "BETA" not in kwargs:
            kwargs["BETA"] = kwargs["BNT"] + kwargs["BFT"]
        
        self.param = kwargs
        self.count = 0

    def update(self, *args, **kwargs):
        self.count += 1


class SimpleRadar2DSensor(BaseSensor):
    """ Range and Azimuth Measured Radar
    
    This is used with models.SimplePolar2D
    """
    pass
