import numpy as np


class Sensor():
    """ Sensor Base Class """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError


class SimpleRadar2D(Sensor):
    """ Range and Azimuth Measured Radar
    
    This is used with models.SimplePolar2D
    """

    def __init__(
        self,
        time_interval
    ):
        self.param = dict()
        self.param["time_interval"] = time_interval
        self.count = 0

    def update(self):
        self.count += 1
