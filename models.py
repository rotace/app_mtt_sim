import numpy as np
from scipy.linalg import block_diag

import sensors


class Obs():
    """ Observation """

    def __init__(self, y, R, sensor=None):
        """Initialize Observation
        
        Arguments:
            y {np.ndarray} -- sensor value vector
            R {np.ndarray} -- sensor value variance matrix
            sensor {Sensor} -- sensor
        """
        self.y = y
        self.R = R
        self.sensor = sensor



class Model():
    """ Model Base Class """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def residual(self, obs):
        # return (dy, S)
        raise NotImplementedError

    def update(self, obs=None):
        # how to implement update
        if obs:
            # update with observation
            pass
        else:
            # update without observation
            pass

        raise NotImplementedError



class KalmanModel(Model):
    """ Kalman Model
    
        ref) Design and Analysis of Modern Tracking Systems
                    3.3 Kalman Filtering
                    3.4 Extended Kalman Filtering
    """
    def __init__(self, x, F, H, P, Q, is_nonlinear):
        assert isinstance(x, np.ndarray)
        assert isinstance(F, np.ndarray)
        assert isinstance(H, np.ndarray)
        assert isinstance(P, np.ndarray)
        assert isinstance(Q, np.ndarray)
        
        self.is_nonlinear = is_nonlinear
        self.x_dim = H.shape[1]
        self.y_dim = H.shape[0]
        assert x.shape == (self.x_dim, ), "x.shape invalid, actual:"+str(x.shape)
        assert F.shape == (self.x_dim, self.x_dim), "F.shape invalid, actual:"+str(F.shape)
        assert P.shape == (self.x_dim, self.x_dim), "P.shape invalid, actual:"+str(P.shape)
        assert Q.shape == (self.x_dim, self.x_dim), "Q.shape invalid, actual:"+str(Q.shape)

        self.x = x
        self.F = F
        self.H = H
        self.P = P
        self.Q = Q
        self._predict_step()

    def update(self, obs):
        """Update Model"""
        if obs:
            if self.is_nonlinear:
                self.H = self._update_H(self.x)
            self._inovate_step(obs.R)
            self._correct_step(obs.y)
            self._predict_step()
        else:
            self._predict_step()

    def residual(self, obs):
        """ Residual

        dy = y - H*x
        S = H*P*H' + R
        """
        if self.is_nonlinear:
            self.H = self._update_H(self.x)
        dy = obs.y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + obs.R
        return (dy, S)

    def _inovate_step(self, R):
        """Inovate Step
        
        S = H*P*H' + R
        K = (P*H')/S
        """
        self.S = self.H @ self.P @ self.H.T + R
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)

    def _correct_step(self, y):
        """Correct Step
        
         P = P - K*H*P
         x = x + K*(y-Hx)
        """
        self.P = self.P - self.K @ self.H * self.P
        self.x = self.x + self.K @ (y-self.H @ self.x)

    def _predict_step(self):
        """Predict Step
        
         P = F*P*F' + Q
         x = F*x
        """
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x = self.F @ self.x
    
    def _update_H(self, x):
        # You need implement at subclass if use nonlinear H
        raise NotImplementedError



class SimplePolar2D(KalmanModel):
    """ Simple Polar (R,Theta) Model
    
     * Linear Kalman Filter
     * Constant Velocity Model
     * State Vector x is Polar Coord
     * Input Vector y is Polar Coord
     * R and Theta is Independent
    """

    @staticmethod
    def create_factory(q):
        return SimplePolar2D(None, q)

    def create(self, obs):
        return SimplePolar2D(obs, self.q)

    def __init__(self, obs, q):
        
        if not obs:
            self.q = q
            return

        dT = obs.sensor.param["time_interval"]
        
        x = np.array(
            [obs.y[0], obs.y[1], 0.0, 0.0]
        )

        P = block_diag(obs.R, np.eye(2) * 0.1)
        
        F = np.array(
            [[1.0, 0.0, dT, 0.0],
            [0.0, 1.0, 0.0, dT],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
        )

        H = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0]]
        )

        Q = np.array(
            [[dT**3/3, 0, dT**2/2, 0],
            [0, dT**3/3, 0, dT**2/2],
            [dT**2/2, 0, dT, 0],
            [0, dT**2/2, 0, dT]]
        ) * q
        
        super().__init__(x, F, H, P, Q, is_nonlinear=False)