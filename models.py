import numpy as np
from scipy.linalg import block_diag

import sensors


def calc_ellipsoidal_gate(model, obs):
    """Calc Ellipsoidal Gate

        ref) Design and Analysis of Modern Tracking Systems
                    6.3.2 Ellipsoidal Gates

    Arguments:
        model {KalmanModel} -- Kalman Model
        obs {Obs} -- Observation
    """
    dy, S = model.residual(obs)
    detS = np.linalg.det(S)
    PD = obs.sensor.param["PD"]
    COEF = (2*np.pi) ** (0.5*len(obs.y))
    BETA = obs.sensor.param["BNT"] + obs.sensor.param["PFA"] / obs.sensor.param["VC"]
    GATE = 2 * np.log( PD/(1-PD)/COEF/BETA/np.sqrt(detS) )
    return (GATE, dy @ np.linalg.inv(S) @ dy, detS)


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



class KalmanModel():
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

    def gaussian_log_likelihood(self, obs):
        """ Gaussian log Likelihood Function
        
        ref) Design and Analysis of Modern Tracking Systems
                    6.6.2 Extension to JPDA
        """
        dy, S = self.residual(obs)
        log_gij = -0.5 * dy @ np.linalg.inv(S) @ dy
        log_gij += -0.5 * len(obs.y) * np.log(2*np.pi)
        log_gij += -np.log( np.sqrt(np.linalg.det(S)) )
        return log_gij

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



class PDAKalmanModel(KalmanModel):
    """ PDA Kalman Model
    
        ref) Design and Analysis of Modern Tracking Systems
                    6.6.1 The PDA Method
                    6.6.2 Extension to JPDA
    """
    def update(self, obs_dict):
        if obs_dict:

            dyy = np.zeros( self.y_dim ).reshape(1,-1)
            dYY = np.zeros( (self.y_dim, self.y_dim) )
            R = np.zeros( (self.y_dim, self.y_dim) )

            for obs, ratio in obs_dict.items():
                if obs:
                    dy, _ = self.residual(obs)
                    dyy += ratio * dy
                    dYY += ratio * dy.T @ dy
                    R += ratio * obs.R
                else:
                    ratio_0 = ratio

            self._inovate_step(R)
            P0 = ratio_0 * self.P + (1-ratio_0) * (self.P - self.K @ self.H * self.P)
            self.P = P0 + self.K @ (dYY - dyy.T @ dyy ) @ self.K.T
            self.x = self.x + self.K @ dyy.T
            self._predict_step()

        else:
            self._predict_step()



class Simple2DModelFactory():
    """ Simple 2D Model Factory
    
     * Linear Kalman Filter
     * Constant Velocity Model
     * State Vector x is 2D posit/veloc
     * Input Vector y is 2D posit
     * 2D(x1,x2) is Independent

     ex)
      * Cart Model (cart-x, cart-y), x=(px, py, vx, vy), y=(px, py)
      * Polar Model (range, theta), x=(pr, pt, vr, vt), y=(pr, pt)
    """
    def __init__(self, model, q, pv):
        self.model = model
        self.q = q
        self.pv = pv

    def create(self, obs):
        dT = obs.sensor.param["dT"]
        
        x = np.array(
            [obs.y[0], obs.y[1], 0.0, 0.0]
        )

        P = block_diag(obs.R, np.eye(2) * self.pv)
        
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
        ) * self.q
        
        return self.model(x, F, H, P, Q, is_nonlinear=False)
