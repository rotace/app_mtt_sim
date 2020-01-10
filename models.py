import copy
import numpy as np
from scipy.linalg import block_diag

import sensors
import utils


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


def calc_ellipsoidal_gate_volume(model, obs, gate):
    """Calc Ellipsoidal Gate Volume

        ref) Design and Analysis of Modern Tracking Systems
                    6.3.2 Ellipsoidal Gates

    Arguments:
        model {KalmanModel} -- Kalman Model
        obs {Obs} -- Observation
        gate {float} -- gate
    """
    # detS is not used obs
    _, _, detS = calc_ellipsoidal_gate(model, obs)

    if len(obs.y) == 1:
        CM = 2
    elif len(obs.y) == 2:
        CM = np.pi
    elif len(obs.y) == 3:
        CM = 4*np.pi/3
    elif len(obs.y) == 4:
        CM =   np.pi**2/2
    
    return CM * np.sqrt(detS) * gate ** (0.5 * len(obs.y))


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
        self.P = self.P - self.K @ self.H @ self.P
        self.x = self.x + self.K @ (y-self.H @ self.x)

    def _predict_step(self):
        """Predict Step
        
         P = F*P*F' + Q
         x = F*x
        """
        # register current state
        self.x_current = self.x
        # predict next state
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



class ModelFactory():
    """ Base ModelFactory """

    def __init__(self):
        """Initialize Target
        """
        pass

    def create(self, obs):
        raise NotImplementedError



class Simple2DModelFactory(ModelFactory):
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
        self.SD=2
        self.XD=2
        self.YD=1

    def create(self, obs):
        assert obs.y.shape == (self.SD,), "obs.y.shape invalid, actual:" + str(obs.y.shape)
        assert obs.R.shape == (self.SD,self.SD), "obs.R.shape invalid, actual:" + str(obs.R.shape)
        
        dT = obs.sensor.param["dT"]
        
        x = np.zeros(self.SD*self.XD)
        x[0:self.SD] = obs.y

        P = block_diag(obs.R, np.eye(2) * self.pv)

        H = np.zeros( (self.SD, self.SD*self.XD) )
        H[0:self.SD, 0:self.SD] = np.eye(self.SD)

        F = np.array(
            [[1.0, dT],
            [0.0, 1.0]],
        )

        Q = np.array(
            [[dT**3/3, dT**2/2],
            [dT**2/2, dT]]
        ) * self.q

        F = utils.swap_block_matrix( block_diag( *tuple([F for i in range(self.SD)]) ), self.SD )
        Q = utils.swap_block_matrix( block_diag( *tuple([Q for i in range(self.SD)]) ), self.SD )
        
        return self.model(x, F, H, P, Q, is_nonlinear=False)



class SingerModelFactory(ModelFactory):
    """ Singer Model Factory
    
     * Linear Kalman Filter
     * Multi Space Dimension (SD = 1D~3D supported)
     * Singer Acceleration Model
     * State Vector x is posit/veloc/accel (XD=3) for each Direction (SD=Multi)
     * Input Vector y is posit (YD=1) for each Direction (SD=Multi) without velocity measurement
     * Input Vector y is posit/veloc (YD=2) for each Direction (SD=Multi) with velocity measurement
     * (x1,x2, ..., xX) is Independent

     ex)
      * Cart 2D Model (cart-x, cart-y), x=(px, py, vx, vy, ax, ay), y=(px, py)
      * Polar 2D Model (range, theta), x=(pr, pt, vr, vt, ar, at), y=(pr, pt)

    ref) Design and Analysis of Modern Tracking Systems
        4.2.1 Singer Acceleration Model
    """
    def __init__(self, model, tm, sm, SD=2, is_vel_measure_enabled=False):
        """
        Arguments:
        tm {float} -- target maneuver time constant
        sm {float} -- target maneuver starndard deviation
        SD {integer} -- space dimension
        is_vel_measure_enabled {bool} -- velocity measurement is available or not
        """
        self.model = model
        self.tm = tm
        self.sm = sm
        self.SD = SD
        self.XD = 3
        if is_vel_measure_enabled:
            self.YD = 2
        else:
            self.YD = 1

    def create(self, obs):
        assert obs.y.shape == (self.SD*self.YD,), "obs.y.shape invalid, actual:" + str(obs.y.shape)
        assert obs.R.shape == (self.SD*self.YD,self.SD*self.YD), "obs.R.shape invalid, actual:" + str(obs.R.shape)

        dT = obs.sensor.param["dT"]
        tm = self.tm
        sm = self.sm
        beta = 1/tm
        rm = np.exp(-beta * dT)
        
        x = np.zeros(self.SD*self.XD)
        x[0:self.SD*self.YD] = obs.y

        P = np.zeros( (self.SD*self.XD, self.SD*self.XD) )
        P[0:self.SD*self.YD, 0:self.SD*self.YD] = obs.R

        H = np.zeros( (self.SD*self.YD, self.SD*self.XD) )
        H[0:self.SD*self.YD, 0:self.SD*self.YD] = np.eye(self.SD*self.YD)

        # H = np.zeros( (self.YD, self.XD) )
        # H[0:self.YD, 0:self.YD] = np.eye(self.YD)
        # H = utils.swap_block_matrix( block_diag( *tuple([H for i in range(self.SD)]) ), self.SD )

        F = np.array(
            [[1.0, dT, 1/beta/beta*(-1+beta*dT+rm)],
            [0.0, 1.0, 1/beta*(1-rm)],
            [0.0, 0.0, rm]]
        )

        q11 = 0.5/beta**5 * (+1 - rm**2 - 4*beta*dT*rm + 2*beta*dT - 2*beta**2*dT**2 + 2*beta**3*dT**3/3 )
        q12 = 0.5/beta**4 * (+1 + rm**2 + 2*beta*dT*rm - 2*beta*dT + 1*beta**2*dT**2 - 2*rm)
        q13 = 0.5/beta**3 * (+1 - rm**2 - 2*beta*dT*rm)
        q22 = 0.5/beta**3 * (-3 - rm**2 + 4*rm + 2*beta*dT)
        q23 = 0.5/beta**2 * (+1 + rm**2 - 2*rm)
        q33 = 0.5/beta**1 * (+1 - rm**2)

        Q = np.array(
            [[q11, q12, q13],
            [q12, q22, q23],
            [q13, q23, q33]]
        ) * 2*sm**2/tm

        F = utils.swap_block_matrix( block_diag( *tuple([F for i in range(self.SD)]) ), self.SD )
        Q = utils.swap_block_matrix( block_diag( *tuple([Q for i in range(self.SD)]) ), self.SD )
        
        return self.model(x, F, H, P, Q, is_nonlinear=False)


class Target():
    """ Base Target """

    def __init__(self):
        """Initialize Target
        """
        pass

    def update(self):
        raise NotImplementedError


class SingerTarget(Target):
    """ Singer Target Maneuver Model (1D, posit/veloc/accel)
    
    ref) Design and Analysis of Modern Tracking Systems
        4.2.1 Singer Acceleration Model
    """

    def __init__(self, tm, sm, x_init=np.array([0.,0.,0.])):
        """
        Arguments:
        tm {float} -- target maneuver time constant
        sm {float} -- target maneuver starndard deviation
        dT {float} -- sampling interval
        x_init  {np.array} -- state vector initial value
        """
        super().__init__()

        self.tm = tm
        self.sm = sm
        self.x  = x_init

    def update(self, dT):
        """
        Arguments:
        dT {float} -- sampling interval
        """
        tm = self.tm
        sm = self.sm
        beta = 1/tm
        rm = np.exp(-beta * dT)

        F = np.array(
            [
                [1.0, dT , 0.5*dT**2],
                [0.0, 1.0, 1.0*dT**1],
                [0.0, 0.0, rm],
            ]
        )

        # F = np.array(
        #     [
        #         [1.0,  dT, 1/beta/beta*(-1+beta*dT+rm)],
        #         [0.0, 1.0, 1/beta*(1-rm)],
        #         [0.0, 0.0, rm],
        #     ]
        # )

        self.x = F @ self.x
        self.x[2] += np.sqrt(1-rm**2) * np.random.normal(0.0, sm)



class ModelEvaluator():
    """ Evaluate Each Model
    
    * Estimate Prediction Errors of posit/veloc/accel etc.
    """

    def __init__(self, model_factory, target, R, sensor):
        """
        Arguments:
        model_factory {ModelFactory} -- model factory
        target {Target} -- target
        R {np.array} -- observation error covariance matrix
        sensor {Sensor} -- sensor
        """
        self.model_factory = model_factory
        self.target = target
        self.R = R
        self.sensor = sensor

    def _initialize_simulation(self):
        self.tgt_list = []
        self.obs_list = []
        self.mdl_list = []
        tgt = self.target
        obs = Obs(np.random.multivariate_normal(tgt.x[:len(self.R)], self.R), self.R, self.sensor)
        mdl = self.model_factory.create(obs)
        assert tgt.x.shape == mdl.x.shape, "target.x.shape and model.x.shape invalid, actual:" + str(tgt.x.shape) + str(mdl.x.shape)
        return (tgt, obs, mdl)

    def _calc_RMSE(self, is_prediction_error):
        # calc RMSE
        count = 0
        RMSE = np.zeros(self.target.x.shape)
        for mdl, tgt in zip(self.mdl_list, self.tgt_list):

            if is_prediction_error:
                RMSE += (tgt.x-mdl.x)**2
            else:
                RMSE += (tgt.x-mdl.x_current)**2
           
            count += 1
        RMSE = np.sqrt(RMSE/count)
        return RMSE

    def estimate_prediction_error(self):
        tgt, obs, mdl = self._initialize_simulation()

        # init for prediction
        tgt.update(self.sensor.param["dT"])

        # simulate
        count = 1000
        while count>0:
            self.tgt_list.append(copy.deepcopy(tgt))
            self.obs_list.append(copy.deepcopy(obs))
            self.mdl_list.append(copy.deepcopy(mdl))

            mdl.update(obs)
            tgt.update(self.sensor.param["dT"])
            obs = Obs(np.random.multivariate_normal(tgt.x[:len(self.R)], self.R), self.R, self.sensor)

            count -= 1
        
        return self._calc_RMSE(True)

    def estimate_current_error(self):
        tgt, obs, mdl = self._initialize_simulation()

        # simulate
        count = 1000
        while count>0:
            self.tgt_list.append(copy.deepcopy(tgt))
            self.obs_list.append(copy.deepcopy(obs))
            self.mdl_list.append(copy.deepcopy(mdl))

            tgt.update(self.sensor.param["dT"])
            obs = Obs(np.random.multivariate_normal(tgt.x[:len(self.R)], self.R), self.R, self.sensor)
            mdl.update(obs)

            count -= 1
        
        return self._calc_RMSE(False)