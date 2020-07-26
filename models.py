import copy
import enum
import numpy as np
from scipy.linalg import block_diag

import sensors
import utils

class CoordType(enum.Enum):
    CART = enum.auto() # 3.7.2 Cartesian Coordinate Tracking (NED system)
    POLAR = enum.auto() # 3.7.3 Spherical/Sensor Coordinate Tracking (RHV system)
    HTURN_VS = enum.auto() # 4.3.1 Horizontal Turn Model With Velocity States
    HTURN_NCS = enum.auto() # 4.3.3 Nearly Constasnt Speed Horizontal Turn Model
    OTHER = enum.auto() # ex) Cylinder Coord [R,AZIM,HEIGHT]


class ValueType(enum.Enum):
    # -- Kinematic Value
    # ---- cart system
    POSIT_X = enum.auto()
    POSIT_Y = enum.auto()
    POSIT_Z = enum.auto()
    VELOC_X = enum.auto()
    VELOC_Y = enum.auto()
    VELOC_Z = enum.auto()
    ACCEL_X = enum.auto()
    ACCEL_Y = enum.auto()
    ACCEL_Z = enum.auto()
    # ---- polar system
    POSIT_DIST = enum.auto()
    POSIT_AZIM = enum.auto()
    POSIT_ELEV = enum.auto()
    VELOC_DIST = enum.auto()
    VELOC_AZIM = enum.auto()
    VELOC_ELEV = enum.auto()
    ACCEL_DIST = enum.auto()
    ACCEL_AZIM = enum.auto()
    ACCEL_ELEV = enum.auto()
    # ---- other( horizontal turn etc. )
    OMEGA = enum.auto()
    SPEED = enum.auto()
    # -- Attribute Value
    INTENSITY = enum.auto()

    @staticmethod
    def generate_value_type(crd_type, SD, RD, is_vel_measure_enabled):
        if    crd_type == CoordType.CART:
            val_type = [
                ValueType.POSIT_X,
                ValueType.POSIT_Y,
                ValueType.POSIT_Z,
                ValueType.VELOC_X,
                ValueType.VELOC_Y,
                ValueType.VELOC_Z,
                ValueType.ACCEL_X,
                ValueType.ACCEL_Y,
                ValueType.ACCEL_Z,
            ]
        elif crd_type == CoordType.POLAR:
            val_type = [
                ValueType.POSIT_DIST,
                ValueType.POSIT_AZIM,
                ValueType.POSIT_ELEV,
                ValueType.VELOC_DIST,
                ValueType.VELOC_AZIM,
                ValueType.VELOC_ELEV,
                ValueType.ACCEL_DIST,
                ValueType.ACCEL_AZIM,
                ValueType.ACCEL_ELEV,
            ]
        else:
            raise NotImplementedError

        idx = np.zeros(9, dtype=bool)
        for i in range(3):
            for j in range(3):
                if j < SD:
                    if i < RD:
                        idx[3*i+j] = True

        if is_vel_measure_enabled:
            idx[SD] = True

        val_type = [ xt for i,xt in enumerate(val_type) if idx[i] ]

        return val_type


class ModelType:
    """ Model Type """
    def __init__(self, crd_type, val_type, SD, RD):
        self.crd_type = crd_type
        self.val_type = val_type
        self.SD = SD
        self.RD = RD

    def extract_x(self, x, val_type):
        idx = np.array([ self.val_type.index(vt) for vt in val_type ])

        for i in range(len(x.shape)):

            if   i==0:
                if len(x.shape) == 1: # vector
                    x = x[idx]
                elif len(x.shape) == 2: # matrix
                    x = x[idx,:]
                else:
                    assert False, "x.shape invalid, actual:" + str(x.shape)
            
            elif i==1:
                if len(x.shape) == 2: # matrix
                    x = x[:,idx]
                else:
                    assert False, "x.shape invalid, actual:" + str(x.shape)

        return x

    @staticmethod
    def generate_model_type(crd_type, SD, RD, is_vel_measure_enabled=False):
        val_type = ValueType.generate_value_type(crd_type, SD, RD, is_vel_measure_enabled)
        mdl_type = ModelType(crd_type, val_type, SD, RD)
        return mdl_type


class Obs():
    """ Observation """

    def __init__(self, y, R, sensor=None):
        """Initialize Observation
        
        Arguments:
            y {np.ndarray} -- sensor value vector
            R {np.ndarray} -- sensor value variance matrix
            sensor {Sensor} -- sensor
        """
        assert isinstance(y, np.ndarray) or isinstance(y, float) ,type(y)
        assert isinstance(R, np.ndarray) or isinstance(R, float), type(R)
        assert isinstance(sensor, sensors.BaseSensor) or sensor is None, type(sensor)
        self.y = y
        self.R = R
        self.sensor = sensor



class KalmanModel():
    """ Kalman Model
    
        ref) Design and Analysis of Modern Tracking Systems
                    3.3 Kalman Filtering
                    3.4 Extended Kalman Filtering
    """
    def __init__(self, x, F, H, P, Q, is_nonlinear ,x_type=None, y_type=None):
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
        self._x_type = x_type
        self._y_type = y_type
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
        """ Residual Vector

        dy = y - H*x
        S = H*P*H' + R
        """
        if self.is_nonlinear:
            self.H = self._update_H(self.x)
        dy = obs.y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + obs.R
        return (dy, S)

    def norm_of_residual(self, obs):
        """ Norm(d^2) of Residual Vector
        d^2 = dy S^-1 dy
        """
        dy, S = self.residual(obs)
        dist = dy @ np.linalg.inv(S) @ dy
        detS = np.linalg.det(S)
        M = len(dy)
        return (dist, detS, M)

    def gaussian_log_likelihood(self, obs):
        """ Gaussian log Likelihood Function
        
        ref) Design and Analysis of Modern Tracking Systems
                    6.6.2 Extension to JPDA
        """
        dist, detS, _ = self.norm_of_residual(obs)
        log_gij  = -0.5 * dist - np.log( np.sqrt(detS))
        log_gij += -0.5 * len(obs.y) * np.log(2*np.pi)
        return log_gij
    
    def volume_of_ellipsoidal_gate(self, obs, gate):
        """Calc Ellipsoidal Gate Volume

        ref) Design and Analysis of Modern Tracking Systems
                    6.3.2 Ellipsoidal Gates

        Arguments:
            model {KalmanModel} -- Kalman Model
            obs {Obs} -- Observation
            gate {float} -- gate
        """
        _, S = self.residual(obs)

        if len(obs.y) == 1:
            CM = 2
        elif len(obs.y) == 2:
            CM = np.pi
        elif len(obs.y) == 3:
            CM = 4*np.pi/3
        elif len(obs.y) == 4:
            CM =   np.pi**2/2
        
        return CM * np.sqrt(np.linalg.det(S)) * gate ** (0.5 * len(obs.y))

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
        ratio_0 = 0.0
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
            self.x = self.x + self.K @ dyy.flatten()
            self._predict_step()

        else:
            self._predict_step()



class ModelFactory():
    """ Base ModelFactory """

    def __init__(self, model, dT, SD=2, RD=2, P0=np.array([0.]), is_polar=False, is_vel_measure_enabled=False):
        assert 1 <= SD <= 3
        assert 2 <= RD <= 3

        self.model = model
        self.dT=dT
        self.P0=P0
        self.SD=SD
        self.RD=RD
        self.XD=self.SD*self.RD
        self.YD=self.SD

        if is_polar:
            crd_type = CoordType.POLAR
        else:
            crd_type = CoordType.CART

        if is_vel_measure_enabled:
            self.YD += 1

        self._x_type = ModelType.generate_model_type(crd_type=crd_type,SD=self.SD,RD=self.RD)
        self._y_type = ModelType.generate_model_type(crd_type=crd_type,SD=self.SD,RD=1,is_vel_measure_enabled=is_vel_measure_enabled)

    def create(self, obs):
        raise NotImplementedError



class SimpleModelFactory(ModelFactory):
    """ Simple Model Factory
    
     * Linear Kalman Filter
     * Constant Velocity Model
     * State Vector x is posit/veloc(/accel)
     * Input Vector y is posit
     * (x1,x2,x3) is Independent
     * when is     vel_measure_enabled, y=(p_1, p_2, ..., p_SD, v_1)
     * when is not vel_measure_enabled, y=(p_1, p_2, ..., p_SD)

     ex)
      * Cart 2D Model (cart-x, cart-y), x=(px, py, vx, vy), y=(px, py)
      * Polar 2D Model (range, theta), x=(pr, pt, vr, vt), y=(pr, pt)
    """
    def __init__(self, model, dT, q, SD=2, RD=2, P0=np.array([0.]), is_polar=False, is_vel_measure_enabled=False):
        """
        Arguments:
        tm {float} -- target maneuver time constant
        sm {float} -- target maneuver starndard deviation
        P0 {np.array} -- initial covariance matrix
        SD {integer} -- space dimension
        RD {integer} -- rank dimension
        is_vel_measure_enabled {bool} -- velocity measurement is available or not
        
        example:
        SimpleModelFactory(
            model = models.KalmanModel
            q  = [0.1, 0.2], # each for SD1(=x), SD2(=y)
            P0 = block_diag(obs.R, np.eye(RD) * self.pv),
            SD = 2,
            RD = RD,
            is_polar = False
        )
        """
        super().__init__(model=model, dT=dT, SD=SD, RD=RD, P0=P0, is_polar=is_polar, is_vel_measure_enabled=is_vel_measure_enabled)

        if np.isscalar(q):
            q = [q]*SD
        else:
            assert len(q) == SD

        self.q = q

    def create(self, obs):
        assert obs.y.shape == (self.SD,), "obs.y.shape invalid, actual:" + str(obs.y.shape)
        assert obs.R.shape == (self.SD,self.SD), "obs.R.shape invalid, actual:" + str(obs.R.shape)
        
        dT = self.dT
        x = np.zeros(self.XD)
        x[0:self.YD] = obs.y

        H = np.zeros( (self.YD, self.XD) )
        H[0:self.YD, 0:self.YD] = np.eye(self.YD)

        F = np.array(
            [[1.0, dT , dT**2/2],
            [ 0.0, 1.0, dT],
            [ 0.0, 0.0, 1.0]]
        )
        F = F[3-self.RD:, 3-self.RD:]

        Q = np.array(
            [[dT**5/20, dT**4/8, dT**3/6],
            [ dT**4/8 , dT**3/3, dT**2/2],
            [ dT**3/6 , dT**2/2, dT]]
        )
        Q = Q[3-self.RD:, 3-self.RD:]

        F = utils.swap_block_matrix( block_diag( *tuple([F   for i   in     range(self.SD)        ]) ), self.SD )
        Q = utils.swap_block_matrix( block_diag( *tuple([Q*q for i,q in zip(range(self.SD),self.q)]) ), self.SD )

        P = np.zeros( (self.XD, self.XD) )
        R = np.zeros( (self.XD, self.XD) )
        P[:len(self.P0), :len(self.P0)] = self.P0
        R[:self.YD, :self.YD] = obs.R
        P = np.maximum(P, Q)
        P = np.maximum(P, R)
        
        return self.model(x, F, H, P, Q, is_nonlinear=False, x_type=self._x_type, y_type=self._y_type)



class SingerModelFactory(ModelFactory):
    """ Singer Model Factory
    
     * Linear Kalman Filter
     * Multi Space Dimension (SD = 1D~3D supported)
     * Singer Acceleration Model
     * State Vector x is posit/veloc/accel (XD=3) for each Direction (SD=Multi)
     * Input Vector y is posit (YD=1) for each Direction (SD=Multi) without velocity measurement
     * Input Vector y is posit/veloc (YD=2) for each Direction (SD=Multi) with velocity measurement
     * (x1,x2, ..., xX) is Independent
     * when is     vel_measure_enabled, y=(p_1, p_2, ..., p_SD, v_1)
     * when is not vel_measure_enabled, y=(p_1, p_2, ..., p_SD)

     ex)
      * Cart 2D Model (cart-x, cart-y), x=(px, py, vx, vy, ax, ay), y=(px, py)
      * Polar 2D Model (range, theta), x=(pr, pt, vr, vt, ar, at), y=(pr, pt)

    ref) Design and Analysis of Modern Tracking Systems
        4.2.1 Singer Acceleration Model
    """
    def __init__(self, model, dT, tm, sm, SD=2, P0=np.array([0.]), is_polar=False, is_vel_measure_enabled=False):
        """
        Arguments:
        tm {float} -- target maneuver time constant
        sm {float} -- target maneuver starndard deviation
        P0 {np.array} -- initial covariance matrix
        SD {integer} -- space dimension
        is_vel_measure_enabled {bool} -- velocity measurement is available or not

        example:
        SingerModelFactory(
            model = models.KalmanModel
            tm = [10.0, 4.0], # each for SD1(=x), SD2(=y)
            sm = 15.0,        # if scalar, sm_SD1 = sm_SD2
            P0 = block_diag(obs.R, np.eye(XD) * self.pv),
            SD = 2,
            is_vel_measure_enabled = False
        )
        """
        super().__init__(model=model, dT=dT, SD=SD, RD=3, P0=P0, is_polar=is_polar, is_vel_measure_enabled=is_vel_measure_enabled)

        if np.isscalar(tm):
            tm = [tm]*SD
        else:
            assert len(tm) == SD

        if np.isscalar(sm):
            sm = [sm]*SD
        else:
            assert len(sm) == SD

        self.tm = tm
        self.sm = sm

    def create(self, obs):
        assert obs.y.shape == (self.YD,), "obs.y.shape invalid, actual:" + str(obs.y.shape)
        assert obs.R.shape == (self.YD,self.YD), "obs.R.shape invalid, actual:" + str(obs.R.shape)

        dT = self.dT
        x = np.zeros(self.XD)
        x[0:self.YD] = obs.y

        H = np.zeros( (self.YD, self.XD) )
        H[0:self.YD, 0:self.YD] = np.eye(self.YD)

        F_list = []
        Q_list = []

        for tm, sm in zip(self.tm, self.sm):
            beta = 1/tm
            rm = np.exp(-beta * dT)

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

            F_list.append(F)
            Q_list.append(Q)

        F = utils.swap_block_matrix( block_diag( *tuple(F_list) ), self.SD )
        Q = utils.swap_block_matrix( block_diag( *tuple(Q_list) ), self.SD )

        P = np.zeros( (self.XD, self.XD) )
        R = np.zeros( (self.XD, self.XD) )
        P[:len(self.P0), :len(self.P0)] = self.P0
        R[:self.YD, :self.YD] = obs.R
        P = np.maximum(P, Q)
        P = np.maximum(P, R)
        
        return self.model(x, F, H, P, Q, is_nonlinear=False, x_type=self._x_type, y_type=self._y_type)


class Target():
    """ Base Target """

    def __init__(self, SD=2, RD=3, x0=np.array([0., 0.]), start_time=0, end_time=np.inf):
        assert 1 <= SD <= 3
        assert 2 <= RD <= 3

        self.SD = SD
        self.RD = RD
        self.XD = self.SD*self.RD

        self.x = np.zeros(self.XD)
        self.x[:len(x0)] = x0
        self._x_type = ModelType.generate_model_type(crd_type=CoordType.CART,SD=self.SD,RD=self.RD)

        self.current_time = 0
        self.start_time = start_time
        self.end_time = end_time

    def update_x(self, T, dT):
        raise NotImplementedError

    def update(self, dT):
        self.current_time += dT
        self.update_x(self.current_time-self.start_time, dT)

    def is_exist(self):
        return self.start_time <= self.current_time < self.end_time
    
    def calc_match_price(self, model):
        gate = 13.3
        x_tgt, x_tgt_type = self.convert_x_into(mdl_type=model._x_type)
        x_common_type = list(set(x_tgt_type.val_type) & set(model._x_type.val_type))
        dx = model._x_type.extract_x(model.x, x_common_type) - x_tgt_type.extract_x(x_tgt, x_common_type)
        P  = model._x_type.extract_x(model.P, x_common_type)
        dist = dx @ np.linalg.inv(P) @ dx
        return gate - dist

    def is_in_gate(self, model):
        return self.calc_match_price(model) > 0
    
    def convert_x_into(self, mdl_type):
        """ convert target's x_type into model's x_type

        target's x_type is cartesian, so
        if model's x_type is cartesian, no conversion
        if model's x_type is polar, convert into polar
        """

        if   mdl_type.crd_type == CoordType.CART:
            x = self.x
            x_type = self._x_type

        elif mdl_type.crd_type == CoordType.POLAR:
            if 1 <= self.SD <= 2:
                x = np.zeros((self.RD, 3))
                x[:self.RD,:self.SD] = self.x.reshape((self.RD, self.SD))
                x = x.flatten()
            elif self.SD == 3:
                x = self.x
            else:
                assert False, "self.SD invalid, actual:" + str(self.SD)

            x = utils.cart2polar(x)
            x_type = ModelType.generate_model_type(crd_type=CoordType.POLAR,SD=3,RD=3)

        else:
            raise NotImplementedError
        
        return (x, x_type)


class SimpleTarget(Target):
    """ Simple Target Maneuver Model (1D~3D, posit/veloc/accel)

        default:
        * Cart 2D Model (cart-x, cart-y), x=(px, py, vx, vy)
    """

    def __init__(self, SD=2, x0=np.array([0.,0.]), start_time=0, end_time=np.inf):
        """
        Arguments:
        x0  {np.array} -- state vector initial value
        """
        super().__init__(SD=SD, RD=3, x0=x0, start_time=start_time, end_time=end_time)

    def update_x(self, T, dT):
        """
        Arguments:
        dT {float} -- sampling interval
        """
        F = np.array(
            [
                [1.0, dT , 0.5*dT**2],
                [0.0, 1.0, 1.0*dT**1],
                [0.0, 0.0, 1.0],
            ]
        )
        F = utils.swap_block_matrix( block_diag( *tuple([F for i in range(self.SD)]) ), self.SD )
        self.x = F @ self.x



class SingerTarget(Target):
    """ Singer Target Maneuver Model (1D, posit/veloc/accel)
    
    ref) Design and Analysis of Modern Tracking Systems
        4.2.1 Singer Acceleration Model
    """

    def __init__(self, tm, sm, SD=2, x0=np.array([0.,0.]), start_time=0, end_time=np.inf):
        """
        Arguments:
        tm {float} -- target maneuver time constant
        sm {float} -- target maneuver starndard deviation
        x0  {np.array} -- state vector initial value
        """
        super().__init__(SD=SD, RD=3, x0=x0, start_time=start_time, end_time=end_time)

        if np.isscalar(tm):
            tm = [tm]*SD
        else:
            assert len(tm) == SD

        if np.isscalar(sm):
            sm = [sm]*SD
        else:
            assert len(sm) == SD

        self.tm = tm
        self.sm = sm

    def update_x(self, T, dT):
        """
        Arguments:
        dT {float} -- sampling interval
        """
        F_list = []

        for tm, sm in zip(self.tm, self.sm):
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

            F_list.append(F)

        F = utils.swap_block_matrix( block_diag( *tuple(F_list) ), self.SD )
        self.x = F @ self.x

        beta = 1/np.array(self.tm)
        rm = np.exp(-beta * dT)
        self.x[-self.SD:] += np.sqrt(1-rm**2) * np.random.normal(0.0, np.array(self.sm))



class ModelEvaluator():
    """ Evaluate Each Model
    
    * Estimate Prediction Errors of posit/veloc/accel etc.
    """

    def __init__(self, sensor, model_factory, target, R):
        """
        Arguments:
        sensor {Sensor} -- sensor
        model_factory {ModelFactory} -- model factory
        target {Target} -- target
        R {np.array} -- observation error covariance matrix
        """
        self.model_factory = model_factory
        self.target = target
        self.R = R
        self.sensor = sensor

    def _dT(self):
        return self.model_factory.dT

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
        tgt.update(self._dT())

        # simulate
        count = 1000
        while count>0:
            self.tgt_list.append(copy.deepcopy(tgt))
            self.obs_list.append(copy.deepcopy(obs))
            self.mdl_list.append(copy.deepcopy(mdl))

            mdl.update(obs)
            tgt.update(self._dT())
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

            tgt.update(self._dT())
            obs = Obs(np.random.multivariate_normal(tgt.x[:len(self.R)], self.R), self.R, self.sensor)
            mdl.update(obs)

            count -= 1
        
        return self._calc_RMSE(False)