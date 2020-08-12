import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import models
import sensors

class BaseTrackFactory():
    """ Track Factory Base Class """
    def __init__(self, track, **kwargs):
        self.track = track
        self.param = kwargs

    def set_attr(self, **kwargs):
        if "model_factory" in kwargs:
            self.model_factory = kwargs["model_factory"]
        if "tracker" in kwargs:
            self.tracker = kwargs["tracker"]

    def create(self, obs):

        if hasattr(self, "tracker"):
            timestamp = self.tracker.timestamp
        else:
            timestamp = 0

        return self.track(
            obs,
            self.model_factory.create(obs),
            timestamp=timestamp,
            **self.param
        )



class BaseTrack(models.BaseExporter):
    """ Base Track

        ref) Design and Analysis of Modern Tracking Systems
                    6.2 Track Score Function
                    6.3 Gating
                    6.4 Global Nearest Neighbor Method

     * Use Kalman Filter
     * Use Log Likelihood Ratio as Score
    """
    trk_id_counter = 0
    @classmethod
    def _generate_id(cls):
        cls.trk_id_counter+=1
        return cls.trk_id_counter

    def __init__(self, obs, model, **kwargs):
        # set param
        if "gate" not in kwargs:
            kwargs["gate"] = None
        
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = 0

        self.param = kwargs
        self.sensor = obs.sensor
        # set data
        self.obs_list = [copy.deepcopy(obs)]
        self.scr_list = [self._calc_init_score(obs)]
        self.cnt_list = [self.param["timestamp"]]
        self.model = model
        self.mdl_list = [copy.deepcopy(self.model)]
        self.trk_id = BaseTrack._generate_id()
    
    def get_id(self):
        return self.trk_id
        
    def to_series(self, timestamp, scan_id):
        series = super().to_series(timestamp, scan_id)
        # integer
        trk_id = self.get_id()
        obs_id = self.obs_list[-1].get_id() if self.obs_list[-1] else -1
        value=[trk_id, obs_id] 
        label=["TRK_ID", "OBS_ID"]
        series = series.append( pd.Series(value, index=label) )
        # string
        value=[self.model._x_type.crd_type.name]
        label=["CRD_TYPE"]
        series = series.append( pd.Series(value, label) )
        # real
        x_val = list(self.model.x)
        x_lbl = [ v.name for v in self.model._x_type.val_type ]
        P_val = [ pij for i, pi in enumerate(self.model.P) for j, pij in enumerate(pi) if i<=j ]
        P_lbl = [ "P" + str(i) + str(j) for i in range(self.model.P.shape[0]) for j in range(self.model.P.shape[1]) if i<=j]
        return series.append( pd.Series(x_val+P_val, index=x_lbl+P_lbl, dtype=float) )

    @staticmethod
    def from_series(series):
        assert isinstance(series, pd.Series), "series is invalid, actual:"  + str(type(series))
        cov_type_str = [ cv for cv in series.index.values if cv[0] == "P" ]
        val_type_str = list(set(series.index.values) &  {vt.name for vt in models.ValueType})
        val_type = [ vt  for vt_str in val_type_str for vt in models.ValueType if vt_str == vt.name ]
        mdl_type = models.ModelType(series["CRD_TYPE"], val_type, None, None)
        x = series[val_type_str].values
        p = series[cov_type_str].values
        P = np.zeros((len(x), len(x)))
        idx=0
        for i in range(len(x)):
            for j in range(len(x)):
                if i<=j:
                    P[i,j] = p[idx]
                    idx += 1
        P = np.triu(P) + np.triu(P).T
        class DummyModel():
            def __init__(self, x, P, x_type):
                self.x = x
                self.P = P
                self._x_type = x_type
        class DummyTrack():
            def __init__(self, model, trk_id):
                self.model = model
                self.trk_id = trk_id
            def get_id(self):
                return self.trk_id
        return DummyTrack(model=DummyModel(x=x, P=P, x_type=mdl_type), trk_id=series["TRK_ID"])

    def assign(self, obs):
        # set data
        self.obs_list.append(copy.deepcopy(obs))
        self.mdl_list.append(copy.deepcopy(self.model))
        self.cnt_list.append(self.cnt_list[-1]+1)
        self.scr_list.append(self._calc_match_score(obs))
        # update model
        self.model.update(obs)

    def unassign(self, sensor=sensors.BaseSensor()):
        # set data
        self.obs_list.append(None)
        self.mdl_list.append(copy.deepcopy(self.model))
        self.cnt_list.append(self.cnt_list[-1]+1)
        if not sensor or not sensor.is_trk_in_range(self):
            self.scr_list.append(self._calc_not_in_range_score())
        else:
            self.scr_list.append(self._calc_miss_score())
        # update model
        self.model.update(None)

    def get_gate(self, obs):
        dist, detS, M = self.model.norm_of_residual(obs)
        gate = obs.sensor.calc_ellipsoidal_gate(detS, M)

        if self.param["gate"]:
            gate = self.param["gate"]
        return (gate, dist)

    def is_in_gate(self, obs):
        gate, dist = self.get_gate(obs)
        return gate > dist

    def calc_match_price(self, obs):
        dist, detS, M = self.model.norm_of_residual(obs)
        gate = obs.sensor.calc_ellipsoidal_gate(detS, M)
        if False:
            # p.339 version (using term for penalizing tracks with greater prediction uncertainty)
            # however, in some case, penalty term is too strong for associate new track with next obs
            # because new track has greater prediction uncertainty P0, esspecially velocity uncertainty.
            return gate - dist - np.log(detS)
        else:
            # p.340 version (simple)
            return gate - dist

    def judge_confirmation(self):
        raise NotImplementedError

    def judge_deletion(self):
        raise NotImplementedError

    def _calc_init_score(self, obs):
        return obs.sensor.calc_LLR0()

    def _calc_match_score(self, obs):
        log_gij = self.model.gaussian_log_likelihood(obs)
        return self.scr_list[-1] + obs.sensor.calc_match_dLLR(log_gij)

    def _calc_miss_score(self):
        return self.scr_list[-1] + self.sensor.calc_miss_dLLR()
    
    def _calc_not_in_range_score(self):
        return self.scr_list[-1]
  
    def plot_obs_list(self):
        plt.plot(
            [obs.y[0] if obs is not None else None for obs in self.obs_list ],
            [obs.y[1] if obs is not None else None for obs in self.obs_list ],
            marker="D", color="r", alpha=.5, linestyle="None"
        )

    def plot_scr_list(self):
        plt.plot(
            self.cnt_list,
            self.scr_list,
            marker="D", color="r", alpha=.5, linestyle="None"
        )
    
    def plot_mdl_list(self):
        plt.plot(
            [mdl.x[0] if mdl is not None else None for mdl in self.mdl_list ],
            [mdl.x[1] if mdl is not None else None for mdl in self.mdl_list ],
            marker="D", color="g", alpha=.5, linestyle="None"
        )


class SimpleManagedTrack(BaseTrack):
    """ Simple Managed Track

        ref) Design and Analysis of Modern Tracking Systems
                    6.2.4 Score-Based Track Confirmation and Deletion p.334
    """
    def __init__(self, obs, model_factory, **kwargs):
        super().__init__(obs, model_factory, **kwargs)
        
        # parameter for another deletion argorithm of p.334
        if "ND" not in self.param:
            self.param["ND"] = 5    # continuous miss number for delete

    def judge_confirmation(self):
        return True

    def judge_deletion(self):
        # another deletion argorithm of p.334
        ND = self.param["ND"]
        return self.obs_list[-ND:] == [None]*ND


class ScoreManagedTrack(BaseTrack):
    """ LLR Score Managed Track

        ref) Design and Analysis of Modern Tracking Systems
                    6.2.4 Score-Based Track Confirmation and Deletion p.333
    """
    def __init__(self, obs, model_factory, **kwargs):
        super().__init__(obs, model_factory, **kwargs)

        # parameter for confirmation and deletion argorithm of p.333
        if "PFD" not in self.param:
            self.param["PFD"] = 1.e-3
        if "alpha" not in self.param:
            self.param["alpha"] = 4/3600/1000
        if "beta" not in self.param:
            self.param["beta"] = 0.1

        self.param["THD"] = np.log( self.param["PFD"] )
        self.param["T1"] = np.log( self.param["beta"] / (1-self.param["alpha"]) )
        self.param["T2"] = np.log( (1-self.param["beta"]) / self.param["alpha"] )

    def judge_confirmation(self):
        # use score for confirmation
        return self.scr_list[-1] > self.param["T2"]

    def judge_deletion(self):
        # use score for deletion
        is_del_a = self.scr_list[-1] < self.param["T1"]
        is_del_b =  self.scr_list[-1] - max(self.scr_list) < self.param["THD"]
        return is_del_a or is_del_b


class PDATrack(BaseTrack):
    """ LLR Track for PDA

    ref) Design and Analysis of Modern Tracking Systems
                    6.6.1 The PDA Method
                    6.6.2 Extension to JPDA
                    6.6.4 PDA Track Initiation and Deletion
    """
    def __init__(self, obs, model_factory, **kwargs):
        super().__init__(obs, model_factory, **kwargs)
        self.pt_list = [0.5]

        # set param
        if "P22" not in self.param:
            self.param["P22"] = 0.98

    def assign(self, obs_dict):
        # track score
        score = 0.0
        for obs, ratio in obs_dict.items():
            if obs:
                score += ratio * (self._calc_match_score(obs) - self.scr_list[-1])
            else:
                score += ratio * (self._calc_miss_score() - self.scr_list[-1])

        # comfirmation and deletion parameter
        delta = 0.0
        count = 0
        for obs, ratio in obs_dict.items():
            if obs:
                count += 1
                gate, _ = self.get_gate(obs)
                vg = self.model.volume_of_ellipsoidal_gate( obs, gate )
                delta -= vg * np.exp( self.model.gaussian_log_likelihood(obs) )
            else:
                pass
        if count:
            delta /= count - self.sensor.param["PD"] * self.pt_list[-1]
            delta += 1.0
            delta *= self.sensor.param["PD"]
        else:
            delta = self.sensor.param["PD"]
        ptkk = (1-delta)/(1-delta*self.pt_list[-1])*self.pt_list[-1]
        p22 = self.param["P22"]

        #  # set data
        self.obs_list.append(obs_dict)
        self.cnt_list.append(self.cnt_list[-1]+1)
        self.scr_list.append(score)
        self.pt_list.append( p22*ptkk + (1-p22)*(1-ptkk) )
        
        # # update model
        self.model.update(obs_dict)

    def judge_confirmation(self):
        # use pt for confirmation
        return self.pt_list[-1] > 0.95

    def judge_deletion(self):
        # use pt for deletion
        return self.pt_list[-1] < 0.4

class MultiSensorScoreManagedTrack(BaseTrack):
    """ Multi Sensor LLR Track

        ref) Design and Analysis of Modern Tracking Systems
                    9.5 General Expression for Multisensor Data Association

     * Use Kalman Filter
     * Use Log Likelihood Ratio as Score
     * Observation to Track Association Based System
    """
    def calc_match_price(self, obs):
        raise NotImplementedError



class TrackEvaluator():
    """ Evaluate Each Track
    
    * Evaluate Gating, Conformation, Deletion etc.
    """
    
    def __init__(self, sensor, model_factory, track_factory, target, R):
        """
        Arguments:
        sensor {Sensor} -- sensor
        model_factory {ModelFactory} -- model factory
        track_factory {TrackFactory} -- track factory
        target {Target} -- target
        R {np.array} -- observation error covariance matrix
        """
        self.track_factory = track_factory
        self.track_factory.set_attr(model_factory=model_factory)
        self.target = target
        self.R = R
        self.sensor = sensor
    
    def _dT(self):
        return self.track_factory.model_factory.dT

    def _initialize_simulation(self):
        self.tgt_list = []
        self.obs_list = []
        self.trk_list = []
        tgt = self.target
        obs = models.Obs(np.random.multivariate_normal(tgt.x[:len(self.R)], self.R), self.R, self.sensor)
        trk = self.track_factory.create( obs )
        assert tgt.x.shape == trk.model.x.shape, "target.x.shape and model.x.shape invalid, actual:" + str(tgt.x.shape) + str(trk.model.x.shape)
        return (tgt, obs, trk)

    def _update(self, tgt, obs, trk):
        # update target
        tgt.update(self._dT())

        # update observation
        obs = models.Obs(np.random.multivariate_normal(tgt.x[:len(self.R)], self.R), self.R, self.sensor)

        dist, detS, M = trk.model.norm_of_residual(obs)
        gate = obs.sensor.calc_ellipsoidal_gate(detS, M)
        mch_scr = trk._calc_match_score(obs)
        ini_scr = trk._calc_init_score(obs)

        # track update
        trk.assign(obs)
        return (tgt, obs, trk, dict(dist=dist, gate=gate, mch_scr=mch_scr, ini_scr=ini_scr))

    def plot_score(self, n_count=10):
        # init
        tgt, obs, trk = self._initialize_simulation()
        data_tbl = []

        # simulate
        i_count = n_count
        while i_count>0:
            
            tgt, obs, trk, data = self._update(tgt, obs, trk)
            data_tbl.append( data )
            i_count -= 1

        dist_list = [ data.get("dist") for data in data_tbl ]
        gate_list = [ data.get("gate") for data in data_tbl ]
        mch_scr_list = [ data.get("mch_scr") for data in data_tbl ]
        ini_scr_list = [ data.get("ini_scr") for data in data_tbl ]

        _, (axU, axD) = plt.subplots(nrows=2, sharex=True)

        axU.plot(dist_list, marker="D", label="dist")
        axU.plot(gate_list, marker="D", label="gate")
        axU.legend()

        axD.plot(mch_scr_list, marker="D", label="match_score")
        axD.plot(ini_scr_list, marker="D", label="init_score")
        axD.legend()

        plt.show()

    # def plot_stat_score(self, n_count=10, n_run=10):
    #     data_tbl_runs = []
    #     i_run = n_run
    #     while i_run>0:

    #         # init
    #         tgt, obs, trk = self._initialize_simulation()
    #         data_tbl = []

    #         # simulate
    #         i_count = n_count
    #         while i_count>0:
                
    #             tgt, obs, trk, data = self._update(tgt, obs, trk)
    #             data_tbl.append( data )
    #             i_count -= 1

    #         data_tbl_runs.append( data_tbl )
    #         i_run -= 1

    #     # calc average
    #     dist_list=[]
    #     gate_list=[]
    #     mch_scr_list=[]
    #     ini_scr_list=[]
    #     for ii in range(n_count):
    #         dist_list.append( sum([data_tbl[ii].get("dist") for data_tbl in data_tbl_runs])/n_run )
    #         gate_list.append( sum([data_tbl[ii].get("gate") for data_tbl in data_tbl_runs])/n_run )
    #         mch_scr_list.append( sum([data_tbl[ii].get("mch_scr") for data_tbl in data_tbl_runs])/n_run )
    #         ini_scr_list.append( sum([data_tbl[ii].get("ini_scr") for data_tbl in data_tbl_runs])/n_run )

    #     _, (axU, axD) = plt.subplots(nrows=2, sharex=True)

    #     axU.plot(dist_list, marker="D", label="dist")
    #     axU.plot(gate_list, marker="D", label="gate")
    #     axU.legend()

    #     axD.plot(mch_scr_list, marker="D", label="match_score")
    #     axD.plot(ini_scr_list, marker="D", label="init_score")
    #     axD.legend()

    #     plt.show()
