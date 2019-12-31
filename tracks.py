import copy
import numpy as np
import matplotlib.pyplot as plt

import models


class BaseTrackFactory():
    """ Track Factory Base Class """
    def __init__(self, track, **kwargs):
        self.track = track
        self.param = kwargs

    def create(self, obs, tracker):
        return self.track(obs, tracker, **self.param)

    def calc_init_score(self, obs):
        return self.track.calc_init_score(obs)



class BaseTrack():
    """ Track Base Class """

    def __init__(self, obs, tracker, **kwargs):
        # set param
        if "gate" not in kwargs:
            kwargs["gate"] = None

        self.param = kwargs
        self.tracker = tracker
        # set data
        self.obs_list = [copy.deepcopy(obs)]
        # self.mdl_list = [None]
        self.cnt_list = [self.tracker.count]
        # create model
        self.model = self.tracker.model_factory.create(obs)
        self.mdl_list = [copy.deepcopy(self.model)]

    def assign(self, obs):
        # set data
        self.obs_list.append(copy.deepcopy(obs))
        self.mdl_list.append(copy.deepcopy(self.model))
        self.cnt_list.append(self.tracker.count)
        # update model
        self.model.update(obs)

    def unassign(self):
        # set data
        self.obs_list.append(None)
        self.mdl_list.append(copy.deepcopy(self.model))
        self.cnt_list.append(self.tracker.count)
        # update model
        self.model.update(None)

    def is_in_gate(self, obs):
        gate, norm_dist, _ = models.calc_ellipsoidal_gate(self.model, obs)
        if self.param["gate"]:
            gate = self.param["gate"]
        return gate > norm_dist

    def calc_match_score(self, obs):
        raise NotImplementedError

    def judge_confirmation(self):
        raise NotImplementedError

    def judge_deletion(self):
        raise NotImplementedError

    @staticmethod
    def calc_init_score(obs):
        raise NotImplementedError
  
    def plot_obs_list(self):
        plt.plot(
            [obs.y[0] if obs is not None else None for obs in self.obs_list ],
            [obs.y[1] if obs is not None else None for obs in self.obs_list ],
            marker="D", color="r", alpha=.5, linestyle="None"
        )
    
    def plot_mdl_list(self):
        plt.plot(
            [mdl.x[0] if mdl is not None else None for mdl in self.mdl_list ],
            [mdl.x[1] if mdl is not None else None for mdl in self.mdl_list ],
            marker="D", color="g", alpha=.5, linestyle="None"
        )

    def plot_gate(self):
        gate_list = [
            models.calc_ellipsoidal_gate(mdl, obs)
            if obs is not None else (None, None, None)
            for mdl, obs in zip(self.mdl_list, self.obs_list)
        ]
        plt.plot(
            self.cnt_list,
            [ gate for gate, dist, detS in gate_list ],
            marker="D", color="r", alpha=.5, linestyle="None"
        )
        plt.plot(
            self.cnt_list,
            [ dist  for gate, dist, detS in gate_list ],
            marker="D", color="g", alpha=.5, linestyle="None"
        )


class DistTrack(BaseTrack):
    """ Distance Scored Track

        ref) Design and Analysis of Modern Tracking Systems
                    6.4 Global Nearest Neighbor Method

     * Use Kalman Filter
     * Use Generalized Distance as Score
    """
    def __init__(self, obs, tracker, **kwargs):
        super().__init__(obs, tracker, **kwargs)
        
        # set param
        if "ND" not in self.param:
            self.param["ND"] = 5    # continuous miss number for delete

    def calc_match_score(self, obs):
        gate, norm_dist, detS = models.calc_ellipsoidal_gate(self.model, obs)
        return gate - norm_dist - np.log(detS)

    def judge_confirmation(self):
        return True

    def judge_deletion(self):
        ND = self.param["ND"]
        return self.obs_list[-ND:] == [None]*ND

    @staticmethod
    def calc_init_score(obs):
        return 0



class LLRTrack(BaseTrack):
    """ LLR Scored Track

        ref) Design and Analysis of Modern Tracking Systems
                    6.2 Track Score Function

     * Use Kalman Filter
     * Use Log Likelihood Ratio as Score
    """
    def __init__(self, obs, tracker, **kwargs):
        super().__init__(obs, tracker, **kwargs)
        self.scr_list = [self.calc_init_score(obs)]
        
        # set param
        if "PFD" not in self.param:
            self.param["PFD"] = 1.e-3
        if "alpha" not in self.param:
            self.param["alpha"] = 4/3600/1000
        if "beta" not in self.param:
            self.param["beta"] = 0.1

        self.param["THD"] = np.log( self.param["PFD"] )
        self.param["T1"] = np.log( self.param["beta"] / (1-self.param["alpha"]) )
        self.param["T2"] = np.log( (1-self.param["beta"]) / self.param["alpha"] )

    def assign(self, obs):
        self.scr_list.append(self.calc_match_score(obs))
        super().assign(obs)

    def unassign(self):
        self.scr_list.append(self._calc_miss_score())
        super().unassign()

    def calc_match_score(self, obs):
        dLk = np.log( obs.sensor.param["VC"] ) + self.model.gaussian_log_likelihood(obs)
        dLs = np.log( obs.sensor.param["PD"] / obs.sensor.param["PFA"] )
        return self.scr_list[-1] + dLk + dLs

    def _calc_miss_score(self):
        dLk = 0
        dLs = np.log( 1.0 - self.tracker.sensor.param["PD"] )
        return self.scr_list[-1] + dLk + dLs

    def judge_confirmation(self):
        return self.scr_list[-1] > self.param["T2"]

    def judge_deletion(self):
        is_del_a = self.scr_list[-1] < self.param["T1"]
        is_del_b =  self.scr_list[-1] - max(self.scr_list) < self.param["THD"]
        return is_del_a or is_del_b
    
    @staticmethod
    def calc_init_score(obs):
        L0 = np.log( obs.sensor.param["BNT"] * obs.sensor.param["VC"] )
        dLk = 0
        dLs = np.log( obs.sensor.param["PD"] / obs.sensor.param["PFA"] )
        return L0 + dLk + dLs

    def plot_scr_list(self):
        plt.plot(
            self.cnt_list,
            self.scr_list,
            marker="D", color="r", alpha=.5, linestyle="None"
        )


class PDALLRTrack(LLRTrack):
    """ LLR Track for PDA

    ref) Design and Analysis of Modern Tracking Systems
                    6.6.1 The PDA Method
                    6.6.2 Extension to JPDA
                    6.6.4 PDA Track Initiation and Deletion
    """
    def __init__(self, obs, tracker, **kwargs):
        super().__init__(obs, tracker, **kwargs)
        self.pt_list = [0.5]

        # set param
        if "P22" not in self.param:
            self.param["P22"] = 0.98

    def assign(self, obs_dict):
        # track score
        score = 0.0
        for obs, ratio in obs_dict.items():
            if obs:
                score += ratio * (self.calc_match_score(obs) - self.scr_list[-1])
            else:
                score += ratio * (self._calc_miss_score() - self.scr_list[-1])

        # comfirmation and deletion parameter
        delta = 0.0
        count = 0
        for obs, ratio in obs_dict.items():
            if obs:
                count += 1
                vg = models.calc_ellipsoidal_gate_volume(self.model, obs, self.param["gate"] )
                delta -= vg * np.exp( self.model.gaussian_log_likelihood(obs) )
            else:
                pass
        if count:
            delta /= count - self.tracker.sensor.param["PD"] * self.pt_list[-1]
            delta += 1.0
            delta *= self.tracker.sensor.param["PD"]
        else:
            delta = self.tracker.sensor.param["PD"]
        ptkk = (1-delta)/(1-delta*self.pt_list[-1])*self.pt_list[-1]
        p22 = self.param["P22"]

        #  # set data
        self.obs_list.append(obs_dict)
        self.cnt_list.append(self.tracker.count)
        self.scr_list.append(score)
        self.pt_list.append( p22*ptkk + (1-p22)*(1-ptkk) )
        
        # # update model
        self.model.update(obs_dict)

class MultiSensorLLRTrack(BaseTrack):
    """ Multi Sensor LLR Track

        ref) Design and Analysis of Modern Tracking Systems
                    9.5 General Expression for Multisensor Data Association

     * Use Kalman Filter
     * Use Log Likelihood Ratio as Score
     * Observation to Track Association Based System
    """
    def calc_match_score(self, obs):
        raise NotImplementedError
