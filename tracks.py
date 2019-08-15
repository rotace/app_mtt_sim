import numpy as np

import models


class BaseTrackFactory():
    """ Track Factory Base Class """
    def __init__(self, track, gate=None):
        self.track = track
        self.gate  = gate

    def create(self, obs, tracker):
        return self.track(obs, tracker, gate=self.gate)

    def calc_init_score(self, obs):
        return self.track.calc_init_score(obs)



class BaseTrack():
    """ Track Base Class """

    def __init__(self, obs, tracker, gate):
        # set data
        self.gate = gate
        self.tracker = tracker
        self.obs_list = [obs]
        self.cnt_list = [self.tracker.count]
        # create model
        self.model = self.tracker.model_factory.create(obs)

    def assign(self, obs):
        # set data
        self.obs_list.append(obs)
        self.cnt_list.append(self.tracker.count)
        # update model
        self.model.update(obs)

    def unassign(self):
        # set data
        self.obs_list.append(None)
        self.cnt_list.append(self.tracker.count)
        # update model
        self.model.update(None)

    def is_in_gate(self, obs):
        gate, norm_dist, _ = models.calc_ellipsoidal_gate(self.model, obs)
        if self.gate:
            gate = self.gate
        return gate > norm_dist

    def calc_match_score(self, obs):
        raise NotImplementedError

    @staticmethod
    def calc_init_score(obs):
        raise NotImplementedError



class DistTrack(BaseTrack):
    """ Distance Scored Track

        ref) Design and Analysis of Modern Tracking Systems
                    6.4 Global Nearest Neighbor Method

     * Use Kalman Filter
     * Use Generalized Distance as Score
    """
    def calc_match_score(self, obs):
        gate, norm_dist, detS = models.calc_ellipsoidal_gate(self.model, obs)
        return gate - norm_dist - np.log(detS)

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
    def __init__(self, obs, tracker, gate):
        super().__init__(obs, tracker, gate)
        self.scr_list = [self.calc_init_score(obs)]

    def assign(self, obs):
        super().assign(obs)
        self.scr_list.append(self.calc_match_score(obs))

    def unassign(self):
        super().unassign()
        self.scr_list.append(self._calc_miss_score())

    def calc_match_score(self, obs):
        dLk = np.log( obs.sensor.param["VC"] ) + self.model.gaussian_log_likelihood(obs)
        dLs = np.log( obs.sensor.param["PD"] / obs.sensor.param["PFA"] )
        return self.scr_list[-1] + dLk + dLs

    def _calc_miss_score(self):
        dLk = 0
        dLs = np.log( 1.0 - self.tracker.sensor.param["PD"] )
        return self.scr_list[-1] + dLk + dLs

    @staticmethod
    def calc_init_score(obs):
        L0 = np.log( obs.sensor.param["BNT"] * obs.sensor.param["VC"] )
        dLk = 0
        dLs = np.log( obs.sensor.param["PD"] / obs.sensor.param["PFA"] )
        return L0 + dLk + dLs



class PDALLRTrack(LLRTrack):
    """ LLR Track for PDA """
    def assign(self, obs_dict):
        score = 0
        for obs, ratio in obs_dict.items():
            if obs:
                score += ratio * (self.calc_match_score(obs) - self.scr_list[-1])
            else:
                score += ratio * (self._calc_miss_score() - self.scr_list[-1])
        self.scr_list.append( score )



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
