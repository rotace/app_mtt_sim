import numpy as np

import utils
import sensors


class Track():
    """ Track Base Class """

    def __init__(self, obs, tracker):
        # single sensor track
        if not obs.sensor:
            obs.sensor = tracker.sensor
        # set data
        self.tracker = tracker
        self.obs_list = [obs]
        # create model
        self.model = self.tracker.modeler.create(obs)

    def assign(self, obs):
        # set data
        self.obs_list.append(obs)
        # update model
        self.model.update(obs)

    def unassign(self):
        # set data
        self.obs_list.append(None)
        # update model
        self.model.update(None)

    def calc_match_score(self, obs):
        raise NotImplementedError



class DistTrack(Track):
    """ Distance Track

        ref) Design and Analysis of Modern Tracking Systems
                    6.4 Global Nearest Neighbor Method

     * Use Kalman Filter
     * Use Generalized Distance as Score
    """
    
    def calc_match_score(self, obs):
        dy, S = self.model.residual(obs)
        return 20 - dy @ np.linalg.inv(S) @ dy



class LLRTrack(Track):
    """ LLR Track

        ref) Design and Analysis of Modern Tracking Systems
                    6.2 Track Score Function

     * Use Kalman Filter
     * Use Log Likelihood Ratio as Score
    """
    
    def calc_match_score(self, obs):
        raise NotImplementedError



class MultiSensorLLRTrack(Track):
    """ Multi Sensor LLR Track

        ref) Design and Analysis of Modern Tracking Systems
                    9.5 General Expression for Multisensor Data Association

     * Use Kalman Filter
     * Use Log Likelihood Ratio as Score
     * Observation to Track Association Based System
    """
    
    def calc_match_score(self, obs):
        raise NotImplementedError



class Tracker():
    """ Track Base Class """

    def __init__(self, sensor, modeler):
        self.trk_list = []
        self.sensor = sensor
        self.modeler = modeler

    def update(self):
        self.sensor.update()

    def register_scan(self, obs_list):
        # self.update()
        raise NotImplementedError



class GNN(Tracker):
    """Calculate Association of Observations by GNN Method

        ref) Design and Analysis of Modern Tracking Systems
                    6.4 Global Nearest Neighbor Method
    """

    def register_scan(self, obs_list):
        self.update()
        ignore_thresh = -1000000

        #---- calc score

        # init
        M = len(self.trk_list)
        N = len(obs_list)
        S = np.empty( (M + N, N) )
        S.fill(ignore_thresh)
        
        # set new or false target score
        S[range(M, M + N), range(N)] = 0
        
        # set match score
        for i, trk in enumerate(self.trk_list):
            S[i, range(N)] = [
                trk.calc_match_score(obs)
                for obs in obs_list
            ]

        #---- calc association

        _, assign = utils.calc_best_assignment_by_auction(S)
        unassign = set(range(M)) - set(assign)

        print(S)
        print(assign)
        print(unassign)

        #---- update trackfile

        for j_obs, i_trk in enumerate(assign):
            if i_trk < M:
                # update trackfile with observation
                self.trk_list[i_trk].assign( obs_list[j_obs] )

            else:
                # create trackfile
                self.trk_list.append( DistTrack( obs_list[j_obs], self) )

        # update trackfile without observation
        for i_trk in unassign:
            self.trk_list[i_trk].unassign()

        #---- track confirmation and deletion



class JPDA(Tracker):
    """Calculate Association of Observations by JPDA Method

        ref) Design and Analysis of Modern Tracking Systems
                    6.6 The All-Neighbors Data Association Approach
    """
    pass



class MHT(Tracker):
    """Calculate Association of Observations by MHT Method

        ref) Design and Analysis of Modern Tracking Systems
                    6.7 Multiple Hypothesis Tracking
                    16. Multiple Hypothesis Tracking System Design
    """
    pass
