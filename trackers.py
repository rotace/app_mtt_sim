import copy
import numpy as np

import utils

IGNORE_THRESH = -1000000


class BaseTracker():
    """ Tracker Base Class """

    def __init__(self, sensor, model_factory, track_factory):
        self.sensor = sensor
        self.model_factory = model_factory
        self.track_factory = track_factory
        self.count = 0

    def update(self):
        self.count += 1
        self.sensor.update()

    def _calc_score_matrix(self, hyp, obs_list):
        # init
        M = len(hyp.trk_list)
        N = len(obs_list)
        S = np.empty( (M + N, N) )
        S.fill(IGNORE_THRESH)
        
        # set new or false target score
        S[range(M, M + N), range(N)] = [
            self.track_factory.calc_init_score(obs)
            for obs in obs_list
        ]
        
        # set match score
        for i, trk in enumerate(hyp.trk_list):
            S[i, range(N)] = [
                trk.calc_match_score(obs)
                if trk.is_in_gate(obs) else IGNORE_THRESH
                for obs in obs_list
            ]
        
        return S

    def register_scan(self, obs_list):
        raise NotImplementedError




class GNN(BaseTracker):
    """Calculate Association of Observations by GNN Method

        ref) Design and Analysis of Modern Tracking Systems
                    6.4 Global Nearest Neighbor Method
    """
    def __init__(self, sensor, model_factory, track_factory):
        super().__init__(
            sensor=sensor,
            model_factory=model_factory,
            track_factory=track_factory
        )
        self.trk_list = []

    def register_scan(self, obs_list):
        self.update()

        # single sensor track
        for obs in obs_list:
            if not obs.sensor:
                obs.sensor = self.sensor

        #---- calc score

        # init
        M = len(self.trk_list)
        N = len(obs_list)
        S = self._calc_score_matrix(self, obs_list)

        #---- calc association

        _, assign = utils.calc_best_assignment_by_auction(S)
        unassign = set(range(M)) - set(assign)

        # print(S)
        # print(assign)
        # print(unassign)

        #---- update trackfile

        for j_obs, i_trk in enumerate(assign):
            if i_trk < M:
                # update trackfile with observation
                self.trk_list[i_trk].assign( obs_list[j_obs] )

            else:
                # create trackfile
                self.trk_list.append( self.track_factory.create( obs_list[j_obs], self) )

        # update trackfile without observation
        for i_trk in unassign:
            self.trk_list[i_trk].unassign()

        #---- track confirmation and deletion



class JPDA(BaseTracker):
    """Calculate Association of Observations by JPDA Method

        ref) Design and Analysis of Modern Tracking Systems
                    6.6 The All-Neighbors Data Association Approach
    """
    def __init__(self, sensor, model_factory, track_factory):
        super().__init__(
            sensor=sensor,
            model_factory=model_factory,
            track_factory=track_factory
        )
        self.trk_list = []

    def register_scan(self, obs_list):
        self.update()

        # single sensor track
        for obs in obs_list:
            if not obs.sensor:
                obs.sensor = self.sensor

        #---- calc score

        # init
        M = len(self.trk_list)
        N = len(obs_list)
        S = self._calc_score_matrix(self, obs_list)

        #---- calc association
        assign_idx_hyp_list = utils.calc_n_best_assignments_by_murty(S, IGNORE_THRESH, 10)

        hyp_assign_dict_list = list()
        all_assign_dict = {}
        for trk in self.trk_list:
            all_assign_dict[trk] = set()

        for _, assign in assign_idx_hyp_list:
            
            hyp_assign_dict = {}
            for trk in self.trk_list:
                hyp_assign_dict[trk] = None

            for j_obs, i_trk in enumerate(assign):
                if i_trk < M:
                    hyp_assign_dict[self.trk_list[i_trk]] = obs_list[j_obs]
                    all_assign_dict[self.trk_list[i_trk]].add(obs_list[j_obs])
            
            hyp_assign_dict_list.append(hyp_assign_dict)

        
        hyp_log_score_list = []
        for hyp_assign_dict in hyp_assign_dict_list:

            if N-M>0:
                hyp_log_score = np.log( self.sensor.param["BETA"] ) *(N-M)
            elif M-N>0:
                hyp_log_score = np.log(1-self.sensor.param["PD"]) *(M-N)
            else:
                hyp_log_score = 0

            for trk, obs in hyp_assign_dict.items():
                if obs:
                    hyp_log_score += np.log(self.sensor.param["PD"])
                    hyp_log_score += trk.model.gaussian_log_likelihood(obs)
                else:
                    hyp_log_score += np.log(1-self.sensor.param["PD"])
                    hyp_log_score += np.log(self.sensor.param["BETA"])

            hyp_log_score_list.append(hyp_log_score)
        
        hyp_score_list = np.exp(np.array(hyp_log_score_list))
        self.hyp_score_list = hyp_score_list

        # update trackfile with related observation
        for trk in self.trk_list:
            assign_score_dict = {}
            for normed_hyp_score, hyp_assign_dict in zip(hyp_score_list/hyp_score_list.sum(), hyp_assign_dict_list):
                obs = hyp_assign_dict[trk]
                if obs in assign_score_dict:
                    assign_score_dict[obs] += normed_hyp_score
                else:
                    assign_score_dict[obs] = normed_hyp_score
            trk.assign( assign_score_dict )

        # create trackfile of all observation
        for obs in obs_list:
            self.trk_list.append( self.track_factory.create( obs, self ) )

        #---- track confirmation and deletion




class MHT(BaseTracker):
    """Calculate Association of Observations by conventional MHT Method

        ref) Design and Analysis of Modern Tracking Systems
                    6.7 Multiple Hypothesis Tracking
    """
    class Hypothesis():
        def __init__(self, trk_list):
            self.trk_list = trk_list

        def create(self):
            return copy.deepcopy(self)

    def __init__(self, sensor, model_factory, track_factory):
        super().__init__(
            sensor=sensor,
            model_factory=model_factory,
            track_factory=track_factory
        )
        self.hyp_list = []

    def register_scan(self, obs_list):
        self.update()

        # single sensor track
        for obs in obs_list:
            if not obs.sensor:
                obs.sensor = self.sensor

        if not self.hyp_list:
            self.hyp_list.append(self.Hypothesis(trk_list=[]))

        # TODO implement fast algorithm (p.366)
        new_hyp_list = []
        for hyp in self.hyp_list:
            S = self._calc_score_matrix(hyp, obs_list)
            new_hyp_list += [
                (scores, assign, hyp)
                for scores, assign
                in utils.calc_n_best_assignments_by_murty(S, IGNORE_THRESH, 10)
            ]

        new_hyp_sort = sorted( new_hyp_list, key=lambda x:x[0].sum() )[::-1]

        # limit hypothesis
        if len(new_hyp_sort)>10:
            new_hyp_sort = new_hyp_list[:10]

        child_hyp_list = []
        for _, assign, parent_hyp in new_hyp_sort:

            M = len(parent_hyp.trk_list)            
            unassign = set(range(M)) - set(assign)

            child_hyp = parent_hyp.create()
            for j_obs, i_trk in enumerate(assign):
                if i_trk < M:
                    # update trackfile with observation
                    child_hyp.trk_list[i_trk].assign( obs_list[j_obs] )

                else:
                    # create trackfile
                    child_hyp.trk_list.append( self.track_factory.create( obs_list[j_obs], self) )

            # update trackfile without observation
            for i_trk in unassign:
                child_hyp.trk_list[i_trk].unassign()

            #---- track confirmation and deletion
            child_hyp_list.append(child_hyp)

        self.hyp_list = child_hyp_list




class TOMHT(BaseTracker):
    """Calculate Association of Observations by track-oriented MHT Method

        ref) Design and Analysis of Modern Tracking Systems
                    7. Advanced Methods for MTT Data Association
                    16. Multiple Hypothesis Tracking System Design
    """
    # TODO need to implement MTT multiscan data association in utils module
    pass
