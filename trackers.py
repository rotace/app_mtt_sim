import copy
import numpy as np
from scipy import integrate, interpolate, stats
import matplotlib.pyplot as plt

import utils
import models

IGNORE_THRESH = -1000000


class BaseTracker():
    """ Tracker Base Class """

    def __init__(self, sensor, model_factory, track_factory):
        self.sensor = sensor
        self.track_factory = track_factory
        self.timestamp = 0

        self.track_factory.set_attr(
            tracker=self,
            model_factory=model_factory
        )

    def update(self):
        self.timestamp += 1
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
                self.trk_list.append( self.track_factory.create( obs_list[j_obs] ) )

        # update trackfile without observation
        for i_trk in unassign:
            self.trk_list[i_trk].unassign()

        #---- track confirmation and deletion
        
        # delete trackfile
        self.trk_list = [ trk for trk in self.trk_list if not trk.judge_deletion() ]

        # confirmation and representation
        return [ trk for trk in self.trk_list if trk.judge_confirmation() ]



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
            self.trk_list.append( self.track_factory.create( obs ) )

        #---- track confirmation and deletion
        # TODO: implement track confirmation and deletion




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
                    child_hyp.trk_list.append( self.track_factory.create( obs_list[j_obs] ) )

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




class TrackerEvaluator():
    """ Evaluate Each Tracker

    * calc MOE(Measures of Effectiveness) etc.

    ref) Design and Analysis of Modern Tracking Systems
        13.6 MTT System Evaluation Metrics
    """

    def __init__(self, tracker, tgt_list, R, PD=None, PFA=None):
        self.tracker = tracker
        self.tgt_list = tgt_list
        self.sensor = tracker.sensor

        self.R = R

        if PD is None:
            self.PD = self.sensor.param["PD"]
        else:
            self.PD = PD

        if PFA is None:
            self.PFA= self.sensor.param["PFA"]
        else:
            self.PFA = PFA

    def _initialize_simulation(self):
        return (copy.deepcopy(self.tracker), copy.deepcopy(self.tgt_list))

    @staticmethod
    def _calc_score_matrix(tgt_list, trk_list):
        """ calc score matrix between target and track

        ref) Design and Analysis of Modern Tracking Systems
        13.6.1 Track-to-Truth Assignment
        """
        # init
        M = len(tgt_list)
        N = len(trk_list)
        S = np.empty( (M + N, N) )
        S.fill(IGNORE_THRESH)
        
        # set extra track score
        S[range(M, M + N), range(N)] = [0.0]*N
        
        # set match score
        for i, tgt in enumerate(tgt_list):
            S[i, range(N)] = [
                tgt.calc_match_score(trk.model)
                if tgt.is_in_gate(trk.model) else IGNORE_THRESH
                for trk in trk_list
            ]
        return S

    def _update_sim_param(self, i_scan):
        # user can change sim parameter R, PD, PFA
        pass
    
    def _update(self, tracker, tgt_list, i_scan=None):
        assert "y_mins" in self.sensor.param
        assert "y_maxs" in self.sensor.param
        assert "y_stps" in self.sensor.param

        # sensor characteristics
        self._update_sim_param(i_scan)
        y_mins = self.sensor.param["y_mins"]
        y_maxs = self.sensor.param["y_maxs"]
        y_stps = self.sensor.param["y_stps"]
        n_mesh = int(np.prod([ abs((y_max-y_min)/y_stp) for y_min, y_max, y_stp in zip(y_mins, y_maxs, y_stps) ]))

        # count targets
        n_tgt = len(tgt_list)
        
        # init observation
        obs_list = []

        # add target observation
        obs_list.extend([
            models.Obs(
                np.random.multivariate_normal(tgt.x[:len(self.R)], self.R),
                self.R,
                tracker.sensor
            )
            for tgt in tgt_list if tgt.is_exist() and np.random.choice([True, False], p=[self.PD, 1-self.PD])
        ])
        
        # add false alarm observation
        obs_list.extend([
            models.Obs(
                np.array([ np.random.uniform(y_min, y_max) for y_min, y_max in zip(y_mins, y_maxs) ]),
                self.R,
                tracker.sensor
            )
            for k in range(stats.binom.rvs(n=n_mesh, p=self.PFA))
        ])

        # register scan data
        trk_list = tracker.register_scan(obs_list)

        # tgt_list update
        [ tgt.update(self.sensor.param["dT"]) for tgt in tgt_list ]

        # calc MOF (Measure of Fit) and assignment
        S = self._calc_score_matrix(tgt_list, trk_list)
        _, assign = utils.calc_best_assignment_by_auction(S)

        # create track to truth assignment table
        trk_truth = [0]*(n_tgt+1)
        for j_trk, i_tgt in enumerate(assign):
            if i_tgt < n_tgt:
                trk_truth[i_tgt] = trk_list[j_trk].get_id()
            else:
                trk_truth[n_tgt] += 1

        return (tracker, tgt_list, trk_list, obs_list, trk_truth)
    
    def plot_position(self, n_scan=10, is_all_obs_displayed=False):
        # init
        tracker, tgt_list = self._initialize_simulation()
        trk_scan_list = []
        tgt_scan_list = []
        obs_scan_list = []

        # simulate
        for i_scan in range(n_scan):

            tracker, tgt_list, trk_list, obs_list, _ = self._update(tracker, tgt_list)
            tgt_scan_list.append( copy.deepcopy(tgt_list) )
            trk_scan_list.append( copy.deepcopy(trk_list) )
            obs_scan_list.append( copy.deepcopy(obs_list) )


        plt.plot(
            [tgt.x[0] if tgt is not None else None for tgt_list in tgt_scan_list for tgt in tgt_list ],
            [tgt.x[1] if tgt is not None else None for tgt_list in tgt_scan_list for tgt in tgt_list ],
            marker="D", color="b", alpha=.5, linestyle="None", label="tgt"
        )
        plt.plot(
            [trk.model.x[0] if trk is not None else None for trk_list in trk_scan_list for trk in trk_list ],
            [trk.model.x[1] if trk is not None else None for trk_list in trk_scan_list for trk in trk_list ],
            marker="D", color="r", alpha=.5, linestyle="None", label="trk"
        )
        if is_all_obs_displayed:
            plt.plot(
                [obs.y[0] for obs_list in obs_scan_list for obs in obs_list ],
                [obs.y[1] for obs_list in obs_scan_list for obs in obs_list ],
                marker="D", color="g", alpha=.5, linestyle="None", label="obs"
            )
        else:
            plt.plot(
                [trk.obs_list[-1].y[0] if trk is not None and trk.obs_list[-1] is not None else None for trk_list in trk_scan_list for trk in trk_list ],
                [trk.obs_list[-1].y[1] if trk is not None and trk.obs_list[-1] is not None else None for trk_list in trk_scan_list for trk in trk_list ],
                marker="D", color="g", alpha=.5, linestyle="None", label="obs"
            )
        plt.legend()
        plt.axis("equal")
        plt.grid()
        plt.show()

    def estimate_track_statistics(self, n_scan=10, n_run=10):
        """ estimate track statictics

        ref) Design and Analysis of Modern Tracking Systems
        13.3.3 SPRT Analysis of Track Confirmation
        13.6.2 Computation of Track Statictics
        """
        trk_truth_tbl_runs = []
        for i_run in range(n_run):

            # init
            tracker, tgt_list = self._initialize_simulation()
            trk_truth_tbl = []

            # simulate
            for i_scan in range(n_scan):

                tracker, tgt_list, _, _, trk_truth = self._update(tracker, tgt_list, i_scan)
                trk_truth_tbl.append( trk_truth )

            trk_truth_tbl_runs.append( trk_truth_tbl )

        # calc statistics
        # * Cumulative probability of track confirmation (Nc :comfirmation)
        # *            probability of current confirmed track (Nm :maintenance)
        # * Kinematic error means and standard deviation (Na)
        n_tgt = len(tgt_list)
        Na = np.zeros((n_tgt, n_scan))
        Nc = np.zeros((n_tgt, n_scan))
        Nm = np.zeros((n_tgt, n_scan))
        for trk_truth_tbl in trk_truth_tbl_runs:
            gg = np.zeros((n_tgt, n_scan))
            xc = np.zeros((n_tgt, n_scan))
            xm = np.zeros((n_tgt, n_scan))
            for k_scan, trk_truth in enumerate(trk_truth_tbl):
                for i_tgt in range(n_tgt):
                    #  g(i,k): a confirmed track is associated with target i on scan k
                    gg[i_tgt, k_scan] = trk_truth[i_tgt] > 0
                    # xc(i,k): a confirmed track is or has previously been associated with target i
                    xc[i_tgt, k_scan] = np.any(gg[i_tgt, :k_scan])
                    # xm(i,k): a confirmed track is currently associated with target i and xc(i,k)=1
                    xm[i_tgt, k_scan] = gg[i_tgt, k_scan] and xc[i_tgt, k_scan]

                    Na[i_tgt, k_scan] += gg[i_tgt, k_scan]
                    Nc[i_tgt, k_scan] += xc[i_tgt, k_scan]
                    Nm[i_tgt, k_scan] += xm[i_tgt, k_scan]

        # * The expected time to track confirmation (Tc)
        # * The time at which 90% of tracks were confirmed (T90)
        Tc = np.zeros((n_tgt,))
        T90 = np.zeros((n_tgt,))
        for i_tgt in range(n_tgt):
            # Tc =   E(X): expected value of random value X
            # FX = CDF(x): cumulative density function
            #       QF(x): quantile function
            x = np.array(range(n_scan)) * self.sensor.param["dT"]
            FX = Nc[i_tgt, :]/n_run
            # Tc = E(X) = integral( 1-CDF(x) )
            Tc[i_tgt] = integrate.simps(1.0-FX, x)
            # T90 = QF(x=90) = CDF^-1(y=90)
            T90[i_tgt] = interpolate.interp1d(FX, x, fill_value="extrapolate")(0.9)

        result = dict()
        result["Na"] = Na/n_run
        result["Nc"] = Nc/n_run
        result["Nm"] = Nm/n_run
        result["Tc"] = Tc
        result["T90"] = T90

        return result