import math
import time
import queue
import numpy as np

import nlopt


def calc_best_assignment_by_auction( score_matrix, is_maximized=True , is_verbosed=False):
    """Calculate Best Assignment by Auction Method

        ref) Design and Analysis of Modern Tracking Systems
                    6.5.1 The Auction Algorithm

    Arguments:
        score_matrix {numpy.ndarray} -- score( I, J )
    
    Keyword Arguments:
        is_maximized {bool} -- maximize  score or not (default: {True})
    
    Returns:
        tuple of numpy.ndarray -- score( J ), assign( J )
    """

    # i_max: current tracks(M) + new tracks(N)
    # j_max: observations(N)
    i_max, j_max = score_matrix.shape
    
    # supported only i_max >= j_max
    # if i_max < j_max, transpose matrix
    assert i_max >= j_max

    prices_trk = np.zeros(i_max)
    assign_trk = -np.ones(i_max, int)
    eps = 0.2

    obs_queue = queue.Queue()
    for j in range(j_max):
        obs_queue.put(j)

    while not obs_queue.empty():
        j_obs = obs_queue.get()

        # TODO implement fast algorithm instead of sort
        if is_maximized:
            prices_trk_j = score_matrix[:, j_obs]-prices_trk
            i_trk_sort = np.argsort( prices_trk_j )[::-1]
        else:
            prices_trk_j = score_matrix[:, j_obs]+prices_trk
            i_trk_sort = np.argsort( prices_trk_j )

        i_trk1 = i_trk_sort[0]
        i_trk2 = i_trk_sort[1]

        # replace assignment if already exist
        if assign_trk[i_trk1] >= 0:
            obs_queue.put( assign_trk[i_trk1] )

        assign_trk[i_trk1] = j_obs
        prices_trk[i_trk1] += eps + abs(prices_trk_j[i_trk1]-prices_trk_j[i_trk2])

        if is_verbosed:
            print("qsize:"+str(obs_queue.qsize()))
            print("j_obs:"+str(j_obs))
            print("assign_trk:"+str(assign_trk))
            print("prices_trk:"+str(prices_trk))
            time.sleep(1)

    scores_obs = np.zeros(j_max)
    assign_obs = np.zeros(j_max, int)
    for i_trk, j_obs in enumerate(assign_trk):
        if j_obs >= 0:
            scores_obs[j_obs] = score_matrix[i_trk, j_obs]
            assign_obs[j_obs] = i_trk

    return ( scores_obs, assign_obs )


def calc_n_best_assignments_by_murty( score_matrix, ignore_thresh , n_best, is_maximized=True ):
    """Calculate N-Best Assignments by Murty Method

        ref) Design and Analysis of Modern Tracking Systems
                    6.5.2 N-Best Solutions to the Assignment Problem

    Arguments:
        score_matrix {numpy.ndarray} -- score( I, J )
        ignore_thresh {float} -- ignore threshold

    Keyword Arguments:
        is_maximized {bool} -- maximize  score or not (default: {True})

    Returns:
        list of tuple -- hypotheses  of assign
    """

    def set_sol_dict( scores, assign, consts ):
        sol_dict[tuple(assign)] = dict(
            t_score=scores.sum(),
            scores=tuple(scores),
            assign=tuple(assign),
            consts=tuple(consts)
        )

    def calc_best_assignment_with_constraints( constraints ):

        # init
        tmp_matrix[...] = score_matrix

        for index, flag in constraints:

            if flag:
                # this index is forced to use as solution
                tmp_matrix[ index[0], : ] = ignore_thresh
                tmp_matrix[ :, index[1] ] = ignore_thresh
            else:
                # this index is forced NOT to use as solution
                tmp_matrix[ index ] = ignore_thresh

        scores, assign = calc_best_assignment_by_auction( tmp_matrix, is_maximized )

        for index, flag in constraints:
            if flag:
                # replace into forced solution
                assign[index[1]] = index[0]
                scores[index[1]] = score_matrix[ index[0], index[1] ]

        return (scores, assign)


    def calc_nth_best_assignment_with_constraints( solution ):

        consts = list(solution["consts"])

        # start with the consts end observation
        if consts:
            j_start = consts[-1][0][1]
        else:
            j_start = 0

        for j in range(j_start, j_max):
            # replace from False into True at last constraint
            if  j is not  j_start:
                consts[-1] = (consts[-1][0], True)

            # append new constraint
            index = (solution["assign"][j], j)
            consts.append( (index, False) ) 

            # calc new best assignment
            scores, assign = calc_best_assignment_with_constraints( consts )

            if is_maximized:
                if not math.isclose( scores.min(), ignore_thresh ):
                    set_sol_dict( scores, assign, consts )
            else:
                if not math.isclose( scores.max(), ignore_thresh ):
                    set_sol_dict( scores, assign, consts )


    # init
    sol_dict = {} # dict remove duplicate solutions
    i_max, j_max = score_matrix.shape
    tmp_matrix = np.zeros( (i_max, j_max) )

    # first best solution
    consts = []
    scores, assign = calc_best_assignment_by_auction( score_matrix, is_maximized )
    set_sol_dict( scores, assign, consts )

    for nth in range(n_best-1):
        # nth best solution
        sol_sort = sorted( list(sol_dict.values()), key=lambda x:x["t_score"] )
        if is_maximized:
            sol_sort = sol_sort[::-1]
        if len(sol_sort) <= nth:
            n_best = len(sol_sort)
            break
        solution = sol_sort[nth]
        calc_nth_best_assignment_with_constraints( solution )

    sol_sort = sorted( list(sol_dict.values()), key=lambda x:x["t_score"] )
    if is_maximized:
        sol_sort = sol_sort[::-1]
    sol_ret = []
    for sol in sol_sort[:n_best]:
        sol_ret.append( (np.array( sol["scores"] ), np.array( sol["assign"] )) )
        # print(sol)

    return sol_ret



def calc_multidimensional_assignment(
    multi_assign,
    multi_score,
    is_NSO_enabled=False,
    is_verbosed=False,
    is_maximized=True
):
    """Calculate Multidimensional  Assignment ( Multiple Scans, Multiple Sensors )
    
    Arguments:
        multi_assign {numpy.ndarray} -- A( I, J )
        multi_score {numpy.ndarray} -- S( I )

        I : Track No.
        J : Scan No.
        A(i, j) : Obs No. assigned at Track(i) of Scan(j)

    Keyword Arguments:
        is_maximized {bool} -- maximize  score or not (default: {True})

    """

    assert multi_score.shape == (multi_assign.shape[0], )

    # cost
    if is_maximized:
        solver = MultiAssignmentSolver(multi_assign, -multi_score, is_NSO_enabled)
    else:
        solver = MultiAssignmentSolver(multi_assign, multi_score, is_NSO_enabled)

    # lagrange multipliers
    U = solver._get_u_init()

    # initialize
    Qmax=-100
    Vmin=-10
    itr=0

    # Iteration
    while (Vmin-Qmax)/abs(Qmax) > 0.01 and itr<20:
        Q, Tq , U = solver._calc_q(U)
        V, Tv = solver._calc_v(Tq)
        if itr==0:
            Qmax=Q
            Vmin=V
            Tmin=Tv
        else:
            Qmax=max([Qmax, Q])
            if Vmin>V:
                Vmin=V
                Tmin=Tv
        U = solver._update_u( U, Qmax, Vmin, Tq )

        if is_verbosed:
            print(itr,"eps, Qmax, Vmin" , np.array([(Vmin-Qmax)/abs(Qmax), Qmax, Vmin]))
            print(itr,"L, Qitr, Vitr" , np.array([(solver.G*U).sum(), Q, V]))
            print("--  G: ", solver.G)
            print("--  U: ", U)
            for i in Tmin:
                print("-->", multi_assign[i,:])
        itr+=1

    # check validation
    is_valid = True
    if (solver._calc_g(Tv)**2).sum() != 0:
        # solution do not satisfy constraints
        # there are some obs that is duplicated or not used
        is_valid = False

    return (Vmin, Tmin, is_valid)


class MultiAssignmentSolver:
    """Multidimensional  Assignment Solver

        This method is based on
                    * Morefield's method
                    * Formula of data association over multiple scans
                    * Lagrangian Relaxation

        ref) Design and Analysis of Modern Tracking Systems
                    7.2 Integer Programmin Approach (Morefield's Method)
                    7.3 Multidimensional Assignment Approach
    """
    def __init__(self, multi_assign, multi_cost, is_NSO_enabled=False):
        # assign
        self.A = multi_assign
        # cost
        self.C = multi_cost
        # obs num of each scans
        self.M = multi_assign.max(axis=0)+1
        # NSO
        self.is_NSO_enabled = is_NSO_enabled

    def _get_u_init(self):
        # return np.zeros( ( (self.M[2:]).sum(), ) )
        return np.zeros( ( (self.M[2:]-1).sum(), ) )

    def _get_u_idx(self, ir, r):
        # return (self.M[2:r]).sum() + ir
        assert 1<r and r<self.A.shape[1], "wrong input parameter r = {}".format(r)
        assert 0<ir and ir<self.M[r], "wrong input parameter ir = {}".format(ir)
        return (self.M[2:r]-1).sum() + ir - 1

    def _calc_match_tracks(self, k, track_list):
        # collect matched tracks
        match_track_list = []
        for i in range(self.A.shape[0]):
            i_vector = self.A[i, 0:k]
            for j, m in enumerate(track_list):
                j_vector = self.A[m, 0:k]
                if (i_vector == j_vector).all():
                    match_track_list.append((i,j))
                    break

        return match_track_list


    def _calc_qr(self, r, U, match_track_list):
        # D(j, ir) : min cost
        # T(j, ir) : Trk No. of min cost
        J = max([ j for i,j in match_track_list])+1
        D = np.zeros( (J, self.M[r]) )
        T = np.zeros( (J, self.M[r]), int )

        # TODO: survey about initialize
        # initialize all D by 1000 -> U become divergence because U not included in D
        # initialize all D by min(U) -> unallowed solution is selected
        
        # initialize by unallowed solution for no-overwrite
        D[:,1:] = 1000
        T[:] = -1
        
        # initialize by minimum conbination of lagrange multipliers
        # it's needed for convergence when no overwrite
        D[:,0] = sum([
            min([-U[self._get_u_idx(im, m)] for im in range(1,self.M[m]) ])
            for m in range(r+1, self.A.shape[1]) if self.M[m] > 1
        ])

        for i,j in match_track_list:
            Dj = self.C[i] - sum([
                U[self._get_u_idx(self.A[i,m], m)]
                for m in range(r+1, self.A.shape[1])
                if self.A[i,m] != 0
            ])
            ir = self.A[i,r]
            if D[j, ir] > Dj:
                D[j, ir] = Dj
                T[j, ir] = i

        # d00 constraint to be no greater than zero (p.419)
        if r==1:
            D[0,0] = min(0.0, D[0,0])

        if D.shape[0] >= D.shape[1]:
            Q, assign = calc_best_assignment_by_auction( D, False )
            track_list = [ T[i,j] for j,i in enumerate(assign) if T[i,j] >= 0 ]
        else:
            Q, assign = calc_best_assignment_by_auction( D.T, False )
            track_list = [ T[i,j] for i,j in enumerate(assign) if T[i,j] >= 0 ]
        
        Q = Q.sum() + U[self._get_u_idx(1, r+1):].sum()

        # for unit test
        self.Dq = D
        self.T = T

        return (Q, track_list)


    def _calc_q(self, U):

        def _min_objective(x, grad):
            """ Min Objective (P411, q(u)) """
            assert grad.size == 0, "solver using gradient is not supported"
            U[self._get_u_idx(1, r+1):] = x[:]
            return -1.*self._calc_qr(r, U, match_track_list)[0]

        # i : ID of tracks
        # r : ID of scans
        # ir: ID of scan r observations
        # j : ID of combination of (i0, i1, ..., ir-1)
        r=1
        track_list = list({ i0:i  for i,i0 in enumerate( self.A[:,0]) }.values())

        while r<self.A.shape[1]-1:
            match_track_list = self._calc_match_tracks(r, track_list)

            if self.is_NSO_enabled:
            # if self.A.shape[1] >= 4:
                # ND Assignment (N>=4)
                # Maximize Q by using Nonsmooth Optim Method

                # TODO: doesn't work properly ( Q(u) is sometimes over than final V(u))
                #  Is not U proper? G is not zero but Q is higher than final V
                #  calc_best_assignment_by_auction doesn't also work properly although eps-> zero
                #  Is globaly optim solver needed? (because localy optim select wrong combination)

                # Paper also saied "The exact manner in which the Lagrangian multipliers are adjusted ... 
                #  ... appears to be a 'subject of current research' " at P420

                opt = nlopt.opt(nlopt.LN_BOBYQA, len(U[self._get_u_idx(1, r+1):]))
                opt.set_min_objective(_min_objective)
                opt.set_xtol_rel(1e-4)
                U[self._get_u_idx(1, r+1):] = opt.optimize(U[self._get_u_idx(1, r+1):])
                Q, track_list = self._calc_qr(r, U, match_track_list)
            else:
                # 3D Assignment (N=3)
                Q, track_list = self._calc_qr(r, U, match_track_list)

            r += 1

        return (Q, track_list, U)

    def _calc_v(self,track_list):
        # D(j, i_last) : cost
        D = np.zeros( (len(track_list), self.M[-1]) )
        T = np.zeros( (len(track_list), self.M[-1]), int )
        # initialize by unallowed solution for no-overwrite
        D[:] = 1000
        T[:] = -1

        for i,j in self._calc_match_tracks(-1, track_list):
            i_last = self.A[i, -1]
            D[j, i_last] = self.C[i]
            T[j, i_last] = i

        if D.shape[0] >= D.shape[1]:
            V, assign = calc_best_assignment_by_auction( D, False )
            track_list = [ T[i,j] for j,i in enumerate(assign) if T[i,j] >= 0 ]
        else:
            V, assign = calc_best_assignment_by_auction( D.T, False )
            track_list = [ T[i,j] for i,j in enumerate(assign) if T[i,j] >= 0 ]
            
        V = V.sum()

        # for unit test
        self.Dv = D

        return (V, track_list)

    def _calc_g(self, track_list):
        G = np.ones( self._get_u_init().shape, int )
        for i in track_list:
            for r, ir in enumerate(self.A[i,2:]):
                r += 2
                if ir!=0:
                    G[self._get_u_idx(ir, r)] -= 1

        return G

    def _update_u( self, U, Q, V, Tq ):
        G = self._calc_g(Tq)
        if np.power(G,2).sum() != 0:
            ca = (V-Q)/np.power(G,2).sum()
            U += ca * G

        # for unit test
        self.G = G

        return U