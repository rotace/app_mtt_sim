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

    def _calc_q(self, multi_assign, C, M, U):
        # D(i1,i2) : cost
        # T(i1,i2) : Trk No. of min cost
        D = np.zeros( (M[0]+1, M[1]+1) )
        T = np.zeros( (M[0]+1, M[1]+1), int )
        D[:,1:] = 1000 # unallowed solution
        D[0, 0] = min(-U)
        T[:] = -1
        for i in range(multi_assign.shape[0]):
            i1 = multi_assign[i, 0]
            i2 = multi_assign[i, 1]
            i3 = multi_assign[i, 2]
            Di = C[i] - U[i3]
            if D[i1, i2] > Di:
                D[i1, i2] = Di
                T[i1, i2] = i

        if D.shape[0] >= D.shape[1]:
            Q, assign = calc_best_assignment_by_auction( D, False )
            track_list = [ T[i,j] for j,i in enumerate(assign) if T[i,j] >= 0 ]
        else:
            Q, assign = calc_best_assignment_by_auction( D.T, False )
            track_list = [ T[i,j] for i,j in enumerate(assign) if T[i,j] >= 0 ]
        
        Q = Q.sum() + U.sum()

        # for unit test
        self.D =  D

        return (Q, track_list)

    def _calc_v(self, multi_assign, track_list, C, M):
        # D(i1-i2, i3) : cost
        D = np.zeros( (len(track_list), M[2]+1) )
        T = np.zeros( (len(track_list), M[2]+1), int )
        D[:,1:] = 1000 # unallowed solution
        for i in range(multi_assign.shape[0]):
            i1 = multi_assign[i, 0]
            i2 = multi_assign[i, 1]
            i3 = multi_assign[i, 2]
            for j, k in enumerate(track_list):
                dual_i1 = multi_assign[k, 0]
                dual_i2 = multi_assign[k, 1]
                if i1 == dual_i1 and i2 == dual_i2:
                    D[j, i3] = C[i]
                    T[j, i3] = i

        if D.shape[0] >= D.shape[1]:
            V, assign = calc_best_assignment_by_auction( D, False )
            track_list = [ T[i,j] for j,i in enumerate(assign) if T[i,j] >= 0 ]
        else:
            V, assign = calc_best_assignment_by_auction( D.T, False )
            track_list = [ T[i,j] for i,j in enumerate(assign) if T[i,j] >= 0 ]
            
        V = V.sum()

        return (V, track_list)

    def _calc_g(self, multi_assign, track_list, M):
        G = np.ones( ( M[2]+1, ) )
        for k in track_list:
            i3 = multi_assign[k, 2]
            G[i3] -= 1

        return G

    def _update_u( self, U, Q, V, G ):
        ca = (V-Q)/np.power(G[1:],2).sum()
        U[1:] += ca * G[1:]

        return U

    def calc_3dimensional_assignment( self, multi_assign, multi_score,  is_maximized=True ):
        """Calculate 3D  Assignment ( 3 Scans, 3 Sensors )

        ref) Design and Analysis of Modern Tracking Systems
                    7.3.2 3D Application of Lagrangian Relaxation

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
        assert multi_assign.shape[1] == 3

        # obs num of each scans
        M = multi_assign.max(axis=0)

        # cost
        if is_maximized:
            C = -multi_score
        else:
            C = multi_score

        # lagrange multipliers
        U = np.zeros( ( M[2]+1, ) )
        Qmax=-100
        Vmin=-10
        itr=0

        # Iteration
        while (Vmin-Qmax)/abs(Qmax) > 0.01 or itr>100:
            Q, T = self._calc_q(multi_assign, C, M, U)
            G = self._calc_g( multi_assign, T, M)
            V, T = self._calc_v( multi_assign, T, C, M)
            if itr==0:
                Qmax=Q
                Vmin=V
                Tmin=T
            else:
                Qmax=max([Qmax, Q])
                if Vmin>V:
                    Vmin=V
                    Tmin=T
            U = self._update_u( U, Qmax, Vmin, G )
            itr+=1

        return (Vmin, Tmin)

    def calc_multidimensional_assignment( self, multi_assign, multi_score,  is_maximized=True ):
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

        # obs num of each scans
        M = multi_assign.max(axis=0)

        # cost
        if is_maximized:
            C = -multi_score
        else:
            C = multi_score

        U = np.zeros( ( M[2]+1, ) )

        opt = nlopt.opt(nlopt.LN_BOBYQA, len(U))
        opt.set_min_objective(lambda x, grad: -1.*self._calc_q(multi_assign, C, M, x)[0]  )
        opt.set_xtol_rel(1e-4)
        U = opt.optimize(U)
        Q = opt.last_optimum_value()

        return (None, None)