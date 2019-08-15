import math
import time
import queue
import numpy as np


def calc_best_assignment_by_auction( score_matrix, is_maximized=True ):
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

        # print("qsize:"+str(obs_queue.qsize()))
        # print("j_obs:"+str(j_obs))
        # print("assign_trk:"+str(assign_trk))
        # print("prices_trk:"+str(prices_trk))
        # time.sleep(1)

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
