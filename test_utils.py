import unittest
import utils
import numpy as np
from scipy.linalg import block_diag

np.set_printoptions(suppress=True)

class TestUtils(unittest.TestCase):

    def test_cart2polar_and_polar2cart(self):
        # loop test
        expected = np.array([1,1,0,1,-1,0,-1,-1,0])
        np.testing.assert_almost_equal(utils.polar2cart(utils.cart2polar(expected)), expected)
        np.testing.assert_almost_equal(utils.cart2polar(utils.polar2cart(expected)), expected)

        expected = np.array([1,np.pi/4,np.pi/4,4,5,6,7,8,9])
        np.testing.assert_almost_equal(utils.polar2cart(utils.cart2polar(expected)), expected)
        np.testing.assert_almost_equal(utils.cart2polar(utils.polar2cart(expected)), expected)

        expected = np.array([1,np.pi*3/4,-np.pi/4,4,5,6,7,8,9])
        np.testing.assert_almost_equal(utils.polar2cart(utils.cart2polar(expected)), expected)
        np.testing.assert_almost_equal(utils.cart2polar(utils.polar2cart(expected)), expected)

        # approach target
        expected_car = np.array([1,1,0,-1,-1,0,0,0,0])
        expected_pol = np.array([np.sqrt(2),np.pi/4,0,-np.sqrt(2),0,0,0,0,0])
        np.testing.assert_almost_equal(utils.cart2polar(expected_car), expected_pol)
        np.testing.assert_almost_equal(utils.polar2cart(expected_pol), expected_car)

        # horizontal turn target
        expected_car = np.array([1,1,0,-1,1,0,-1,-1,0])
        expected_pol = np.array([np.sqrt(2),np.pi/4,0,0,1,0,0,0,0])
        np.testing.assert_almost_equal(utils.cart2polar(expected_car), expected_pol)
        np.testing.assert_almost_equal(utils.polar2cart(expected_pol), expected_car)

        # vertical turn target
        expected_car = np.array([1,0,-1,-1,0,-1,-1,0,+1])
        expected_pol = np.array([np.sqrt(2),0,np.pi/4,0,0,1,0,0,0])
        np.testing.assert_almost_equal(utils.cart2polar(expected_car), expected_pol)
        np.testing.assert_almost_equal(utils.polar2cart(expected_pol), expected_car)


    def  test_swap_block_matrix(self):
        vec = np.array([0,1,2,3,4,5])
        np.testing.assert_almost_equal(utils.swap_block_matrix(vec, 3), [0,2,4,1,3,5])
        np.testing.assert_almost_equal(utils.swap_block_matrix(vec, 2), [0,3,1,4,2,5])
        
        mat = np.array(
            [
                [11.,12.,0.0,0.0,0.0,0.0],
                [21.,22.,0.0,0.0,0.0,0.0],
                [0.0,0.0,31.,32.,0.0,0.0],
                [0.0,0.0,41.,42.,0.0,0.0],
                [0.0,0.0,0.0,0.0,51.,52.],
                [0.0,0.0,0.0,0.0,61.,62.],
            ]
        )
        expected = np.array(
            [
                [11.,0.0,0.0,12.,0.0,0.0],
                [0.0,31.,0.0,0.0,32.,0.0],
                [0.0,0.0,51.,0.0,0.0,52.],
                [21.,0.0,0.0,22.,0.0,0.0],
                [0.0,41.,0.0,0.0,42.,0.0],
                [0.0,0.0,61.,0.0,0.0,62.],
            ]
        )
        np.testing.assert_almost_equal(utils.swap_block_matrix(mat, 3), expected)

        mat = np.array(
            [
                [11.,12.,13.,0.0,0.0,0.0],
                [21.,22.,23.,0.0,0.0,0.0],
                [0.0,0.0,0.0,31.,32.,33.],
                [0.0,0.0,0.0,41.,42.,43.],
            ]
        )
        expected = np.array(
            [
                [11.,0.0,12.,0.0,13.,0.0],
                [0.0,31.,0.0,32.,0.0,33.],
                [21.,0.0,22.,0.0,23.,0.0],
                [0.0,41.,0.0,42.,0.0,43.],
            ]
        )
        np.testing.assert_almost_equal(utils.swap_block_matrix(mat, 2), expected)
    

    def test_calc_best_assignment_by_auction(self):
        """Calculate Best Assignment by Auction Method

            ref) Design and Analysis of Modern Tracking Systems
                        6.5.1 The Auction Algorithm
        """

        arg = np.array(
            [[1, 10],
            [3, 15]]
        )

        # maximize score
        expected = (
            np.array([1, 15]),
            np.array([0, 1])
        )
        actual = utils.calc_best_assignment_by_auction( arg )
        np.testing.assert_almost_equal(actual, expected)

        # minimize cost
        expected = (
            np.array([3, 10]),
            np.array([1, 0])
        )
        actual = utils.calc_best_assignment_by_auction( arg, False )
        np.testing.assert_almost_equal(actual, expected)

        # sample 1 (minimize)
        arg = np.array(
            [
                [-6.8, -18, -14.8],
                [-11.1, -9, -16.7]
            ]
        ).T
        expected = (
            np.array([-18, -16.7]),
            np.array([1, 2])
        )
        actual = utils.calc_best_assignment_by_auction( arg, False )
        np.testing.assert_almost_equal(actual, expected)



    def test_calc_n_best_assignments_by_murty(self):
        """Calculate N-Best Assignments by Murty Method

            ref) Design and Analysis of Modern Tracking Systems
                        6.5.2 N-Best Solutions to the Assignment Problem
        """

        # infinite cost
        X = 10000000

        # minimize cost
        arg = np.array(
            [[10, 5, 8, 9],
            [7, X, 20, X],
            [X, 21, X, X],
            [X, 15, 17, X],
            [X, X, 16, 22]]
        )
        expected = (
            [
            (np.array([7, 15, 16, 9]), np.array([1, 3, 4, 0])),
            (np.array([7, 5, 17, 22]), np.array([1, 0, 3, 4])),
            (np.array([7, 15, 8, 22]), np.array([1, 3, 0, 4])),
            (np.array([7, 21, 16, 9]), np.array([1, 2, 4, 0])),
            ]
        )
        actual = utils.calc_n_best_assignments_by_murty( arg, X, 4, False )
        np.testing.assert_almost_equal(actual, expected)



    def test_calc_3dimensional_assignment(self ):
        """Calculate 3D  Assignment ( 3 Scans, 3 Sensors )

            ref) Design and Analysis of Modern Tracking Systems
                    7.3.2 3D Application of Lagrangian Relaxation
        """

        # Table 7.4
        multi_assign = np.array(
            [
                [0,1,1],
                [0,2,1],
                [0,2,2],
                [1,0,1],
                [1,0,2],
                [1,1,0],
                [1,2,0],
                [1,1,1],
                [1,1,2],
                [1,2,1],
                [1,2,2],
                [2,0,1],
                [2,0,2],
                [2,1,0],
                [2,2,0],
                [2,1,2],
                [2,2,1],
                [2,2,2],
            ]
        )

        # Table 7.4
        multi_score = np.array(
            [
                10.2,
                4.7,
                5.5,
                6.8,
                5.2,
                6.8,
                10.9,
                18,
                14.8,
                17,
                9.9,
                13.2,
                10.6,
                4.5,
                11.1,
                14.1,
                9,
                16.7,
            ]
        )

        solver = utils.MultiAssignmentSolver(multi_assign, -multi_score)
        np.testing.assert_almost_equal(solver._get_u_idx(1,2), 0)
        np.testing.assert_almost_equal(solver._get_u_idx(2,2), 1)
        
        # lagrange multipliers
        U = solver._get_u_init()

        # First Iteration
        Q, Tq , U = solver._calc_q(U)
        V, Tv = solver._calc_v(Tq)
        Qmax=Q
        Vmin=V
        U = solver._update_u( U, Qmax, Vmin, Tq )

        # Table 7.5
        expected = np.array(
            [
                [0, -10.2, -5.5],
                [-6.8, -18, -17],
                [-13.2, -14.1, -16.7]
            ]
        )
        np.testing.assert_almost_equal(solver.Dq, expected)
        np.testing.assert_almost_equal(Q, -40.4)
        np.testing.assert_almost_equal(Tq, [11, 0, 9])
        np.testing.assert_almost_equal(V, -31.7)
        np.testing.assert_almost_equal(solver.G[-2:], [-2, 1])
        np.testing.assert_almost_equal(U[-2:], [-3.48, 1.74])

        # Second Iteration
        Q, Tq , U = solver._calc_q(U)
        V, Tv = solver._calc_v(Tq)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, Tq )

        # Table 7.6
        expected = np.array(
            [
                [-1.74, -6.72, -7.24],
                [-6.94, -16.54, -13.52],
                [-12.34, -15.84, -18.44]
            ]
        )
        np.testing.assert_almost_equal(solver.Dq, expected)
        np.testing.assert_almost_equal(Q, -38.46)
        np.testing.assert_almost_equal(Tq, [8, 17])
        np.testing.assert_almost_equal(V, -34.7)
        np.testing.assert_almost_equal(Tv, [7, 17])
        np.testing.assert_almost_equal(solver.G[-2:], [1, -1])
        np.testing.assert_almost_equal(U[-2:], [-1.60, -0.14])

        # Third Iteration
        Q, Tq , U = solver._calc_q(U)
        V, Tv = solver._calc_v(Tq)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, Tq )

        np.testing.assert_almost_equal(Qmax, -37.34)

        #  4th Iteration
        Q, Tq , U = solver._calc_q(U)
        V, Tv = solver._calc_v(Tq)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, Tq )

        np.testing.assert_almost_equal(Qmax, -35.144)

        #  5th Iteration
        Q, Tq , U = solver._calc_q(U)
        V, Tv = solver._calc_v(Tq)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, Tq )

        np.testing.assert_almost_equal(Qmax, -35.144)

        #  6th Iteration
        Q, Tq , U = solver._calc_q(U)
        V, Tv = solver._calc_v(Tq)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, Tq )

        np.testing.assert_almost_equal(Qmax, -34.922)

        # unabled NSO
        V, T, is_valid = utils.calc_multidimensional_assignment( multi_assign, multi_score )
        np.testing.assert_almost_equal(T, [7, 17])
        np.testing.assert_almost_equal(V, -34.7)
        np.testing.assert_almost_equal(multi_score[T].sum(), 34.7)
        np.testing.assert_almost_equal(is_valid, True)

        # enabled NSO
        V, T, is_valid = utils.calc_multidimensional_assignment( multi_assign, multi_score, True )
        np.testing.assert_almost_equal(T, [7, 17])
        np.testing.assert_almost_equal(V, -34.7)
        np.testing.assert_almost_equal(multi_score[T].sum(), 34.7)
        np.testing.assert_almost_equal(is_valid, True)


        # unstable case
        multi_assign = np.array(
            [
                [0,1,1,0],
                [0,2,1,0],
                [0,2,2,0],
                [1,0,1,0],
                [1,0,2,0],
                [1,1,0,0],
                [1,2,0,0],
                [1,1,1,1],
                [1,1,2,0],
                [1,2,1,0],
                [1,2,2,0],
                [2,0,1,0],
                [2,0,2,0],
                [2,1,0,0],
                [2,2,0,0],
                [2,1,2,0],
                [2,2,1,0],
                [2,2,2,2],
            ]
        )

        V, T, is_valid = utils.calc_multidimensional_assignment( multi_assign, multi_score, True, False )
        np.testing.assert_almost_equal(is_valid, False) # unstable case do not satisfy constraints

        # stable case
        multi_assign = np.array(
            [
                [0,1,1,0],
                [1,1,1,1],
                [1,1,2,0],
                [2,2,2,2],
            ]
        )
        multi_score = np.array(
            [
                10.2,
                17,
                9.9,
                16.7,
            ]
        )
        V, T, is_valid = utils.calc_multidimensional_assignment( multi_assign, multi_score, True, False )
        np.testing.assert_almost_equal(is_valid, True)


if __name__ == "__main__":
    unittest.main()