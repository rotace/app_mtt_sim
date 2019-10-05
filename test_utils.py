import unittest
import utils
import numpy as np

class TestUtils(unittest.TestCase):
    
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

        # sample (minimize)
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

        solver = utils.MultiAssignmentSolver()

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
        
        M = multi_assign.max(axis=0)
        C = -multi_score
        U = np.zeros( ( M[2]+1, ) )

        # First Iteration
        Q, T = solver._calc_q(multi_assign, C, M, U)
        G = solver._calc_g( multi_assign, T, M)
        V, T = solver._calc_v( multi_assign, T, C, M)
        Qmax=Q
        Vmin=V
        U = solver._update_u( U, Qmax, Vmin, G )

        # Table 7.5
        expected = np.array(
            [
                [0, -10.2, -5.5],
                [-6.8, -18, -17],
                [-13.2, -14.1, -16.7]
            ]
        )
        np.testing.assert_almost_equal(solver.D, expected)
        np.testing.assert_almost_equal(Q, -40.4)
        np.testing.assert_almost_equal(V, -31.7)
        np.testing.assert_almost_equal(G, [1, -2, 1])
        np.testing.assert_almost_equal(U, [0, -3.48, 1.74])

        # Second Iteration
        Q, T = solver._calc_q(multi_assign, C, M, U)
        G = solver._calc_g( multi_assign, T, M)
        V, T = solver._calc_v( multi_assign, T, C, M)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, G )

        # Table 7.6
        expected = np.array(
            [
                [-1.74, -6.72, -7.24],
                [-6.94, -16.54, -13.52],
                [-12.34, -15.84, -18.44]
            ]
        )
        np.testing.assert_almost_equal(solver.D, expected)
        np.testing.assert_almost_equal(Q, -38.46)
        np.testing.assert_almost_equal(V, -34.7)
        np.testing.assert_almost_equal(T, [7, 17])
        np.testing.assert_almost_equal(G, [1, 1, -1])
        np.testing.assert_almost_equal(U, [0, -1.60, -0.14])

        # Third Iteration
        Q, T = solver._calc_q(multi_assign, C, M, U)
        G = solver._calc_g( multi_assign, T, M)
        V, T = solver._calc_v( multi_assign, T, C, M)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, G )

        np.testing.assert_almost_equal(Qmax, -37.34)

        #  4th Iteration
        Q, T = solver._calc_q(multi_assign, C, M, U)
        G = solver._calc_g( multi_assign, T, M)
        V, T = solver._calc_v( multi_assign, T, C, M)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, G )

        np.testing.assert_almost_equal(Qmax, -35.144)

        #  5th Iteration
        Q, T = solver._calc_q(multi_assign, C, M, U)
        G = solver._calc_g( multi_assign, T, M)
        V, T = solver._calc_v( multi_assign, T, C, M)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, G )

        np.testing.assert_almost_equal(Qmax, -35.144)

        #  6th Iteration
        Q, T = solver._calc_q(multi_assign, C, M, U)
        G = solver._calc_g( multi_assign, T, M)
        V, T = solver._calc_v( multi_assign, T, C, M)
        Qmax=max([Qmax, Q])
        Vmin=min([Vmin, V])
        U = solver._update_u( U, Qmax, Vmin, G )

        np.testing.assert_almost_equal(Qmax, -34.922)


        V, T = solver.calc_3dimensional_assignment( multi_assign, multi_score )

        np.testing.assert_almost_equal(T, [7, 17])
        np.testing.assert_almost_equal(V, -34.7)
        np.testing.assert_almost_equal(multi_score[T].sum(), 34.7)
    

        V, T = solver.calc_multidimensional_assignment( multi_assign, multi_score )

        # np.testing.assert_almost_equal(T, [7, 17])
        # np.testing.assert_almost_equal(V, -34.7)
        # np.testing.assert_almost_equal(multi_score[T].sum(), 34.7)


if __name__ == "__main__":
    unittest.main()