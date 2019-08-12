import unittest
import utils
import numpy as np

class TestUtils(unittest.TestCase):
    
    def test_calc_best_assignment_by_auction(self):

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


    def test_calc_n_best_assignments_by_murty(self):
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


if __name__ == "__main__":
    unittest.main()