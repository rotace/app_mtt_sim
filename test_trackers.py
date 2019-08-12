import unittest
import numpy as np

import models
import sensors
import trackers

class TestTracker(unittest.TestCase):

    def test_GNN(self):

        tracker = trackers.GNN(
            sensor=sensors.SimpleRadar2D(
                time_interval = 1.0
            ),
            modeler=models.SimplePolar2D.create_factory(0.1)
        )

        targets = [
            np.array([0,0]),
            np.array([10,20]),
            np.array([40,50]),
        ]

        tracker.register_scan(
            [models.Obs(y, np.eye(2) * 0.001) for y in targets]
        )

        targets = [
            np.array([1,1]),
            np.array([11,21]),
            np.array([41,51]),
        ]

        tracker.register_scan(
            [models.Obs(y, np.eye(2) * 0.001) for y in targets]
        )

        self.assertEqual(1, 2, "test")
        