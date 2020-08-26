import unittest
import numpy as np

import sensors

np.set_printoptions(suppress=True)


class TestSensors(unittest.TestCase):

    def test_counter(self):
        class InheritBaseSensor(sensors.BaseSensor):
            pass
        sensors.BaseSensor.initialize()
        np.testing.assert_equal(InheritBaseSensor._generate_id(), 1)
        np.testing.assert_equal(sensors.BaseSensor._generate_id(), 2)
        InheritBaseSensor.initialize()
        np.testing.assert_equal(sensors.BaseSensor._generate_id(), 1)
        np.testing.assert_equal(InheritBaseSensor._generate_id(), 2)