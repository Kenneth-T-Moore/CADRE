"""
Unit test vs saved data from John's CMF implementation.
"""
from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_check_partials

from CADRE.solar_dymos import SolarExposedAreaComp
from CADRE.test.util import load_validation_data

#
# load saved data from John's CMF implementation.
#
n, m, h, setd = load_validation_data(idx='5')

class TestSolar(unittest.TestCase):

    def test_SunLOSComp(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', SolarExposedAreaComp(num_nodes=1500, use_mbi=True),
                                        promotes=['*'])

        prob.setup()

        prob['fin_angle'] = setd['finAngle']
        prob['azimuth'] = setd['azimuth'].T
        prob['elevation'] = setd['elevation'].T

        prob.run_model()

        for var in ['exposed_area']:

            tval = np.transpose(setd['exposedArea'], [2, 0, 1])

            #print('error', np.sqrt(np.sum((tval-prob[var])**2)))
            #import matplotlib.pyplot as plt; t = np.arange(1500); plt.plot(t, tval[:, 0, 0]); plt.plot(t, prob[var][:, 0, 0])

            assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))

    def test_SunLOSComp_derivs(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', SolarExposedAreaComp(num_nodes=5, use_mbi=True),
                                        promotes=['*'])

        prob.setup()

        prob['fin_angle'] = setd['finAngle']
        prob['azimuth'] = 3.0 * np.random.random((5, ))
        prob['elevation'] = 3.0 * np.random.random((5, ))

        prob.run_model()

        np.set_printoptions(linewidth=100000, edgeitems=10000)
        J = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(J, atol=3.0E-5, rtol=1.0E-2)

if __name__ == "__main__":
    unittest.main()