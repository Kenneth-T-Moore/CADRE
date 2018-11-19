"""
Unit test vs saved data from John's CMF implementation.
"""
from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_check_partials

from CADRE.power_dymos.power_cell_voltage import PowerCellVoltage
from CADRE.power_dymos.power_group import PowerGroup
from CADRE.power_dymos.power_solar_power import PowerSolarPower
from CADRE.power_dymos.power_total import PowerTotal
from CADRE.test.util import load_validation_data

#
# load saved data from John's CMF implementation.
#
n, m, h, setd = load_validation_data(idx='5')


class TestPower(unittest.TestCase):

    def test_PowerCellVoltage(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', PowerCellVoltage(num_nodes=1500),
                                        promotes=['*'])

        prob.setup()

        prob['LOS'] = setd['LOS']
        prob['temperature'] = setd['temperature'].T
        prob['exposed_area'] = np.transpose(setd['exposedArea'], [2, 0, 1])
        prob['Isetpt'] = setd['Isetpt'].T

        prob.run_model()

        for var in ['V_sol']:

            tval = setd[var].T

            assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))

    def test_PowerSolarPower(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', PowerSolarPower(num_nodes=1500),
                                        promotes=['*'])

        prob.setup()

        prob['V_sol'] = setd['V_sol'].T
        prob['Isetpt'] = setd['Isetpt'].T

        prob.run_model()

        for var in ['P_sol']:

            tval = setd[var]

            assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))

    def test_PowerTotal(self):
        prob = Problem()
        comp = prob.model.add_subsystem('comp', PowerTotal(num_nodes=1500),
                                        promotes=['*'])

        prob.setup()

        prob['P_sol'] = setd['P_sol']
        prob['P_comm'] = setd['P_comm']
        prob['P_RW'] = setd['P_RW'].T

        prob.run_model()

        for var in ['P_bat']:

            tval = setd[var]

            assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))

    def test_partials(self):

        prob = Problem(model = PowerGroup(num_nodes=5))

        prob.setup()

        prob['LOS'] = setd['LOS'][:5]
        prob['temperature'] = setd['temperature'][:, :5].T
        prob['cell_voltage_comp.Isetpt'] = setd['Isetpt'][:, :5].T
        prob['solar_power_comp.Isetpt'] = setd['Isetpt'][:, :5].T
        prob['exposed_area'] = np.transpose(setd['exposedArea'][:, :, :5], [2, 0, 1])

        prob.run_model()

        np.set_printoptions(linewidth=100000, edgeitems=10000)
        J = prob.check_partials(method='cs', compact_print=True, excludes=['*voltage*'])
        assert_check_partials(J, atol=1e-6, rtol=1e-6)

        # large deriv, so ignore atol
        J = prob.check_partials(method='cs', compact_print=True, includes=['*voltage*'])
        assert_check_partials(J, atol=1e2, rtol=1e-2)

if __name__ == "__main__":
    unittest.main()