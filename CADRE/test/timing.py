"""
Time CADRE's execution and derivative calculations.
"""
from __future__ import print_function

import time

import numpy as np

from openmdao.api import Problem

from CADRE.CADRE_group import CADRE
from CADRE.test.util import load_validation_data


n, m, h, setd = load_validation_data(idx='0')

# instantiate Problem with CADRE model
t0 = time.time()
model = CADRE(n=n, m=m)
prob = Problem(model)
print("Instantiation:  ", time.time() - t0, 's')

# do problem setup
t0 = time.time()
prob.setup(check=False)
print("Problem Setup:  ", time.time() - t0, 's')

# set initial values from validation data
input_data = [
    'CP_P_comm',
    'LD',
    'cellInstd',
    'CP_gamma',
    'finAngle',
    'lon',
    'CP_Isetpt',
    'antAngle',
    't',
    'r_e2b_I0',
    'lat',
    'alt',
    'iSOC'
]
for inp in input_data:
    actual = setd[inp]

    # Pickle was recorded with different array order.
    nn = len(actual.shape)
    if nn > 1 and (actual.shape[-1] == 1500 or actual.shape[-1] == 300):
        zz = [nn-1]
        zz.extend(np.arange(nn-1))
        actual = np.transpose(actual, axes=zz)

    prob[inp] = actual

# run model
t0 = time.time()
prob.run_model()
print("Execute Model:  ", time.time() - t0, 's')

inputs = ['CP_gamma']
outputs = ['Data']

# calculate total derivatives
t0 = time.time()
J1 = prob.compute_totals(of=outputs, wrt=inputs)
print("Compute Totals: ", time.time() - t0, 's')

# Probably no reason to always do the fd totals. Should take about 300 times a single execution
# anyway.
exit()

# calculate total derivatives via finite difference
model.approx_totals(method='fd')

t0 = time.time()
J1 = prob.compute_totals(of=outputs, wrt=inputs)
print("Approx Totals:  ", time.time() - t0, 's')
