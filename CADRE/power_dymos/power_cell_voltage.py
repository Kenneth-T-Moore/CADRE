"""
Power discipline for CADRE: Power Cell Voltage component.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range
import os

import numpy as np

from openmdao.api import ExplicitComponent

try:
    from MBI import MBI
except:
    MBI = None

try:
    from CADRE.MultiView import MultiView
    multiview_installed = True
except:
    multiview_installed = False

from smt.surrogate_models import RMTB, RMTC, KRG

#Set to True for interactive plotting of surrogate results.
USE_MV = False

multiview_installed = USE_MV & multiview_installed


class PowerCellVoltage(ExplicitComponent):
    """
    Compute the output voltage of the solar panels.
    """

    def initialize(self):
        fpath = os.path.dirname(os.path.realpath(__file__))

        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")
        self.options.declare('filename', fpath + '/../data/Power/curve.dat',
                             desc="File containing surrogate model for voltage.")
        self.options.declare('use_mbi', False,
                             desc="Set to True to build surroagate models using the MBI package. (legacy)")

    def setup(self):
        nn = self.options['num_nodes']
        filename = self.options['filename']
        use_mbi = self.options['use_mbi']

        dat = np.genfromtxt(filename)

        nT, nA, nI = dat[:3]
        nT = int(nT)
        nA = int(nA)
        nI = int(nI)
        T = dat[3:3 + nT]
        A = dat[3 + nT:3 + nT + nA]
        I = dat[3 + nT + nA:3 + nT + nA + nI]  # noqa: E741
        V = dat[3 + nT + nA + nI:].reshape((nT, nA, nI), order='F')

        if use_mbi:
            self.MBI = MBI(V, [T, A, I], [6, 6, 15], [3, 3, 3])

        else:
            temps, areas, currents = np.meshgrid(T, A, I, indexing='ij')

            xt = np.array([temps.flatten(), areas.flatten(), currents.flatten()]).T
            yt = V.flatten()
            xlimits = np.array([
                    [temps.min(), temps.max()],
                    [areas.min(), areas.max()],
                    [currents.min(), currents.max()],
                ])

            this_dir = os.path.split(__file__)[0]

            # Create the _smt_cache directory if it doesn't exist
            if not os.path.exists(os.path.join(this_dir, '_smt_cache')):
                os.makedirs(os.path.join(this_dir, '_smt_cache'))

            self.interp = interp = RMTB(
                    xlimits=xlimits,
                    num_ctrl_pts=10,
                    order=5,
                    approx_order=5,
                    nonlinear_maxiter=2,
                    solver_tolerance=1.e-20,
                    energy_weight=1.e-13,
                    regularization_weight=2.3,
                    # smoothness=np.array([1., 1., 1.]),
                    extrapolate=False,
                    print_global=True,
                    data_dir=os.path.join(this_dir, '_smt_cache'),
                )

            interp.set_training_values(xt, yt)
            interp.train()

            if multiview_installed:
                info = {'nx':3,
                        'ny':1,
                        'user_func':interp.predict_values,
                        'resolution':100,
                        'plot_size':8,
                        'dimension_names':[
                            'Temperature',
                            'Area',
                            'Current'],
                        'bounds':xlimits.tolist(),
                        'X_dimension':0,
                        'Y_dimension':2,
                        'scatter_points':[xt, yt],
                        'dist_range': 0.0,
                        }

                # Initialize display parameters and draw GUI
                MultiView(info)

        self.x = np.zeros((84 * nn, 3), order='F')
        self.xV = self.x.reshape((nn, 7, 12, 3), order='F')

        # Inputs
        self.add_input('LOS', np.zeros((nn, )), units=None,
                       desc='Line of Sight over Time')

        self.add_input('temperature', np.zeros((nn, 5)), units='degK',
                       desc='Temperature of solar cells over time')

        self.add_input('exposed_area', np.zeros((nn, 7, 12)), units='m**2',
                       desc='Exposed area to sun for each solar cell over time')

        self.add_input('Isetpt', np.zeros((nn, 12)), units='A',
                       desc='Currents of the solar panels')

        # Outputs
        self.add_output('V_sol', np.zeros((nn, 12)), units='V',
                        desc='Output voltage of solar panel over time')

        rows = np.arange(nn*12)
        cols = np.tile(np.repeat(0, 12), nn) + np.repeat(np.arange(nn), 12)

        self.declare_partials('V_sol', 'LOS', rows=rows, cols=cols)

        row = np.tile(np.repeat(0, 5), 12) + np.repeat(np.arange(12), 5)
        rows = np.tile(row, nn) + np.repeat(12*np.arange(nn), 60)
        col = np.tile(np.arange(5), 12)
        cols = np.tile(col, nn) + np.repeat(5*np.arange(nn), 60)

        self.declare_partials('V_sol', 'temperature', rows=rows, cols=cols)

        row = np.tile(np.arange(12), 7)
        rows = np.tile(row, nn) + np.repeat(12*np.arange(nn), 84)
        cols = np.arange(nn*7*12)

        self.declare_partials('V_sol', 'exposed_area', rows=rows, cols=cols)

        row_col = np.arange(nn*12)

        self.declare_partials('V_sol', 'Isetpt', rows=row_col, cols=row_col)

    def setx(self, inputs):
        temperature = inputs['temperature']
        LOS = inputs['LOS']
        exposed_area = inputs['exposed_area']
        Isetpt = inputs['Isetpt']

        for p in range(12):
            i = 4 if p < 4 else (p % 4)
            for c in range(7):
                self.xV[:, c, p, 0] = temperature[:, i]
                self.xV[:, c, p, 1] = LOS * exposed_area[:, c, p]
                self.xV[:, c, p, 2] = Isetpt[:, p]

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        nn = self.options['num_nodes']
        use_mbi = self.options['use_mbi']

        self.setx(inputs)
        if use_mbi:
            raw = self.MBI.evaluate(self.x)[:, 0].reshape((nn, 7, 12), order='F')
        else:
            raw = self.interp.predict_values(self.x).reshape((nn, 7, 12), order='F')

        outputs['V_sol'] = np.zeros((nn, 12))
        for c in range(7):
            outputs['V_sol'] += raw[:, c, :]

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        nn = self.options['num_nodes']
        use_mbi = self.options['use_mbi']

        exposed_area = inputs['exposed_area']
        LOS = inputs['LOS']

        if use_mbi:
            raw1 = self.MBI.evaluate(self.x, 1)[:, 0].reshape((nn, 7, 12), order='F')
            raw2 = self.MBI.evaluate(self.x, 2)[:, 0].reshape((nn, 7, 12), order='F')
            raw3 = self.MBI.evaluate(self.x, 3)[:, 0].reshape((nn, 7, 12), order='F')

        else:
            raw1 = self.interp.predict_derivatives(self.x, 0).reshape(nn, 7, 12, order='F')
            raw2 = self.interp.predict_derivatives(self.x, 1).reshape(nn, 7, 12, order='F')
            raw3 = self.interp.predict_derivatives(self.x, 2).reshape(nn, 7, 12, order='F')

        dV_dL = np.empty((nn, 12))
        dV_dT = np.zeros((nn, 12, 5))
        dV_dA = np.zeros((nn, 7, 12))
        dV_dI = np.empty((nn, 12))

        for p in range(12):
            i = 4 if p < 4 else (p % 4)
            for c in range(7):
                dV_dL[:, p] += raw2[:, c, p] * exposed_area[:, c, p]
                dV_dT[:, p, i] += raw1[:, c, p]
                dV_dA[:, c, p] += raw2[:, c, p] * LOS
                dV_dI[:, p] += raw3[:, c, p]

        partials['V_sol', 'LOS'] = dV_dL.flatten()
        partials['V_sol', 'temperature'] = dV_dT.flatten()
        partials['V_sol', 'exposed_area'] = dV_dA.flatten()
        partials['V_sol', 'Isetpt'] = dV_dI.flatten()