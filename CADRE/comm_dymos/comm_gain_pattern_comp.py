from __future__ import print_function, division, absolute_import

import os

import numpy as np

from openmdao.api import ExplicitComponent

from CADRE.kinematics import fixangles

from MBI import MBI

try:
    from CADRE.MultiView import MultiView
    multiview_installed = True
except:
    multiview_installed = False

from smt.surrogate_models import RMTB, RMTC, KRG

# Set to True to use Surrogate Model Toolbox instead of MBI
USE_SMT = True

#Set to True for interactive plotting of surrogate results.
USE_MV = False

multiview_installed = USE_MV & multiview_installed


class CommGainPatternComp(ExplicitComponent):
    """
    Determines transmitter gain based on an external az-el map.
    """

    def initialize(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

        rawG_file = os.path.join(parent_path, 'data/Comm/Gain.txt')

        self.options.declare('num_nodes', types=(int,))
        self.options.declare('rawG_file', types=(str,), default=rawG_file)

    def setup(self):

        nn = self.options['num_nodes']

        rawGdata = np.genfromtxt(self.options['rawG_file'])
        rawG = (10 ** (rawGdata / 10.0)).reshape((361, 361), order='F')

        pi = np.pi
        az = np.linspace(0, 2 * pi, 361)
        el = np.linspace(0, 2 * pi, 361)

        if USE_SMT:

            azs, els = np.meshgrid(az, el, indexing='ij')

            xt = np.array([azs.flatten(), els.flatten()]).T
            yt = rawG.flatten()
            xlimits = np.array([
                    [azs.min(), azs.max()],
                    [els.min(), els.max()],
                ])

            this_dir = os.path.split(__file__)[0]

            # Create the _smt_cache directory if it doesn't exist
            if not os.path.exists(os.path.join(this_dir, '_smt_cache')):
                os.makedirs(os.path.join(this_dir, '_smt_cache'))

            self.interp = interp = RMTB(
                    xlimits=xlimits,
                    num_ctrl_pts=22,
                    order=6,
                    approx_order=6,
                    nonlinear_maxiter=2,
                    solver_tolerance=1.e-20,
                    energy_weight=1.e-5,
                    regularization_weight=1e-8,
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

        else:
            self.MBI = MBI(rawG, [az, el], [15, 15], [4, 4])
            self.MBI.seterr('raise')

        self.x = np.zeros((nn, 2), order='F')

        # Inputs
        self.add_input('azimuthGS', np.zeros(nn), units='rad',
                       desc='Azimuth angle from satellite to ground station in '
                            'Earth-fixed frame over time')

        self.add_input('elevationGS', np.zeros(nn), units='rad',
                       desc='Elevation angle from satellite to ground station '
                            'in Earth-fixed frame over time')

        # Outputs
        self.add_output('gain', np.zeros(nn), units=None,
                        desc='Transmitter gain over time')

        row_col = np.arange(nn)

        self.declare_partials('gain', 'elevationGS', rows=row_col, cols=row_col)
        self.declare_partials('gain', 'azimuthGS', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        result = fixangles(self.options['num_nodes'], inputs['azimuthGS'], inputs['elevationGS'])
        self.x[:, 0] = result[0]
        self.x[:, 1] = result[1]
        if USE_SMT:
            outputs['gain'][:] = self.interp.predict_values(self.x).flatten()
        else:
            outputs['gain'] = self.MBI.evaluate(self.x)[:, 0]

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        if USE_SMT:
            partials['gain', 'azimuthGS'] = self.interp.predict_derivatives(self.x, 0).flatten()
            partials['gain', 'elevationGS'] = self.interp.predict_derivatives(self.x, 1).flatten()
        else:
            partials['gain', 'azimuthGS'] = self.MBI.evaluate(self.x, 1)[:, 0]
            partials['gain', 'elevationGS'] = self.MBI.evaluate(self.x, 2)[:, 0]
