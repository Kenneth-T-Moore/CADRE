"""
Solar discipline for CADRE
"""
from __future__ import print_function, division, absolute_import
from six.moves import range
import os

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent

from CADRE.kinematics import fixangles

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


class SolarExposedAreaComp(ExplicitComponent):
    """
    Exposed area calculation for a given solar cell

    p: panel ID [0,11]
    c: cell ID [0,6]
    a: fin angle [0,90]
    z: azimuth [0,360]
    e: elevation [0,180]
    LOS: line of sight with the sun [0,1]
    """
    def initialize(self):
        fpath = os.path.dirname(os.path.realpath(__file__))

        self.options.declare('num_nodes', types=(int, ),
                             desc="Number of time points.")
        self.options.declare('raw1_file', fpath + '/data/Solar/Area10.txt',
                             desc="angle, azimuth, elevation points for exposed area interpolation.")
        self.options.declare('raw2_file', fpath + '/data/Solar/Area_all.txt',
                             desc="exposed area at points in raw1_file for exposed area interpolation.")
        self.options.declare('use_mbi', False,
                             desc="Set to True to build surroagate models using the MBI package. (legacy)")

    def setup(self):
        nn = self.options['num_nodes']
        use_mbi = self.options['use_mbi']
        raw1_file = self.options['raw1_file']
        raw2_file = self.options['raw2_file']

        raw1 = np.genfromtxt(raw1_file)
        raw2 = np.loadtxt(raw2_file)

        nc = self.nc = 7
        self.np = 12
        ncp = self.nc * self.np

        self.na = 10
        self.nz = 73
        self.ne = 37
        angle = np.zeros(self.na)
        azimuth = np.zeros(self.nz)
        elevation = np.zeros(self.ne)

        index = 0
        for i in range(self.na):
            angle[i] = raw1[index]
            index += 1
        for i in range(self.nz):
            azimuth[i] = raw1[index]
            index += 1

        index -= 1
        azimuth[self.nz - 1] = 2.0 * np.pi
        for i in range(self.ne):
            elevation[i] = raw1[index]
            index += 1

        angle[0] = 0.0
        angle[-1] = np.pi / 2.0
        azimuth[0] = 0.0
        azimuth[-1] = 2 * np.pi
        elevation[0] = 0.0
        elevation[-1] = np.pi

        counter = 0
        data = np.zeros((self.na, self.nz, self.ne, self.np * self.nc))
        flat_size = self.na * self.nz * self.ne
        for p in range(self.np):
            for c in range(nc):
                data[:, :, :, counter] = \
                    raw2[nc * p + c][119:119 + flat_size].reshape((self.na,
                                                                   self.nz,
                                                                   self.ne))
                counter += 1

        if use_mbi:
            self.MBI = MBI(data, [angle, azimuth, elevation],
                                 [4, 10, 8],
                                 [4, 4, 4])

        else:
            angles, azimuths, elevations = np.meshgrid(angle, azimuth, elevation, indexing='ij')

            xt = np.array([angles.flatten(), azimuths.flatten(), elevations.flatten()]).T
            yt = np.zeros((flat_size, ncp))
            counter = 0
            for p in range(self.np):
                for c in range(nc):
                    yt[:, counter] = data[:, :, :, counter].flatten()
                    counter += 1

            xlimits = np.array([
                [angle[0], angle[-1]],
                [azimuth[0], azimuth[-1]],
                [elevation[0], elevation[-1]],
                ])

            this_dir = os.path.split(__file__)[0]

            # Create the _smt_cache directory if it doesn't exist
            if not os.path.exists(os.path.join(this_dir, '_smt_cache')):
                os.makedirs(os.path.join(this_dir, '_smt_cache'))

            self.interp = interp = RMTB(
                xlimits=xlimits,
                num_ctrl_pts=16,
                order=4,
                approx_order=4,
                nonlinear_maxiter=2,
                solver_tolerance=1.e-20,
                energy_weight=1e-8,
                regularization_weight=1.e-11,
                # smoothness=np.array([1., 1., 1.]),
                extrapolate=False,
                print_global=True,
                data_dir=os.path.join(this_dir, '_smt_cache'),
            )

            interp.set_training_values(xt, yt)
            interp.train()

            if multiview_installed:
                info = {'nx':3,
                    'ny':ncp,
                    'user_func':interp.predict_values,
                    'resolution':100,
                    'plot_size':8,
                    'dimension_names':[
                        'Angle',
                        'Azimuth',
                        'Elevation'],
                    'bounds':xlimits.tolist(),
                    'X_dimension':0,
                    'Y_dimension':1,
                    'scatter_points':[xt, yt],
                    'dist_range': 0.0,
                    }

                # Initialize display parameters and draw GUI
                MultiView(info)

        self.x = np.zeros((nn, 3))

        # Inputs
        self.add_input('fin_angle', 0.0, units='rad',
                       desc='Fin angle of solar panel')

        self.add_input('azimuth', np.zeros((nn, )), units='rad',
                       desc='Azimuth angle of the sun in the body-fixed frame over time')

        self.add_input('elevation', np.zeros((nn, )), units='rad',
                       desc='Elevation angle of the sun in the body-fixed frame over time')

        # Outputs
        self.add_output('exposed_area', np.zeros((nn, self.nc, self.np)),
                        desc='Exposed area to sun for each solar cell over time',
                        units='m**2', lower=-5e-3, upper=1.834e-1)

        self.declare_partials('exposed_area', 'fin_angle')

        rows = np.tile(np.arange(ncp), nn) + np.repeat(ncp*np.arange(nn), ncp)
        cols = np.tile(np.repeat(0, ncp), nn) + np.repeat(np.arange(nn), ncp)

        self.declare_partials('exposed_area', 'azimuth', rows=rows, cols=cols)
        self.declare_partials('exposed_area', 'elevation', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        use_mbi = self.options['use_mbi']
        nn = self.options['num_nodes']

        self.setx(inputs)
        if use_mbi:
            P = self.MBI.evaluate(self.x)
        else:
            P = self.interp.predict_values(self.x)

        outputs['exposed_area'] = P.reshape(nn, self.nc, self.np, order='F')

    def setx(self, inputs):
        """
        Sets our state array
        """
        nn = self.options['num_nodes']

        result = fixangles(nn, inputs['azimuth'], inputs['elevation'])
        self.x[:, 0] = inputs['fin_angle']
        self.x[:, 1] = result[0]
        self.x[:, 2] = result[1]

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        use_mbi = self.options['use_mbi']
        nn = self.options['num_nodes']

        if use_mbi:
            Jfin = self.MBI.evaluate(self.x, 1).reshape(nn, self.nc, self.np, order='F')
            Jaz = self.MBI.evaluate(self.x, 2).reshape(nn, self.nc, self.np, order='F')
            Jel = self.MBI.evaluate(self.x, 3).reshape(nn, self.nc, self.np, order='F')

        else:
            Jfin = self.interp.predict_derivatives(self.x, 0).reshape(nn, self.nc, self.np, order='F')
            Jaz = self.interp.predict_derivatives(self.x, 1).reshape(nn, self.nc, self.np, order='F')
            Jel = self.interp.predict_derivatives(self.x, 2).reshape(nn, self.nc, self.np, order='F')

        partials['exposed_area', 'fin_angle'] = Jfin.flatten()
        partials['exposed_area', 'azimuth'] = Jaz.flatten()
        partials['exposed_area', 'elevation'] = Jel.flatten()
