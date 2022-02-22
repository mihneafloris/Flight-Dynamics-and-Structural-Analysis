import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import numpy as np
import warnings


class MomentZDeflectionY:
    def __init__(self, data_dict):
        self.plotting_nodes = 100

        self.x_1 = data_dict['x_1']
        self.x_2 = data_dict['x_2']
        self.x_3 = data_dict['x_3']
        self.x_a = data_dict['x_a']
        self.l_a = data_dict['l_a']
        self.c_a = data_dict['C_a']
        self.d_1 = data_dict['d_1']
        self.d_3 = data_dict['d_3']
        self.theta = data_dict['theta']

        self.I_yy = data_dict['I_yy']
        self.I_zz = data_dict['I_zz']
        self.E = data_dict['E']

        self.q = data_dict['q(x,z)']  # This should be a function
        self.R_1_y = data_dict['R_1_y']
        self.R_2_y = data_dict['R_2_y']
        self.R_3_y = data_dict['R_3_y']
        self.P_I = data_dict['P_I']
        self.P = data_dict['P']

        self.dx = data_dict['dx']
        self.dy = data_dict['dy']
        self.dz = data_dict['dz']

        self.interpolation_type = data_dict['interpolation_type']
        self.integration_type = data_dict['integration_type']

        # The underscore means that they are private variables which should not be accessed from outside.
        # After certain functions get called, these will all become functions.
        self._M_z = None
        self._S_y = None
        self._v_y = None

    def get_moment_around_z(self):
        if self._M_z is None:
            # First integrate the self.q three times:
            #  1. From 0 to C along dz
            #  2. From 0 to x (variable) along dx
            #  3. From 0 to x (variable) along dx
            # This yields q_contribution_moment (which is still a function of x)

            # ---- FIRST INTEGRATION: FROM 0 TO Z ALONG Z-AXIS ----
            # We need z to be the first input to q function, so time to rename:
            def q_func(z, x):
                return self.q(x, z)

            def integrate_from_0_to_c(q, x):
                return self.integration_type(q, 0, self.c_a, args=(x))

            x_array = np.arange(0, self.l_a, self.dx)

            # I will add the letters of the direction that q was integrated in to the name:
            qz = [integrate_from_0_to_c(self.q, x)[0] for x in x_array]
            qz = self.interpolation_type(x_array, qz)

            # ---- SECOND INTEGRATION: FROM 0 TO X ALONG X-AXIS ----
            def integrate_from_0_to_x(q, x):
                return self.integration_type(q, 0, x)

            qzx = [integrate_from_0_to_x(qz, x)[0] for x in x_array]
            qzx = self.interpolation_type(x_array, qzx)

            # ---- THIRD INTEGRATION: FROM 0 TO X ALONG X-AXIS ----
            qzxx = [integrate_from_0_to_x(qzx, x)[0] for x in x_array]
            qzxx = self.interpolation_type(x_array, qzxx)

            # Then we build self._M_z as a function using the integrated q and the decomposed forces.

            def M_z(x):
                assert type(x) != np.ndarray  # You cannot pass numpy arrays to this function.
                moment = qzxx(x)
                if x > self.x_1:
                    moment += - self.R_1_y * (x - self.x_1)
                if x > (self.x_2 - self.x_a/2):
                    moment += self.P_I * np.sin(self.theta) * (x - self.x_2 - self.x_a/2)
                if x > self.x_2:
                    moment += - self.R_2_y * (x - self.x_2)
                if x > (self.x_2 + self.x_a/2):
                    moment += self.P*np.sin(self.theta)*(x - self.x_2 - self.x_a/2)
                if x > self.x_3:
                    moment += - self.R_3_y * (x - self.x_3)
                return moment

            # Let's now interpolate this so we can make a function that supports inputting numpy array, and can easily
            # take the derivative of it.
            fine_x_array = np.arange(0, self.l_a, self.dx/10)  # added precision because this doesn't take much time anyway
            m_z = [M_z(x) for x in x_array]
            m_z = self.interpolation_type(x_array, m_z)

            self._M_z = m_z
        return self._M_z

    def get_deflection_in_z(self):
        if self._v_y is None:
            if self._M_z is None:
                warnings.warn("The get_moment_around_z function has not yet been called, which is odd...")
                self.get_moment_around_z()

            # Now that we know for sure that we have the moment function at self._M_z, we can get going integrating.
            # We know that the deflection is given by:
            # v_y(x) = - \iint_0^x M_z(x) dx / (E*I_zz) + C_1 x + C_2.
            # The first, and main, part of this equation will be referred to as MZ
            # We will first do the integration, and then find the two constants.

            def integrate_from_0_to_x(func, x):
                return self.integration_type(func, 0, x)

            x_array = np.arange(0, self.l_a, self.dx)

            # ---- FIRST INTEGRATION ----
            m_z_x = np.array([integrate_from_0_to_x(self._M_z, x) for x in x_array])
            m_z_x = self.interpolation_type(x_array, m_z_x)

            # ---- SECOND INTEGRATION ----
            MZ = np.array([integrate_from_0_to_x(m_z_x, x) for x in x_array])
            MZ = self.interpolation_type(x_array, MZ)

            # The constants are found by setting the deflection at x_1 and x_3, so we will do that:
            # C_1 x_1 + C_2 = d_1 - MZ(x_1)
            # C_1 x_3 + C_2 = d_3 - MZ(x_3)
            # the above will be solved using matrix inversion for C_1 and C_3

            # @todo Find a way to also solve for the reaction forces here as well.

            # The last step is then to build up a scipy function which returns the deflections:
            # self._v_z(x) = INT(x) + C_1 *x + C_2.
        return self._v_y

    def get_shear_force_in_y(self):
        if self._S_y is None:
            # The shear force in the y direction is be the derivative of the moment in the x-direction. Therefore,
            # Twe just need to take the derivative of this function, analytically.
            self._S_y = self._M_z.derivative()
        return self._S_y

    def plot_moment_around_z(self, show=False):
        if self._M_z is None:
            raise RuntimeError("Moment not yet calculated, call 'get_moment_around_z' ")
        nodes = np.arange(0, self.l_a, self.dx)
        moment = np.array([self._M_z(x) for x in nodes])
        plt.plot(nodes, moment)
        plt.xlabel("x distance [m]")
        plt.ylabel("Moment [Nm]")
        plt.title("Moment around z axis")
        if show:
            plt.show()
        else:
            plt.savefig('_02_numerical_model/images/moment_around_z.pdf', type="pdf")
            # one of the two below does the trick.
            plt.cla()
            plt.clf()
        return True

    def plot_shear_force_in_y(self):
        if self._S_y is None:
            raise RuntimeError("Shear force not yet calculated, call 'get_moment_shear_force_in_y' ")
        nodes = np.array(self.plotting_nodes)
        plt.plot(nodes, self._S_y(nodes))
        plt.xlabel("x distance [m]")
        plt.ylabel("Shear force [N]")
        plt.title("Shear force in the y direction")
        plt.show()

    def plot_deflection_in_y(self):
        if self._v_y is None:
            raise RuntimeError("Deflection not yet calculated, call 'get_moment_deflection_in_y' ")
        nodes = np.array(self.plotting_nodes)
        plt.plot(nodes, self._v_y(nodes))
        plt.xlabel("x distance [m]")
        plt.ylabel("Deflection [m]")
        plt.title("Deflection in the y direction")
        plt.show()

