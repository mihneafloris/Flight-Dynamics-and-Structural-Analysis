import numpy as np
import matplotlib.pyplot as plt


class MomentXTwist:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        # forces
        self.P = data_dict['P']
        self.q = data_dict['q(x,z)']  # Function

        # positions
        self.theta = data_dict['theta']
        self.x_a = data_dict['x_a']
        self.x_1 = data_dict['x_1']
        self.x_2 = data_dict['x_2']
        self.x_3 = data_dict['x_3']
        self.eta = data_dict['eta']
        self.r_a = data_dict['h_a']/2
        self.c_a = data_dict['C_a']
        self.l_a = data_dict['l_a']

        # cross section
        self.G = data_dict['G']
        self.J = data_dict['J']

        self.dx = data_dict['dx']

        self.interpolation_type = data_dict['interpolation_type']
        self.integration_type = data_dict['integration_type']

        self.qzx = None
        self.qzxx = None

        self._M_x = None
        self._theta = None

        self.x_arr = np.arange(0, self.l_a, self.dx)

        self._contrib_dict = {"R_1_y": 0, "R_1_z": 0,
                              "R_2_y": 0, "R_2_z": 0,
                              "R_3_y": 0, "R_3_z": 0,
                              "C_1_z": 0, "C_2_z": 0,
                              "C_1_y": 0, "C_2_y": 0,
                              "R_I": 0, "C_1_x": 0,
                              "P": 0, "aero": 0}

    def contribution_m_x(self, x):
        contrib = self._contrib_dict.copy()

        if x >= self.x_1:
            contrib['R_1_y'] = self.eta
        if x >= (self.x_2 - self.x_a/2):
            contrib['R_I'] = self.r_a * np.cos(self.theta) - np.sin(self.theta) * (self.r_a + self.eta)
        if x >= self.x_2:
            contrib['R_2_y'] = self.eta
        if x >= (self.x_2 + self.x_a/2):
            contrib['P'] = self.r_a * np.cos(self.theta) - np.sin(self.theta) * (self.r_a + self.eta)
        if x >= self.x_3:
            contrib['R_3_y'] = self.eta

        if self.qzx is None:
            self.get_all_integral_functions()
        contrib['aero'] = - self.qzx(x)  # NOT SURE ABOUT MINUS SIGN

        return contrib

    def contribution_theta(self, x):
        contrib = self._contrib_dict.copy()

        if x >= self.x_1:
            contrib['R_1_y'] = self.eta * (x - self.x_1) / (self.G * self.J)
        if x >= (self.x_2 - self.x_a / 2):
            contrib['R_I'] = (self.r_a * np.cos(self.theta) - np.sin(self.theta) * (self.r_a + self.eta)) * \
                             (x - (self.x_2 - self.x_a / 2)) / (self.G * self.J)
        if x >= self.x_2:
            contrib['R_2_y'] = self.eta * (x - self.x_2) / (self.G * self.J)
        if x >= (self.x_2 + self.x_a / 2):
            contrib['P'] = (self.r_a * np.cos(self.theta) - np.sin(self.theta) * (self.r_a + self.eta)) * \
                            (x - (self.x_2 + self.x_a / 2)) / (self.G * self.J)
        if x >= self.x_3:
            contrib['R_3_y'] = self.eta * (x - self.x_3) / (self.G * self.J)

        contrib['C_1_x'] = 1

        if self.qzx is None:
            self.get_all_integral_functions()
        contrib['aero'] = - self.qzxx(x) / (self.G * self.J)  # NOT SURE ABOUT MINUS SIGN

        return contrib

    def get_moment_around_x(self, updated_data_dict):
        if self._M_x is None:
            m_x = [do_dot(updated_data_dict, self.contribution_m_x(x_v)) for x_v in self.x_arr]
            self._M_x = self.interpolation_type(self.x_arr, m_x)
        return self._M_x

    def get_twist_around_x(self, updated_data_dict):
        if self._theta is None:
            twist = [do_dot(updated_data_dict, self.contribution_theta(x_v)) for x_v in self.x_arr]
            self._theta = self.interpolation_type(self.x_arr, twist)
        return self._theta

    def get_all_integral_functions(self):
        """
        This function will do all the integral calculations to define self.qzx and self.qzxx
        """
        # First we need to create the function we want to integrate:
        def integrand(z, x):
            return (z - self.eta)*self.q(x, z)

        def integrate_along_chord(x):
            return self.integration_type(integrand, self.r_a - self.c_a, self.r_a, args=(x, ))

        qz = [integrate_along_chord(x_v)[0] for x_v in self.x_arr]
        qz = self.interpolation_type(self.x_arr, qz)

        def integrate_0_to_x(q, x):
            return self.integration_type(q, 0, x)

        qzx = [integrate_0_to_x(qz, x_v)[0] for x_v in self.x_arr]
        self.qzx = self.interpolation_type(self.x_arr, qzx)

        qzxx = [integrate_0_to_x(self.qzx, x_v)[0] for x_v in self.x_arr]
        self.qzxx = self.interpolation_type(self.x_arr, qzxx)


def do_dot(data: dict, contribution: dict):
    """
    This function will take the dot product of two dictionaries
    """
    value = contribution['aero']
    for key in contribution.keys():
        if key != 'aero':  # Because aero doesn't have to be multiplied by anything
            value += data[key] * contribution[key]
    return value