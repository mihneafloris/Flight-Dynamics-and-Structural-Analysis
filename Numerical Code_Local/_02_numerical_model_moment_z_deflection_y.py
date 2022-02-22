import matplotlib.pyplot as plt
import numpy as np


class MomentZDeflectionY:
    def __init__(self, data_dict):
        self.x_1 = data_dict['x_1']
        self.x_2 = data_dict['x_2']
        self.x_3 = data_dict['x_3']
        self.x_a = data_dict['x_a']
        self.l_a = data_dict['l_a']
        self.c_a = data_dict['C_a']
        self.r = data_dict['h_a']/2
        self.d_1 = data_dict['d_1']
        self.d_3 = data_dict['d_3']
        self.theta = data_dict['theta']

        self.I_zz = data_dict['I_zz']
        self.E = data_dict['E']

        self.q = data_dict['q(x,z)']  # This should be a function
        self.P = data_dict['P']

        self.dx = data_dict['dx']

        self.interpolation_type = data_dict['interpolation_type']
        self.integration_type = data_dict['integration_type']

        # The underscore means that they are private variables which should not be accessed from outside.
        # After certain functions get called, these will all become functions.
        self._M_z = None
        self._S_y = None
        self._v_y = None

        self._contrib_dict = {"R_1_y": 0, "R_1_z": 0,
                              "R_2_y": 0, "R_2_z": 0,
                              "R_3_y": 0, "R_3_z": 0,
                              "C_1_z": 0, "C_2_z": 0,
                              "C_1_y": 0, "C_2_y": 0,
                              "R_I": 0, "C_1_x": 0,
                              "P": 0, "aero": 0}

        assert len(self._contrib_dict) == 14

        self.x_arr = np.arange(0, self.l_a, self.dx)

        # The following variables are used to cache the functions because calculating them every time is too slow.
        self.qzx = None
        self.qzxx = None
        self.qzxxxx = None

    def contribution_m_z(self, x):
        """
        This function will return a dictionary (which is to become a vector) that will contribute to the matrix used
        for solving for the reaction forces.
        After we have the actual reaction force values, this function can be used to create the moments, although some
        modifications will have to be made.
        """
        contrib_dict = self._contrib_dict.copy()
        if x >= self.x_1:
            contrib_dict["R_1_y"] = - (x - self.x_1)
        if x >= (self.x_2 - self.x_a / 2):
            contrib_dict["R_I"] = np.sin(self.theta) * (x - (self.x_2 - self.x_a / 2))
        if x >= self.x_2:
            contrib_dict["R_2_y"] = - (x - self.x_2)
        if x >= (self.x_2 + self.x_a / 2):
            contrib_dict["P"] += np.sin(self.theta) * (x - (self.x_2 + self.x_a / 2))
        if x >= self.x_3:
            contrib_dict["R_3_y"] = - (x - self.x_3)

        if self.qzxx is None:
            self.get_all_integral_functions()

        contrib_dict["aero"] += self.qzxx(x)

        return contrib_dict

    def contribution_s_y(self, x):
        """
        This function will return a dictionary (which can easily be transformed into the correct vector) whose contents
        are to contribute for solving the reaction forces using boundary conditions.
        Note the similarities with the "get_matrix_contribution_m_z" function
        """
        contrib_dict = self._contrib_dict.copy()

        if x >= self.x_1:
            contrib_dict['R_1_y'] = -1
        if x >= (self.x_2 - self.x_a / 2):
            contrib_dict["R_I"] = np.sin(self.theta)
        if x >= self.x_2:
            contrib_dict["R_2_y"] = -1
        if x >= (self.x_2 + self.x_a / 2):
            contrib_dict["P"] = np.sin(self.theta)
        if x >= self.x_3:
            contrib_dict["R_3_y"] = -1

        if self.qzx is None:
            self.get_all_integral_functions()

        contrib_dict["aero"] = self.qzx(x)

        return contrib_dict

    def contribution_v_y(self, x):
        contrib = self._contrib_dict.copy()

        if x >= self.x_1:
            contrib["R_1_y"] = (x - self.x_1) ** 3 / (6 * self.E * self.I_zz)
        if x >= (self.x_2 - self.x_a / 2):
            contrib["R_I"] = - np.sin(self.theta) * (x - (self.x_2 - self.x_a / 2)) ** 3 / (6 * self.E * self.I_zz)
        if x >= self.x_2:
            contrib["R_2_y"] = (x - self.x_2) ** 3 / (6 * self.E * self.I_zz)
        if x >= (self.x_2 + self.x_a / 2):
            contrib["P"] = - np.sin(self.theta) * (x - (self.x_2 + self.x_a / 2)) ** 3 / (6 * self.E * self.I_zz)
        if x >= self.x_3:
            contrib["R_3_y"] = (x - self.x_3) ** 3 / (6 * self.E * self.I_zz)

        contrib["C_1_y"] = x
        contrib["C_2_y"] = 1

        # That leaves only the pain that is the quintuple integration. However, I put all that in a separate function.
        if self.qzxxxx is None:
            self.get_all_integral_functions()
        contrib["aero"] = - self.qzxxxx(x) / (self.E * self.I_zz)

        return contrib

    def get_moment_around_z(self, updated_data_dict):
        if self._M_z is None:
            m_z = [do_dot(updated_data_dict, self.contribution_m_z(x_v)) for x_v in self.x_arr]
            self._M_z = self.interpolation_type(self.x_arr, m_z)

        return self._M_z

    def get_deflection_in_y(self, updated_data_dict):
        if self._v_y is None:
            v_y = [do_dot(updated_data_dict, self.contribution_v_y(x_v)) for x_v in self.x_arr]
            self._v_y = self.interpolation_type(self.x_arr, v_y)
        return self._v_y

    def get_shear_force_in_y(self, updated_data_dict):
        if self._S_y is None:
            s_y = [do_dot(updated_data_dict, self.contribution_s_y(x_v)) for x_v in self.x_arr]
            self._S_y = self.interpolation_type(self.x_arr, s_y)
        return self._S_y

    def get_all_integral_functions(self):
        """
        This function will do all the integral calculations to define self.qzx, self.qzxx and self.qzxxxx
        """

        # The first thing we need to do is transform q(x,z) into q(z, x) as we will integrate first over z (scipy thing)
        def q_func(z_var, x_var):
            return self.q(x_var, z_var)

        def integrate_over_chord(x_var):
            return self.integration_type(q_func, self.r - self.c_a, self.r, args=(x_var,))

        # --- First integration ---
        qz = [integrate_over_chord(x_v)[0] for x_v in self.x_arr]
        qz = self.interpolation_type(self.x_arr, qz)

        def integrate_from_0_to_x(q, x_var):  # q needs to be only a function of x
            return self.integration_type(q, 0, x_var)

        # --- Second integration ---
        qzx = [integrate_from_0_to_x(qz, x_v)[0] for x_v in self.x_arr]
        self.qzx = self.interpolation_type(self.x_arr, qzx)

        # --- Third integration ---
        qzxx = [integrate_from_0_to_x(self.qzx, x_v)[0] for x_v in self.x_arr]
        self.qzxx = self.interpolation_type(self.x_arr, qzxx)

        # --- Forth and fifth integrations
        qz3x = [integrate_from_0_to_x(self.qzxx, x_v)[0] for x_v in self.x_arr]
        qz3x = self.interpolation_type(self.x_arr, qz3x)

        qz4x = [integrate_from_0_to_x(qz3x, x_v)[0] for x_v in self.x_arr]
        self.qzxxxx = self.interpolation_type(self.x_arr, qz4x)

        # pfew, this takes a long time to compute


def do_dot(data: dict, contribution: dict):
    """
    This function will take the dot product of two dictionaries
    """

    value = contribution['aero']
    for key in contribution.keys():
        if key != 'aero':  # Because aero doesn't have to be multiplied by anything
            value += data[key] * contribution[key]
    return value
