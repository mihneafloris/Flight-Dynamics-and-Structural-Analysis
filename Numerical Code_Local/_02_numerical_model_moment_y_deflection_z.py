import numpy as np


class MomentYDeflectionZ:
    def __init__(self, data_dict):
        self.x_1 = data_dict["x_1"]
        self.x_2 = data_dict["x_2"]
        self.x_3 = data_dict["x_3"]
        self.x_a = data_dict["x_a"]
        self.l_a = data_dict['l_a']

        self.dx = data_dict['dx']

        self.theta = data_dict["theta"]

        self.P = data_dict["P"]
        self.q = data_dict["q(x,z)"]

        self.E = data_dict["E"]
        self.I_yy = data_dict["I_yy"]

        self._M_y = None
        self._S_z = None
        self._v_z = None

        self.x_arr = np.arange(0, self.l_a, self.dx)

        self._contrib_dict = {"R_1_y": 0, "R_1_z": 0,
                              "R_2_y": 0, "R_2_z": 0,
                              "R_3_y": 0, "R_3_z": 0,
                              "C_1_z": 0, "C_2_z": 0,
                              "C_1_y": 0, "C_2_y": 0,
                              "R_I": 0, "C_1_x": 0,
                              "P": 0, "aero": 0}

        self.interpolation_type = data_dict['interpolation_type']
        self.integration_type = data_dict['integration_type']

    def contribution_m_y(self, x):
        contrib = self._contrib_dict.copy()

        if x >= self.x_1:
            contrib['R_1_z'] = - (x - self.x_1)
        if x >= (self.x_2 - self.x_a/2):
            contrib['R_I'] = x - (self.x_2 - self.x_a/2)
        if x >= self.x_2:
            contrib['R_2_z'] = - (x - self.x_2)
        if x >= (self.x_2 + self.x_a / 2):
            contrib['R_I'] = x - (self.x_2 + self.x_a/2)
        if x >= self.x_3:
            contrib['R_3_z'] = - (x - self.x_3)

        return contrib

    def contribution_s_z(self, x):
        contrib = self._contrib_dict.copy()

        if x >= self.x_1:
            contrib['R_1_z'] = -1
        if x >= (self.x_2 - self.x_a / 2):
            contrib['R_I'] = 1
        if x >= self.x_2:
            contrib['R_2_z'] = -1
        if x >= (self.x_2 + self.x_a / 2):
            contrib['P'] = 1
        if x >= self.x_3:
            contrib['R_3_z'] = -1

        return contrib

    def contribution_v_z(self, x):
        contrib = self._contrib_dict.copy()

        ei6 = 6 * self.E * self.I_yy

        if x >= self.x_1:
            contrib['R_1_z'] = (x - self.x_1)**3 / ei6
        if x >= (self.x_2 - self.x_a/2):
            contrib['R_I'] = - (x - (self.x_2 - self.x_a/2))**3 / ei6
        if x >= self.x_2:
            contrib['R_2_z'] = (x - self.x_2)**3 / ei6
        if x >= (self.x_2 + self.x_a/2):
            contrib['P'] = - (x - (self.x_2 + self.x_a/2))**3 / ei6
        if x >= self.x_3:
            contrib['R_3_z'] = (x - self.x_3)**3 / ei6

        contrib['C_1_z'] = x
        contrib['C_2_z'] = 1

        return contrib

    def get_moment_around_y(self, updated_data_dict):
        if self._M_y is None:
            m_z = [do_dot(updated_data_dict, self.contribution_m_y(x_v)) for x_v in self.x_arr]
            self._M_y = self.interpolation_type(self.x_arr, m_z)
        return self._M_y

    def get_shear_force_in_z(self, updated_data_dict):
        if self._S_z is None:
            s_z = [do_dot(updated_data_dict, self.contribution_s_z(x_v)) for x_v in self.x_arr]
            self._S_z = self.interpolation_type(self.x_arr, s_z)
        return self._S_z

    def get_deflection_in_z(self, updated_data_dict):
        if self._v_z is None:
            # If it is none, then we need to make it, for which we use "get_contribution_m_z"
            v_z = [do_dot(updated_data_dict, self.contribution_v_z(x_v)) for x_v in self.x_arr]
            self._v_z = self.interpolation_type(self.x_arr, v_z)
        return self._v_z


def do_dot(data: dict, contribution: dict):
    """
    This function will take the dot product of two dictionaries
    """
    value = contribution['aero']
    for key in contribution.keys():
        if key != 'aero':  # Because aero doesn't have to be multiplied by anything
            value += data[key] * contribution[key]
    return value
