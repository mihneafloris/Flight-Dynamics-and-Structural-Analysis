import math as m
import numpy as np

from scipy import interpolate

class Aeroload:
    def __init__(self, data_dict):
        """Store the data dictionary as a class variable"""
        self.data_dict = data_dict


    def calc_xrange(self, n, l_a):
        """Calculates the range of x positions as is described in section 3.2.1 of the simulation plan."""
        theta = []
        for i in range(0, n + 1):
            theta.append(i * m.pi / n)
        xrange = []
        for i in range(0, n):
            xrange.append(l_a / 4 * ((1 - m.cos(theta[i])) + (1 - m.cos(theta[i + 1]))))
        return xrange


    def calc_zrange(self, n, c_a, h):
        """Calculated the range of z positions as is described in section 3.2.1 of the simulation plan."""
        theta = []
        for i in range(0, n + 1):
            theta.append(i * m.pi / n)
        zrange = []
        for i in range(0, n):
            zrange.append(h / 2 - c_a / 4 * ((1 - m.cos(theta[i])) + (1 - m.cos(theta[i + 1]))))
        return zrange

    def get_loadfunction(self):
        """Returns a SciPy 2D interpolated function which represents the aerodynamic load function q(x,z)"""
        data = np.loadtxt(self.data_dict["aerodynamicload_path"], delimiter=',') * 10 ** 6  # convert from kN/mm^2 to kN/m^2
        self.data_dict["q(x,z)"] = interpolate.interp2d(self.calc_xrange(data.shape[1], self.data_dict["l_a"]), self.calc_zrange(data.shape[0],
                                    self.data_dict["C_a"], self.data_dict["h_a"]), data, kind=self.data_dict["interpolation_type_2d"])
        return self.data_dict

