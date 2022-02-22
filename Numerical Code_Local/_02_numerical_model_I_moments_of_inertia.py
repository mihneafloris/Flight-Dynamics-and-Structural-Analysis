import math as m
from matplotlib import pyplot as plt
import numpy as np

class MomentsOfInertia:
    def __init__(self, data_dict):
        """Initiate class"""

        # Set input variables
        self.ca = data_dict['C_a']
        self.ha = data_dict['h_a']
        self.tsk = data_dict['t_sk']
        self.tsp = data_dict['t_sp']
        self.tst = data_dict['t_st']
        self.hst = data_dict['h_st']
        self.wst = data_dict['w_st']
        self.nst = data_dict['n_st']

        # Set empty variables
        self.centroid_y = None
        self.centroid_z = None
        self.totalarea = None

    def calculate_centroid(self, data_dict, return_dict=False):
        """Calculating the location of the centroid, returning an array containing x and z position"""
        # Centroid in y direction is located at 0
        self.centroid_y = 0

        # Calculate total area and perimeter
        self.arc = m.pi * self.ha / 2 * self.tsk
        self.sheet = m.sqrt((self.ca - self.ha / 2) ** 2 + (self.ha / 2) ** 2) * self.tsk
        self.stringer = self.tst * (self.hst + self.wst)
        self.spar = self.tsp * self.ha
        self.totalarea = self.arc + 2 * self.sheet + self.nst * self.stringer + self.spar
        self.perimeter = 2 * self.ha / 2 * m.pi / 2 + 2 * m.sqrt((self.ca - self.ha / 2) ** 2 + (self.ha / 2) ** 2)
        self.stringerspacing = self.perimeter / self.nst

        # Calculate centroid, respective to the center of the spar
        self.totalarea_dis = 2 * self.ha / m.pi / 2 * self.arc + 2 * -1 * (
                    self.ca - self.ha / 2) / 2 * self.sheet + 0 * self.spar  # Area distance from skin and spar
        self.ncirc = 2 * m.floor(self.ha * m.pi / 4 / self.stringerspacing) + 1
        self.stringerspacing_radians = self.stringerspacing / self.ha / m.pi * 2 * m.pi
        self.stringerlocations_y = []
        self.stringerlocations_z = []
        self.stringerlocations_y.append(0)
        self.stringerlocations_z.append(self.ha / 2)
        self.totalarea_dis += self.stringer * self.ha / 2  # Stringer leading edge
        for i in range((self.ncirc - 1) // 2):  # Stringers on arc
            z = self.ha / 2 * m.cos((i + 1) * self.stringerspacing_radians)
            y = self.ha / 2 * m.sin((i + 1) * self.stringerspacing_radians)
            self.stringerlocations_y.append(y)
            self.stringerlocations_z.append(z)
            self.stringerlocations_y.append(-y)
            self.stringerlocations_z.append(z)
            self.totalarea_dis += z * self.stringer * 2  # Add twice for top and bottom
        self.sheetangle = m.atanh(self.ha / 2 / (self.ca - self.ha / 2))
        for i in range((self.nst - self.ncirc) // 2):  # Stringers on the sheet
            z = self.ca - self.ha / 2 - i * self.stringerspacing * m.cos(
                self.sheetangle) - self.stringerspacing / 2 * m.cos(self.sheetangle)
            y = i * self.stringerspacing * m.sin(self.sheetangle) + self.stringerspacing / 2 * m.sin(self.sheetangle)
            self.stringerlocations_y.append(y)
            self.stringerlocations_z.append(-z)
            self.stringerlocations_y.append(-y)
            self.stringerlocations_z.append(-z)
            self.totalarea_dis += -z * self.stringer * 2  # Add twice for top and bottom
        self.centroid_z = self.totalarea_dis / self.totalarea

        if return_dict:
            data_dict['centroid_z'] = self.centroid_z
            data_dict['centroid_y'] = self.centroid_y
            return data_dict

        return self.centroid_z, self.centroid_y

    def calculate_moments_of_inertia_izz(self, data_dict):
        """Calculating the moments of inertia around the z axis"""
        r = self.ha / 2
        l = self.ca - r
        self.i_zz = m.pi * r ** 3 * self.tsk / 2 + 2 * r ** 3 * self.tsp / 3 +  self.tsk*m.sqrt(l**2 + r**2) * r**2#2* self.tsk * l ** 3 / 12 * (r / m.sqrt(l ** 2 + r ** 2)) ** 2
        for i in range(len(self.stringerlocations_z)):  # Append per stringer
            self.i_zz += self.stringer * (self.stringerlocations_y[i] ** 2)
        data_dict["I_zz"] = self.i_zz
        return data_dict,self.i_zz

    def calculate_moments_of_inertia_iyy(self, data_dict):
        """Calculating the moments of inertia around the x axis"""
        r = self.ha / 2
        l = self.ca - r
        self.i_yy = m.pi * self.tsk * r ** 3 / 2 + m.pi * r * self.tsk * (2 * r / m.pi - self.centroid_z) ** 2 + \
                    self.tsk*m.sqrt(l**2+r**2)*l**2/6 + 2 * self.tsk * m.sqrt(r ** 2 + l ** 2) * (
                                -l / 2 - self.centroid_z) ** 2 + 2*r*self.tsp*self.centroid_z**2 #self.tsk * (r ** 2 + l ** 2) ** (3 / 2) * l / (6 * r)
        for i in range(len(self.stringerlocations_z)):  # append per stringer
            self.i_yy += self.stringer * ((self.stringerlocations_z[i] - self.centroid_z) ** 2)
        data_dict["I_yy"] = self.i_yy
        return data_dict,self.i_yy

    def plot(self):
        """Plotting the cross-section for verification"""
        plt.scatter(self.stringerlocations_z, self.stringerlocations_y, marker='D', color='r')  # Stringers
        plt.scatter(self.centroid_z, self.centroid_y)  # Centroid
        plt.plot([-self.ca + self.ha / 2, 0], [0, self.ha / 2], [-self.ca + self.ha / 2, 0], [0, -self.ha / 2], [0, 0],
                 [self.ha / 2, -self.ha / 2])  # Sheets
        # Creation of the arc
        z = []
        y = []
        for i in range(31):
            z.append(self.ha / 2 * m.cos(m.pi / 2 - i / 30 * m.pi))
            y.append(self.ha / 2 * m.sin(m.pi / 2 - i / 30 * m.pi))
        plt.plot(z, y)  # Arc
        plt.show()

