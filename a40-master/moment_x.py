import numpy as np
from scipy.integrate import nquad
from scipy.integrate import simps
import _02_numerical_model.III_aeroload as ae

def macaulay(x, x0, val):
    if (x - x0) <= 0:
        return 0
    else:
        return (x-x0)**val

class Moment_x:

    def __init__(self, dict):
        '''
        :param dict:

        :param eta: position of shear center on z-axis
        :param c: chord length

        :param dA: length of segment A divided by its thickness
        :param dB: length of segment B divided by its thickness
        :param dC: length of segment C divided by its thickness
        :param dD: length of segment D divided by its thickness
        :param A1: area of cell 1
        :param A2: area of cell 2
        :param t: skin thickness
        '''

        self.dict = dict

        #forces
        self.P = dict['P']
        self.Ri = dict['Ri']
        self.R1y = dict['R1y']
        self.R2y = dict['R2y']
        self.R3y = dict['R3y']

        #positions
        self.theta = dict['theta']
        self.dpy = dict['dpy']
        self.dpz = dict['dpz']
        self.dry = dict['dry']
        self.drz = dict['drz']
        self.xa = dict['xa']
        self.x1 = dict['x1']
        self.x2 = dict['x2']
        self.x3 = dict['x3']
        self.eta = dict['eta']
        self.c = dict['c']

        #cross section
        self.dA = dict['dA']
        self.dB = dict['dB']
        self.dC = dict['dC']
        self.dD = dict['dD']
        self.A1 = dict['A1']
        self.A2 = dict['A2']
        self.t = dict['t']
        self.G = dict['G']
        self.J = dict['J']

    def q(self, x, z):
        '''
        :param x: position along x-axis
        :param z: position along z-axis
        :return: aerodynamic load which has to be integrated over dx and dz
        '''
        f = ae.get_loadfunction(dict)
        return f(x, z)*(z - self.eta)

    def moment_x(self, x):
        '''
        :param x: position along x-axis
        :return: Moment around x-axis at x
        '''
        M = nquad(self.q, [0, self.c], [0, x]) \
            + self.R1y * self.eta * macaulay(x, self.x1, 0) + self.R2y * self.eta * macaulay(x, self.x2, 0) \
            + self.R3y * self.eta * macaulay(x, self.x3, 0) \
            + self.Ri*(self.r*np.cos(self.theta) - np.sin(self.theta)*(self.eta+self.r))*macaulay(x, self.x2+self.xa/2, 0) \
            + self.P

        return M

    def shear_flow(self, x):
        '''
        :param x: position along x-axis
        :return: shear flow q1 and q2
        '''
        T = self.moment_x(x)
        M = np.array([[T/2],
                      [0]])
        a = np.array([[self.A2*(self.dA + self.dB) + self.A1*self.dA, -self.A2*self.dA - (self.dA+self.dC+self.dD)*self.A1],
                      [self.A1, self.A2]])

        q = np.matmul(np.linalg.inv(a), M)
        return q

    def dThetadx(self, x):
        '''
        :param x: position along x-axis
        :return: d(theta)/dx at x
        '''
        dthetadx = (1/self.J/self.G)*self.moment_x(x)
        return dthetadx

    def TotalTwist(self, x_points):
        '''
        :param x_points: set of positions along x-axis
        :return: total twist of the aileron
        '''

        dtdx = []

        for x in x_points:
            dtdx.append(self.dThetadx(x))

        totalTwist = simps(dtdx, x_points)

        return totalTwist
