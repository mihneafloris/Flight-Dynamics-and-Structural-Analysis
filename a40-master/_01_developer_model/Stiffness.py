
""" 
 This is file is part of the developer code of the AE3212-II project. 
 The class defined in this file has methods to calculate the cross-sectional 
 properties of the aileron, i.e. the centroid, moment of inertias, torsional 
 constant and shear center.

 Note that comments are largely missing from this code. The most part of the 
 code should speak for itself. 

 Author: Sam van Elsloo
 Date:   06/02/2021
"""


import sys
import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import scipy as sp
from scipy import interpolate
from scipy import integrate

class Crosssection:
    def __init__(self, nst, Ca, ha, tsk, tsp, tst, hst, wst):
        self.nst = nst
        self.Ca = Ca
        self.ha = ha
        self.tsk = tsk
        self.tsp = tsp
        self.tst = tst
        self.hst = hst
        self.wst = wst

        self.totarea = 0
        self.zc = 0
        self.yc = 0
        self.Izz = 0
        self.Iyy = 0

        self.zsc = 0
        self.ysc = 0
        self.J = 0

    def perimeter(self):
        temp = self.Ca - self.ha / 2
        self.lsk = m.sqrt(temp ** 2 + (self.ha / 2) ** 2)
        lcirc = self.ha * m.pi / 2
        self.lcirc = lcirc
        self.P = lcirc + self.lsk * 2

    def strdistr(self):
        """
         First compute the spacing of the stingers, and how many stringers are 
         on the circular part
        """
        if self.nst % 2 == 0:
            print("Please enter an odd number of stringers.")
            sys.exit(0)
        self.spacing = self.P / self.nst
        self.ncirc   = (self.lcirc / 2 // self.spacing) * 2 + 1
        self.nsk     = self.nst - self.ncirc

        """ 
         Then compute the coordinates of the stringers on the circular part. 
         Origin in leading edge, and leading edge
         stringer is the first stringer, then counting in clockwise direction
        """
        self.stcoord = np.zeros((self.nst, 2))
        circlst      = np.linspace(-self.ncirc // 2 + 1, self.ncirc // 2, 
                                   int(self.ncirc), endpoint=True)
        self.phi     = self.spacing / (2 * self.lcirc) * m.pi * 2

        self.stcoord[0:int(self.ncirc // 2 + 1), 0] = \
            -self.ha/2 * (1-np.cos(circlst[int(self.ncirc//2):]*self.phi))
        self.stcoord[0:int(self.ncirc // 2 + 1), 1] = \
             self.ha/2 *    np.sin(circlst[int(self.ncirc//2):]*self.phi)

        if self.ncirc > 1:
            self.stcoord[-int(self.ncirc // 2):, 0] = \
                -self.ha/2 * (1-np.cos(circlst[:int(self.ncirc//2)]*self.phi))
            self.stcoord[-int(self.ncirc // 2):, 1] = \
                 self.ha/2 *    np.sin(circlst[:int(self.ncirc//2)]*self.phi)

        psi    = np.arctan(self.ha / 2 / (self.Ca - self.ha / 2))
        zcoord = -(self.Ca - np.arange(self.spacing / 2 * np.cos(psi), \
                                       (self.Ca - self.ha / 2), 
                                        self.spacing * np.cos(psi)))
        ycoord = np.arange(self.spacing / 2 * np.sin(psi), self.ha / 2, 
                           self.spacing * np.sin(psi))

        ### np.flipud flips an array (reverses the entries)
        self.stcoord[int(self.ncirc // 2 + 1):int(self.nst // 2 + 1), 0] = \
            np.flipud(zcoord)
        self.stcoord[int(self.ncirc // 2 + 1):int(self.nst // 2 + 1), 1] = \
            np.flipud(ycoord)

        ### Safegaurd for when there is only one stringer on the leading edge
        if self.ncirc == 1:
            self.stcoord[int(self.nst // 2 + 1):, 0] = zcoord
            self.stcoord[int(self.nst // 2 + 1):, 1] = -ycoord
        else:
            self.stcoord[int(self.nst // 2 + 1):-int(self.ncirc // 2), 0] = \
                zcoord
            self.stcoord[int(self.nst // 2 + 1):-int(self.ncirc // 2), 1] = \
               -ycoord

    def areastringer(self):
        self.areast = self.tst * self.hst + self.tst * self.wst

    def sparprop(self):
        self.zsp    = - self.ha / 2.
        self.ysp    = 0
        self.areasp = self.ha*self.tsp
        self.Izzsp  = 1./12.*self.ha**3*self.tsp
        self.Iyysp  = 0

    def skinprop(self):
        temp        = self.Ca - self.ha / 2
        self.lsk    = m.sqrt(temp**2+(self.ha/2)**2)
        self.zsk    = -(self.Ca-temp/2)
        self.ysk    = self.ha/4
        self.areask = self.lsk*self.tsk
        self.Izzsk  = 1./12.*self.lsk*self.tsk*(self.ha/2)**2
        self.Iyysk  = 1./12.*self.lsk*self.tsk*((self.Ca-self.ha/2))**2

    def circprop(self):
        self.zcirc    = -(self.ha/2-2*self.ha/2/m.pi)
        self.ycirc    = 0
        self.areacirc = self.ha*m.pi/2*self.tsk
        self.Izzcirc  = m.pi/2*(self.ha/2)**3*self.tsk
        self.Iyycirc  = m.pi/2*(self.ha/2)**3*self.tsk-\
                        (-self.ha/2-self.zcirc)**2*self.ha*m.pi/2*self.tsk

    def centroid(self):
        self.totarea = self.nst*self.areast+self.areasp+\
            2*self.areask+self.areacirc
        self.Qz = np.sum(self.stcoord[:,0])*self.areast+self.areasp*self.zsp+\
            2*self.areask*self.zsk+self.areacirc *self.zcirc
        self.Qy = np.sum(self.stcoord[:,1])*self.areast+self.areasp*self.ysp+\
            self.areacirc*self.ycirc
        self.zc = self.Qz / self.totarea
        self.yc = self.Qy / self.totarea

    def inertia(self):
        self.Izz  = self.Izzsp + 2*self.Izzsk + self.Izzcirc
        self.Izz += (self.ysp-self.yc)**2*self.areasp + \
            (self.ysk-self.yc)**2*self.areask*2 + \
            (self.ycirc-self.yc)**2*self.areacirc
        """
         np.einsum implements Einstein-summation. Basially, 'i,i->' will 
         calculate the dot-product of two vectors.
        """
        self.Izz += self.areast*np.einsum('i,i->',self.stcoord[:, 1]-self.yc,
                                                  self.stcoord[:,1]-self.yc)

        self.Iyy  = self.Iyysp + 2*self.Iyysk + self.Iyycirc
        self.Iyy += (self.zsp-self.zc)**2*self.areasp + \
            (self.zsk-self.zc)**2*self.areask*2 + \
            (self.zcirc-self.zc)**2*self.areacirc
        self.Iyy += self.areast*np.einsum('i,i->', self.stcoord[:, 0]-self.zc, 
                                                   self.stcoord[:, 0]-self.zc)

    def compute_bending_properties(self):
        self.perimeter()
        self.strdistr()
        self.areastringer()
        self.sparprop()
        self.skinprop()
        self.circprop()
        self.centroid()
        self.inertia()

    def plot_crosssection(self):
        fig, ax = plt.subplots()
        ax.plot(self.stcoord[:, 0], self.stcoord[:, 1], 'xr')
        ax.plot((-self.Ca, -self.ha / 2),     (0, self.ha / 2),  color="black")
        ax.plot((-self.Ca, -self.ha / 2),     (0, -self.ha / 2), color="black")
        ax.plot((-self.ha / 2, -self.ha / 2), (-self.ha / 2, self.ha / 2), 
                color="black")
        ax.plot(self.zc, self.yc, 'x')
        semicircle = ptc.Arc((-self.ha / 2, 0), self.ha, self.ha, 
                             theta1=-90, theta2=-270, color="black")
        upperskin = ptc.ConnectionPatch((self.Ca, 0), 
                                        (self.Ca - self.ha / 2, self.ha / 2), 
                                        "data")
        ax.add_patch(semicircle)
        ax.set_xlabel(r'$z$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        string = 'Cross-section of aileron'
        ax.set_title(string)
        plt.gca().invert_xaxis()
        plt.show()

    def compute_shearcenter(self):
        k = 1/self.Izz
        h = self.ha/2

        """ Base shear flow """
        ### Base shear flow in area (1)
        phi1list = []
        ilist = []
        boom1list = []
        boomeffect1list = []
        for i in range(0,int(self.ncirc//2)+1):
            phi1list.append(self.phi*i)
            ilist.append(i)
            boomeffect = 0
            for j in range(i+1):
                boomeffect += self.areast * self.stcoord[j,1]
            boom1list.append(self.areast*self.stcoord[i,1])
            boomeffect1list.append(boomeffect)

        """ 
         scipy.interpolate here is used to easily create a continuous function
         out of a discrete data-set.
        """
        boomf        = sp.interpolate.interp1d(phi1list,ilist,kind="previous",
                                               fill_value="extrapolate")
        boomeffect1f = sp.interpolate.interp1d(phi1list,boomeffect1list,
                                               kind="previous",
                                               fill_value="extrapolate")

        ### Base shear flow in area (1)
        def qb1f(theta):
            return k*(self.tsk * h**2*(np.cos(theta)-1) - boomeffect1f(theta))

        ### Base shear flow in area (2)
        def qb2f(y):
            return -k*self.tsp/2.*y**2

        ### Base shear flow in area (3)
        qb10 = qb1f(m.pi/2)
        qb20 = qb2f(h)

        s3list = [0]
        ilist = [0]
        boom3list = [0]
        boomeffect3list = [0]
        for i in range(int(self.ncirc // 2 + 1),int(self.nst // 2 + 1)):
            s3list.append(self.lsk + (i - int(self.nst//2)-0.5)*self.spacing)
            ilist.append(i - int(self.ncirc // 2 ))
            boomeffect = 0
            for j in range(int(self.ncirc // 2 + 1),i+1):
                boomeffect += self.areast * self.stcoord[j,1]
            boom3list.append(self.areast*self.stcoord[i,1])
            boomeffect3list.append(boomeffect)
        boomf        = sp.interpolate.interp1d(s3list,ilist,kind="previous",
                                               fill_value="extrapolate")
        boomeffect3f = sp.interpolate.interp1d(s3list,boomeffect3list,
                                               kind="previous",
                                               fill_value="extrapolate")

        def qb3f(s):
            return -k*(self.tsk*(h*s-h/self.lsk/2.*s**2) + boomeffect3f(s)) + \
                    qb10 + qb20

        ### Base shear flow in area (4)
        qb30 = qb3f(self.lsk)

        s4list = [0]
        ilist = [0]
        boom4list = [0]
        boomeffect4list = [0]
        for i in range(int(self.nst // 2 + 1),self.nst-int(self.ncirc // 2)):
            s4list.append((i - int(self.nst//2)-0.5)*self.spacing)
            ilist.append(i - int(self.nst//2+1))
            boomeffect = 0
            for j in range(int(self.nst // 2 + 1),i+1):
                boomeffect += self.areast * self.stcoord[j,1]
            boom4list.append(self.areast*self.stcoord[i,1])
            boomeffect4list.append(boomeffect)

        boomf        = sp.interpolate.interp1d(s4list,ilist,kind="previous",
                                               fill_value="extrapolate")
        boomeffect4f = sp.interpolate.interp1d(s4list,boomeffect4list,
                                               kind="previous",
                                               fill_value="extrapolate")

        def qb4f(s):
            return -k*(self.tsk*(-h/self.lsk/2.*s**2) + boomeffect4f(s)) + qb30


        ### Base shear flow in area (5)
        def qb5f(y):
            return -k*self.tsp/2.*y**2

        ### Base sehar flow in area (6)
        qb40 = qb4f(self.lsk)
        qb50 = qb5f(-h)

        phi6list = [-m.pi/2]
        ilist = [0]
        boom6list = [0]
        boomeffect6list = [0.]
        for i in range(self.nst-int(self.ncirc // 2),self.nst):
            phi6list.append(self.phi*(i-self.nst))
            ilist.append(i-self.nst-int(self.ncirc // 2))
            boomeffect = 0
            for j in range(self.nst-int(self.ncirc // 2),i+1):
                boomeffect += self.areast * self.stcoord[j,1]
            boom6list.append(self.areast*self.stcoord[i,1])
            boomeffect6list.append(boomeffect)

        boomf = sp.interpolate.interp1d(phi6list,ilist,kind="previous",
                                        fill_value="extrapolate")
        boomeffect6f = sp.interpolate.interp1d(phi6list,boomeffect6list,
                                               kind="previous",
                                               fill_value="extrapolate")

        def qb6f(theta):
            return k*(self.tsk * h**2*(np.cos(theta)) - boomeffect6f(theta)) + \
                   qb40 - qb50

        """
         Redundant shear flow. b1 is the rate of twist in left cell due to base 
         shear flow in left cell, b2 the rate of twist in right cell due to base
         shear flow in right cell.
        """
        ## Influence of region (1); analytical integral of q1intf(theta)
        def qb1intf(theta):
            initial = k*(self.tsk*(h**3*(np.sin(theta)-theta)))
            for i,phi in enumerate(phi1list):
                if theta >= phi:
                    initial -= k*(h*boom1list[i]*(theta-phi))
            return initial

        ## Influence of region (2); analytical integral of q2intf(y)
        def qb2intf(y):
            return -k*self.tsp/6.*(y**3)

        ## Influence of region (3); analytical integral of q3intf(y)
        def qb3intf(s):
            initial = -k*(self.tsk*(h/2.*s**2-h/self.lsk/6.*s**3)) + \
                       (qb10 + qb20)*s
            for i,s0 in enumerate(s3list):
                if s > s0:
                    initial -= k*(boom3list[i]*(s-s0))
            return initial

        ## Influence of region (4); analytical integral of q4intf(y)
        def qb4intf(s):
            initial = -k*(self.tsk*(-h/self.lsk/6.*s**3)) + (qb30)*s
            for i,s0 in enumerate(s4list):
                if s > s0:
                    initial -= k*(boom4list[i]*(s-s0))
            return initial

        ## Influence of region (5); analytical integral of q5intf(y)
        def qb5intf(y):
            return k*self.tsp/6.*y**3

        ## Influence of region (6); analytical integral of q6intf(y)
        def qb6intf(theta):
            initial = k*(self.tsk*(h**3*(np.sin(theta)+1))) + \
                      (qb40 - qb50)*(theta+m.pi/2)*h
            for i,phi in enumerate(phi6list):
                if theta >= phi:
                    initial -= k*(h*boom6list[i]*(theta-phi))
            return initial

        """
         Calculate actual redundant shear flow; take different approach based
         on multicell or not.
        """

        if self.tsp > 0:
            b = np.array([0., 0.])

            print(qb5intf(-h), qb2intf(h))
            b[0] = qb1intf(m.pi / 2) / self.tsk - qb2intf(h) / self.tsp - \
                   qb5intf(-h) / self.tsp + qb6intf(0) / self.tsk
            b[1] = qb2intf(h) / self.tsp + qb3intf(self.lsk) / self.tsk + \
                   qb4intf(self.lsk) / self.tsk + qb5intf(-h) / self.tsp

            A = np.array([[0., 0.], [0., 0.]])
            A[0, 0] = h * m.pi / self.tsk + 2 * h / self.tsp
            A[0, 1] = -2 * h / self.tsp
            A[1, 0] = -2 * h / self.tsp
            A[1, 1] = 2 * self.lsk / self.tsk + 2 * h / self.tsp

            qredundant = np.linalg.solve(A, -b)

        else:
            b = qb1intf(m.pi / 2) / self.tsk + qb3intf(self.lsk) / self.tsk + \
                qb4intf(self.lsk) / self.tsk + qb6intf(0) / self.tsk
            A = h * m.pi / self.tsk + 2 * self.lsk / self.tsk

            qredundant = -b / A * np.ones(2)

        ### Sum the base shear flow and redundant shear flow.
        def q1f(theta):
            return qb1f(theta) + qredundant[0]

        def q2f(y):
            return qb2f(y) - qredundant[0] + qredundant[1]

        def q3f(s):
            return qb3f(s) + qredundant[1]

        def q4f(s):
            return qb4f(s) + qredundant[1]

        def q5f(y):
            return qb5f(y) - qredundant[0] + qredundant[1]

        def q6f(theta):
            return qb6f(theta) + qredundant[0]

        angle = m.atan(h/(self.Ca-h))
        r     = m.sin(angle) * (self.Ca-h)

        internalbasemoment      = qb1intf(m.pi/2)*h + qb3intf(self.lsk)*r + \
                                  qb4intf(self.lsk)*r + qb6intf(0)*h
        internalredundantmoment = qredundant[0]*m.pi*h*h + \
                                  qredundant[1]*self.lsk*r*2

        externalmoment = internalbasemoment + internalredundantmoment
        eta            = -externalmoment
        self.zsc       = -(eta + self.ha/2.)

    def compute_torsionalstiffness(self):
        if self.tsp > 0:
            h = self.ha/2
            A1 = m.pi*h**2/2.
            A2 = (self.Ca-h)*h

            A = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
            b = np.array([0.,0.,0.])

            ### First row
            A[0,0] = 2.*A1
            A[0,1] = 2.*A2
            b[0] = 1

            ### Second row
            A[1,0] = (h*m.pi/self.tsk + 2*h/self.tsp)/(2*A1)
            A[1,1] = (-2*h/self.tsp)/(2*A1)
            A[1,2] = -1.
            b[1] = 0.

            ### Third row
            A[2,0] = (-2*h/self.tsp)/(2*A2)
            A[2,1] = (2*self.lsk/self.tsk + 2*h/self.tsp)/(2*A2)
            A[2,2] = -1
            b[2] = 0.

            solution = np.linalg.solve(A,b)
            self.J = 1./solution[-1]
            
            print("J is:",self.J)
        else:
            h = self.ha/2.
            b = 1
            A = h*m.pi/self.tsk + 2*self.lsk/self.tsk

            self.J = 1/(b/A)