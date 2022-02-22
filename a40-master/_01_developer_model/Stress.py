
""" 
 This is file is part of the developer code of the AE3212-II project. 
 The class defined in this file has methods to calculate the stress-state in the
 aileron for a given loading.

 Note that comments are largely missing from this code. The most part of the 
 code should speak for itself. Note that large parts of the shear flow 
 calculations share great similarity with those found in the stiffness module.

 Author: Sam van Elsloo
 Date:   06/02/2021
"""


import math as m
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
from scipy import interpolate
import matplotlib.collections as mcoll
import matplotlib.path as mpath

class Stressstate:
    def __init__(self,CS):
        self.nst = CS.nst
        self.Ca = CS.Ca
        self.ha = CS.ha
        self.tsk = CS.tsk
        self.tsp = CS.tsp
        self.tst = CS.tst
        self.hst = CS.hst
        self.wst = CS.wst

        self.ncirc = CS.ncirc
        self.phi = CS.phi
        self.areast = CS.areast
        self.stcoord = CS.stcoord
        self.lsk = CS.lsk
        self.spacing = CS.spacing

        self.totarea = CS.totarea
        self.zc = CS.zc
        self.yc = CS.yc
        self.Izz = CS.Izz
        self.Iyy = CS.Iyy

        self.zsc = CS.zsc
        self.ysc = CS.ysc
        self.J = CS.J

    def compute_unitstressdistributions(self):
        self.compute_unitshearflowdistributionSz()
        self.compute_unitshearflowdistributionSy()
        self.compute_unitshearflowdistributionT()
        self.compute_unitstressdistributionMy()
        self.compute_unitstressdistributionMz()

    def compute_unitshearflowdistributionSz(self):
        k = 1/self.Iyy
        h = self.ha/2

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
                boomeffect += self.areast * (self.stcoord[j,0] - self.zc)
                if j == 0:
                    boomeffect /= 2.
            boom1list.append(self.areast*(self.stcoord[i,0] - self.zc))
            boomeffect1list.append(boomeffect)

        boomeffect1f = sp.interpolate.interp1d(phi1list,boomeffect1list,
                                               kind="previous",
                                               fill_value="extrapolate")

        def qb1f(theta):
            return -k*(self.tsk * h*(-h*theta+h*np.sin(theta)-theta*self.zc) + \
                    boomeffect1f(theta))

        ### Base shear flow in area (2)
        def qb2f(y):
            return -k*self.tsp*(-h-self.zc)*y

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
                boomeffect += self.areast *(self.stcoord[j,0] - self.zc)
            boom3list.append(self.areast*(self.stcoord[i,0] - self.zc))
            boomeffect3list.append(boomeffect)
        boomf = sp.interpolate.interp1d(s3list,ilist,kind="previous",
                                        fill_value="extrapolate")
        boomeffect3f = sp.interpolate.interp1d(s3list,boomeffect3list,
                                               kind="previous",
                                               fill_value="extrapolate")

        def qb3f(s):
            return -k*(self.tsk*((-h-self.zc)*s-(self.Ca-h)/self.lsk/2.*s**2) +\
                    boomeffect3f(s)) + qb10 + qb20

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
                boomeffect += self.areast * (self.stcoord[j, 0] - self.zc)
            boom4list.append(self.areast * (self.stcoord[i, 0] - self.zc))
            boomeffect4list.append(boomeffect)

        boomf = sp.interpolate.interp1d(s4list,ilist,kind="previous",
                                        fill_value="extrapolate")
        boomeffect4f = sp.interpolate.interp1d(s4list,boomeffect4list,
                                               kind="previous",
                                               fill_value="extrapolate")

        def qb4f(s):
            return -k*(self.tsk*((-self.Ca-self.zc)*s+(self.Ca-h)/self.lsk/2.*s**2) + boomeffect4f(s)) + qb30

        ### Base shear flow in area (5)
        def qb5f(y):
            return-k*self.tsp*(-h-self.zc)*y

        ### Base sehar flow in area (6)
        qb40 = qb4f(self.lsk)
        qb50 = qb5f(-h)

        phi6list = [-m.pi/2]
        ilist = [0]
        boom6list = [0]
        boomeffect6list = [0]
        for i in range(self.nst-int(self.ncirc // 2),self.nst):
            phi6list.append(self.phi*(i-self.nst))
            ilist.append(i-self.nst-int(self.ncirc // 2))
            boomeffect = 0
            for j in range(self.nst-int(self.ncirc // 2),i+1):
                boomeffect += self.areast * (self.stcoord[j,0] - self.zc)
            boom6list.append(self.areast*(self.stcoord[j,0] - self.zc))
            boomeffect6list.append(boomeffect)

        boomf = sp.interpolate.interp1d(phi6list,ilist,kind="previous",
                                        fill_value="extrapolate")
        boomeffect6f = sp.interpolate.interp1d(phi6list,boomeffect6list,
                                               kind="previous",
                                               fill_value="extrapolate")

        def qb6f(theta):
            return -k*(self.tsk * h*(-h*(theta+m.pi/2)+h*(np.sin(theta)+1) - \
                    self.zc*(theta+m.pi/2)) + boomeffect6f(theta)) + qb40 - qb50
        self.Szq1f = qb1f
        self.Szq2f = qb2f
        self.Szq3f = qb3f
        self.Szq4f = qb4f
        self.Szq5f = qb5f
        self.Szq6f = qb6f

    def compute_unitshearflowdistributionSy(self):
        ### Shear flow distribution due to Sy
        k = 1/self.Izz
        h = self.ha/2

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

        boomeffect1f = sp.interpolate.interp1d(phi1list,boomeffect1list,
                                               kind="previous",
                                               fill_value="extrapolate")

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

        boomeffect6f = sp.interpolate.interp1d(phi6list,boomeffect6list,
                                               kind="previous",
                                               fill_value="extrapolate")

        def qb6f(theta):
            return k*(self.tsk * h**2*(np.cos(theta)) - boomeffect6f(theta)) + \
                   qb40 - qb50

        """Redundant shear flow"""
        ## Influence of region (1)
        def qb1intf(theta):
            initial = k*(self.tsk*(h**3*(np.sin(theta)-theta)))
            for i,phi in enumerate(phi1list):
                if theta >= phi:
                    initial -= k*(h*boom1list[i]*(theta-phi))
            return initial

        def qb2intf(y):
            return -k*self.tsp/6.*y**3

        def qb3intf(s):
            initial = -k*(self.tsk*(h/2.*s**2-h/self.lsk/6.*s**3)) + \
                       (qb10 + qb20)*s
            for i,s0 in enumerate(s3list):
                if s > s0:
                    initial -= k*(boom3list[i]*(s-s0))
            return initial

        def qb4intf(s):
            initial = -k*(self.tsk*(-h/self.lsk/6.*s**3)) + (qb30)*s
            for i,s0 in enumerate(s4list):
                if s > s0:
                    initial -= k*(boom4list[i]*(s-s0))
            return initial

        def qb5intf(y):
            return -k*self.tsp/6.*y**3

        def qb6intf(theta):
            initial = k*(self.tsk*(h**3*(np.sin(theta)+1))) + \
                      (qb40 - qb50)*(theta+m.pi/2)*h
            for i,phi in enumerate(phi6list):
                if theta >= phi:
                    initial -= k*(h*boom6list[i]*(theta-phi))
            return initial

        if self.tsp > 0:
            b = np.array([0.,0.])
            b[0] = qb1intf(m.pi/2)/self.tsk - qb2intf(h)/self.tsp - \
                   qb5intf(h)/self.tsp + qb6intf(0)/self.tsk
            b[1] = qb2intf(h)/self.tsp + qb3intf(self.lsk)/self.tsk + \
                   qb4intf(self.lsk)/self.tsk + qb5intf(h)/self.tsp

            A = np.array([[0.,0.],[0.,0.]])
            A[0,0] = h*m.pi/self.tsk + 2*h/self.tsp
            A[0,1] = -2*h/self.tsp
            A[1,0] = -2*h/self.tsp
            A[1,1] = 2*self.lsk/self.tsk + 2*h/self.tsp

            qredundant = np.linalg.solve(A,-b)

        else:
            b = qb1intf(m.pi/2)/self.tsk + qb3intf(self.lsk)/self.tsk + \
                qb4intf(self.lsk)/self.tsk + qb6intf(0)/self.tsk
            A = h*m.pi/self.tsk + 2*self.lsk/self.tsk

            qredundant = -b/A * np.ones(2)

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

        self.Syq1f = q1f
        self.Syq2f = q2f
        self.Syq3f = q3f
        self.Syq4f = q4f
        self.Syq5f = q5f
        self.Syq6f = q6f

    def compute_unitshearflowdistributionT(self):
        h = self.ha/2.
        A1 = m.pi*h**2/2.
        A2 = (self.Ca-h)*h

        A = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        b = np.array([0.,0.,0.])

        ### First row
        A[0,0] = 2.*A1
        A[0,1] = 2.*A2
        b[0] = -1

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
        print("q01_t:",solution[0])
        print("q02_t:",solution[1])
        print("q02_t:",solution[2])
        self.Tq1f = solution[0]
        self.Tq2f = -solution[0] + solution[1]
        self.Tq3f = solution[1]
        self.Tq4f = solution[1]
        self.Tq5f = -solution[0] + solution[1]
        self.Tq6f = solution[0]

    def compute_unitstressdistributionMy(self):
        def sigma1f(theta):
            z = (-self.ha/2.*(1-np.cos(theta)) - self.zc)
            return z/self.Iyy
        self.sigma1Myf = sigma1f
        def sigma2f(y):
            z = (-self.ha/2. - self.zc * np.ones(np.size(y)))
            return z/self.Iyy
        self.sigma2Myf = sigma2f
        def sigma3f(s):
            z = (-self.ha/2. - self.zc - (self.Ca-self.ha/2.)/self.lsk * s)
            return z/self.Iyy
        self.sigma3Myf = sigma3f
        def sigma4f(s):
            z = (-self.Ca - self.zc + (self.Ca-self.ha/2.)/self.lsk * s)
            return z/self.Iyy
        self.sigma4Myf = sigma4f
        def sigma5f(y):
            z = (-self.ha/2. - self.zc * np.ones(np.size(y)))
            return z/self.Iyy
        self.sigma5Myf = sigma5f
        def sigma6f(theta):
            z = (-self.ha/2.*(1-np.cos(theta)) - self.zc)
            return z/self.Iyy
        self.sigma6Myf = sigma6f

    def compute_unitstressdistributionMz(self):
        def sigma1f(theta):
            y = self.ha/2.*np.sin(theta)
            return y/self.Izz
        self.sigma1Mzf = sigma1f
        def sigma2f(y):
            return y/self.Izz
        self.sigma2Mzf = sigma2f
        def sigma3f(s):
            y = self.ha/2. + -self.ha/2./self.lsk * s
            return y/self.Izz
        self.sigma3Mzf = sigma3f
        def sigma4f(s):
            y = -self.ha/2./self.lsk * s
            return y/self.Izz
        self.sigma4Mzf = sigma4f
        def sigma5f(y):
            return y/self.Izz
        self.sigma5Mzf = sigma5f
        def sigma6f(theta):
            y = self.ha/2.*np.sin(theta)
            return y/self.Izz
        self.sigma6Mzf = sigma6f

    def compute_stressdistributions(self,Sy,Sz,My,Mz,T):
        def q1f(theta):
            
            #print("Szq1f /cdot Sz:", self.Szq1f(theta)*Sz)
            #print("1:  ", self.Syq1f(theta)*Sy + self.Szq1f(theta)*Sz + self.Tq1f*T)
            return  self.Tq1f*T +self.Syq1f(theta)*Sy +self.Szq1f(theta)*Sz  
        self.q1f = q1f
        def q2f(y):
            #print("Szq2f /cdot Sz:", self.Szq2f(y)*Sz)
            #print("2:   ", self.Syq2f(y)*Sy+ self.Szq2f(y)*Sz + self.Tq2f*T)
            return     self.Tq2f*T +self.Syq2f(y)*Sy +self.Szq2f(y)*Sz
        self.q2f = q2f
        def q3f(s):
            #print("Szq3f /cdot Sz:", self.Szq3f(s)*Sz)
            #print("3:   ", self.Syq3f(s)*Sy+ self.Szq3f(s)*Sz + self.Tq3f*T)
            return   self.Tq3f*T +self.Syq3f(s)*Sy +self.Szq3f(s)*Sz
        self.q3f = q3f
        def q4f(s):
            #print("Szq4f /cdot Sz:", self.Szq4f(s)*Sz)
            #print("4:   ", self.Syq4f(s)*Sy+ self.Szq4f(s)*Sz + self.Tq4f*T)
            return   self.Tq4f*T +self.Syq4f(s)*Sy + self.Szq4f(s)*Sz
        self.q4f = q4f
        def q5f(y):
            #print("Szq5f /cdot Sz:", self.Szq5f(y)*Sz)
            #print("5:  ", self.Syq5f(y)*Sy+ self.Szq5f(y)*Sz + self.Tq5f*T)
            return   self.Tq5f*T + self.Syq5f(y)*Sy +self.Szq5f(y)*Sz
        self.q5f = q5f
        def q6f(theta):
            #print("Szq6f /cdot Sz:", self.Szq6f(theta)*Sz)
            #print("6:  ", self.Syq6f(theta)*Sy+ self.Szq6f(theta)*Sz + self.Tq6f*T)
            return  self.Tq6f*T +self.Syq6f(theta)*Sy + self.Szq6f(theta)*Sz 
        self.q6f = q6f

        def sigma1f(theta):
            return self.sigma1Myf(theta)*My + self.sigma1Mzf(theta)*Mz
        self.sigma1f = sigma1f
        def sigma2f(y):
            return self.sigma2Myf(y)*My + self.sigma2Mzf(y)*Mz
        self.sigma2f = sigma2f
        def sigma3f(s):
            return self.sigma3Myf(s)*My + self.sigma3Mzf(s)*Mz
        self.sigma3f = sigma3f
        def sigma4f(s):
            return self.sigma4Myf(s)*My + self.sigma4Mzf(s)*Mz
        self.sigma4f = sigma4f
        def sigma5f(y):
            return self.sigma5Myf(y)*My + self.sigma5Mzf(y)*Mz
        self.sigma5f = sigma5f
        def sigma6f(theta):
            return self.sigma6Myf(theta)*My + self.sigma6Mzf(theta)*Mz
        self.sigma6f = sigma6f

        def vm1(theta):
            return np.sqrt(self.sigma1f(theta)**2 + \
                           3*(self.q1f(theta)/self.tsk)**2)
        self.vm1 = vm1
        def vm2(y):
            return np.sqrt(self.sigma2f(y)**2+3*(self.q2f(y)/self.tsp)**2)
        self.vm2 = vm2
        def vm3(s):
            return np.sqrt(self.sigma3f(s)**2+3*(self.q3f(s)/self.tsk)**2)
        self.vm3 = vm3
        def vm4(s):
            return np.sqrt(self.sigma4f(s)**2+3*(self.q4f(s)/self.tsk)**2)
        self.vm4 = vm4
        def vm5(y):
            return np.sqrt(self.sigma5f(y)**2+3*(self.q5f(y)/self.tsp)**2)
        self.vm5 = vm5
        def vm6(theta):
            return np.sqrt(self.sigma6f(theta)**2 + \
                           3*(self.q6f(theta)/self.tsk)**2)
        self.vm6 = vm6

    def plot_shearflowdistributions(self):
        ### Plot region 1
        theta1 = np.linspace(0,m.pi/2,num=1000)
        x1 = -(1-np.cos(theta1))*self.ha/2.
        y1 = np.sin(theta1)*self.ha/2.
        z1 = self.q1f(theta1)
        path = mpath.Path(np.column_stack([x1, y1]))
        verts = path.interpolated(steps=1).vertices
        x1, y1 = verts[:, 0,], verts[:, 1]
        maxabs = np.max(np.abs(z1))

        ### Plot region 2
        y2 = np.linspace(0,self.ha/2.,num=1000)
        x2 = -self.ha/2.*np.ones(1000)
        z2 = self.q2f(y2)
        path = mpath.Path(np.column_stack([x2, y2]))
        verts = path.interpolated(steps=1).vertices
        x2, y2 = verts[:, 0,], verts[:, 1]
        maxabs2 = np.max(np.abs(z2))
        maxabs = max(maxabs2,maxabs)

        ### Plot region 3
        s3 = np.linspace(0,self.lsk,num=1000)
        x3 = -self.ha/2. - (self.Ca-self.ha/2)/self.lsk*s3
        y3 = self.ha/2. - (self.ha/2)/self.lsk*s3
        z3 = self.q3f(s3)
        path = mpath.Path(np.column_stack([x3, y3]))
        verts = path.interpolated(steps=1).vertices
        x3, y3 = verts[:, 0,], verts[:, 1]
        maxabs3 = np.max(np.abs(z3))
        maxabs = max(maxabs3,maxabs)

        ### Plot region 4
        s4 = np.linspace(0,self.lsk,num=1000)
        x4 = -self.Ca + (self.Ca-self.ha/2)/self.lsk*s4
        y4 = - (self.ha/2)/self.lsk*s4
        z4 = self.q4f(s4)
        path = mpath.Path(np.column_stack([x4, y4]))
        verts = path.interpolated(steps=1).vertices
        x4, y4 = verts[:, 0,], verts[:, 1]
        maxabs4 = np.max(np.abs(z4))
        maxabs = max(maxabs4,maxabs)

        ### Plot region 5
        y5 = np.linspace(0,-self.ha/2.,num=1000)
        x5 = -self.ha/2.*np.ones(1000)
        z5 = self.q5f(y5)
        path = mpath.Path(np.column_stack([x5, y5]))
        verts = path.interpolated(steps=1).vertices
        x5, y5 = verts[:, 0,], verts[:, 1]
        maxabs5 = np.max(np.abs(z5))
        maxabs = max(maxabs5,maxabs)

        ### Plot region 6
        theta6 = np.linspace(0,-m.pi/2,num=1000)
        x6 = -(1-np.cos(theta6))*self.ha/2.
        y6 = np.sin(theta6)*self.ha/2.
        z6 = self.q6f(theta6)
        path = mpath.Path(np.column_stack([x6, y6]))
        verts = path.interpolated(steps=1).vertices
        x6, y6 = verts[:, 0,], verts[:, 1]
        maxabs6 = np.max(np.abs(z6))
        maxabs = max(maxabs6,maxabs)
        fig = plt.figure(4)

        colorline(x1, y1, z1, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x2, y2, z2, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x3, y3, z3, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x4, y4, z4, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x5, y5, z5, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x6, y6, z6, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'), 
                                   norm=plt.Normalize(-maxabs, maxabs))
        sm.set_array([])
        plt.colorbar(sm,label=r'$q$ [N/m]',fraction = 0.20, pad = 0.04, 
                     orientation = "horizontal")
        plt.xlim(-self.Ca-0.1, 0.1)
        plt.ylim(-self.ha/2-0.02, self.ha/2+0.02)
        plt.axis('scaled')
        plt.gca().invert_xaxis()
        plt.xlabel(r'$-z$ [m]')
        plt.ylabel(r'$y$ [m]')
        plt.title('Shear flow distribution')
        plt.show()

    def plot_directstressdistributions(self):
        ### Plot region 1
        theta1 = np.linspace(0,m.pi/2,num=1000)
        x1 = -(1-np.cos(theta1))*self.ha/2.
        y1 = np.sin(theta1)*self.ha/2.
        z1 = self.sigma1f(theta1)
        path = mpath.Path(np.column_stack([x1, y1]))
        verts = path.interpolated(steps=1).vertices
        x1, y1 = verts[:, 0,], verts[:, 1]
        maxabs = np.max(np.abs(z1))

        ### Plot region 2
        y2 = np.linspace(0,self.ha/2.,num=1000)
        x2 = -self.ha/2.*np.ones(1000)
        z2 = self.sigma2f(y2)
        path = mpath.Path(np.column_stack([x2, y2]))
        verts = path.interpolated(steps=1).vertices
        x2, y2 = verts[:, 0,], verts[:, 1]
        maxabs2 = np.max(np.abs(z2))
        maxabs = max(maxabs2,maxabs)

        ### Plot region 3
        s3 = np.linspace(0,self.lsk,num=1000)
        x3 = -self.ha/2. - (self.Ca-self.ha/2)/self.lsk*s3
        y3 = self.ha/2. - (self.ha/2)/self.lsk*s3
        z3 = self.sigma3f(s3)
        path = mpath.Path(np.column_stack([x3, y3]))
        verts = path.interpolated(steps=1).vertices
        x3, y3 = verts[:, 0,], verts[:, 1]
        maxabs3 = np.max(np.abs(z3))
        maxabs = max(maxabs3,maxabs)

        ### Plot region 4
        s4 = np.linspace(0,self.lsk,num=1000)
        x4 = -self.Ca + (self.Ca-self.ha/2)/self.lsk*s4
        y4 = - (self.ha/2)/self.lsk*s4
        z4 = self.sigma4f(s4)
        path = mpath.Path(np.column_stack([x4, y4]))
        verts = path.interpolated(steps=1).vertices
        x4, y4 = verts[:, 0,], verts[:, 1]
        maxabs4 = np.max(np.abs(z4))
        maxabs = max(maxabs4,maxabs)

        ### Plot region 5
        y5 = np.linspace(0,-self.ha/2.,num=1000)
        x5 = -self.ha/2.*np.ones(1000)
        z5 = self.sigma5f(y5)
        path = mpath.Path(np.column_stack([x5, y5]))
        verts = path.interpolated(steps=1).vertices
        x5, y5 = verts[:, 0,], verts[:, 1]
        maxabs5 = np.max(np.abs(z5))
        maxabs = max(maxabs5,maxabs)

        ### Plot region 6
        theta6 = np.linspace(0,-m.pi/2,num=1000)
        x6 = -(1-np.cos(theta6))*self.ha/2.
        y6 = np.sin(theta6)*self.ha/2.
        z6 = self.sigma6f(theta6)
        path = mpath.Path(np.column_stack([x6, y6]))
        verts = path.interpolated(steps=1).vertices
        x6, y6 = verts[:, 0,], verts[:, 1]
        maxabs6 = np.max(np.abs(z6))
        maxabs = max(maxabs6,maxabs)
        fig = plt.figure(5)

        colorline(x1, y1, z1, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x2, y2, z2, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x3, y3, z3, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x4, y4, z4, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x5, y5, z5, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(x6, y6, z6, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'), 
                                   norm=plt.Normalize(-maxabs, maxabs))
        sm.set_array([])
        plt.colorbar(sm,label=r'$\sigma_{xx}$ N/m$^2$',fraction = 0.20, 
                     pad = 0.04, orientation = "horizontal")
        plt.xlim(-self.Ca-0.1, 0.1)
        plt.ylim(-self.ha/2-0.02, self.ha/2+0.02)
        plt.gca().invert_xaxis()
        plt.axis('scaled')
        plt.xlabel(r'$z$ [m]')
        plt.ylabel(r'$y$ [m]')
        plt.title('Direct stress distribution')
        plt.show()

    def plot_vonmisesstressdistributions(self):
        ### Plot region 1
        theta1 = np.linspace(0,m.pi/2,num=1000)
        x1 = -(1-np.cos(theta1))*self.ha/2.
        y1 = np.sin(theta1)*self.ha/2.
        z1 = self.vm1(theta1)
        path = mpath.Path(np.column_stack([x1, y1]))
        verts = path.interpolated(steps=1).vertices
        x1, y1 = verts[:, 0,], verts[:, 1]
        maxabs = np.max(np.abs(z1))

        ### Plot region 2
        y2 = np.linspace(0,self.ha/2.,num=1000)
        x2 = -self.ha/2.*np.ones(1000)
        z2 = self.vm2(y2)
        path = mpath.Path(np.column_stack([x2, y2]))
        verts = path.interpolated(steps=1).vertices
        x2, y2 = verts[:, 0,], verts[:, 1]
        maxabs2 = np.max(np.abs(z2))
        maxabs = max(maxabs2,maxabs)

        ### Plot region 3
        s3 = np.linspace(0,self.lsk,num=1000)
        x3 = -self.ha/2. - (self.Ca-self.ha/2)/self.lsk*s3
        y3 = self.ha/2. - (self.ha/2)/self.lsk*s3
        z3 = self.vm3(s3)
        path = mpath.Path(np.column_stack([x3, y3]))
        verts = path.interpolated(steps=1).vertices
        x3, y3 = verts[:, 0,], verts[:, 1]
        maxabs3 = np.max(np.abs(z3))
        maxabs = max(maxabs3,maxabs)

        ### Plot region 4
        s4 = np.linspace(0,self.lsk,num=1000)
        x4 = -self.Ca + (self.Ca-self.ha/2)/self.lsk*s4
        y4 = - (self.ha/2)/self.lsk*s4
        z4 = self.vm4(s4)
        path = mpath.Path(np.column_stack([x4, y4]))
        verts = path.interpolated(steps=1).vertices
        x4, y4 = verts[:, 0,], verts[:, 1]
        maxabs4 = np.max(np.abs(z4))
        maxabs = max(maxabs4,maxabs)

        ### Plot region 5
        y5 = np.linspace(0,-self.ha/2.,num=1000)
        x5 = -self.ha/2.*np.ones(1000)
        z5 = self.vm5(y5)
        path = mpath.Path(np.column_stack([x5, y5]))
        verts = path.interpolated(steps=1).vertices
        x5, y5 = verts[:, 0,], verts[:, 1]
        maxabs5 = np.max(np.abs(z5))
        maxabs = max(maxabs5,maxabs)

        ### Plot region 6
        theta6 = np.linspace(0,-m.pi/2,num=1000)
        x6 = -(1-np.cos(theta6))*self.ha/2.
        y6 = np.sin(theta6)*self.ha/2.
        z6 = self.vm6(theta6)
        path = mpath.Path(np.column_stack([x6, y6]))
        verts = path.interpolated(steps=1).vertices
        x6, y6 = verts[:, 0,], verts[:, 1]
        maxabs6 = np.max(np.abs(z6))
        maxabs = max(maxabs6,maxabs)
        fig = plt.figure(6)

        colorline(x1, y1, z1, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(x2, y2, z2, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(x3, y3, z3, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(x4, y4, z4, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(x5, y5, z5, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(x6, y6, z6, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-0, maxabs), linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'), 
                                   norm=plt.Normalize(0, maxabs))
        sm.set_array([])
        plt.colorbar(sm,label=r'$\sigma_{vm}$ N/m$^2$',fraction = 0.20, 
                     pad = 0.04, orientation = "horizontal")
        plt.xlim(-self.Ca-0.1, 0.1)
        plt.ylim(-self.ha/2-0.02, self.ha/2+0.02)
        plt.axis('scaled')
        plt.gca().invert_xaxis()
        plt.xlabel(r'$z$ [m]')
        plt.ylabel(r'$y$ [m]')
        plt.title('Von Mises stress distribution')

        plt.show()

    def coord1(self,theta):
        return -self.ha/2*(1-np.cos(theta)), self.ha/2.*np.sin(theta)

    def coord2(self,y):
        return -self.ha/2.*np.ones(np.size(y)), y

    def coord3(self,s):
        return -self.ha/2. - (self.Ca-self.ha/2.)/self.lsk*s, \
                self.ha/2. - self.ha/2./self.lsk*s

    def coord4(self,s):
        return -self.Ca + (self.Ca-self.ha/2.)/self.lsk*s, \
               -self.ha/2./self.lsk*s

    def coord5(self,y):
        return -self.ha/2.*np.ones(np.size(y)), y

    def coord6(self,theta):
        return -self.ha/2*(1-np.cos(theta)), self.ha/2.*np.sin(theta)


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), 
              norm=plt.Normalize(-1.0, 1.0), linewidth=3, alpha=1.0):
    """
    Source:
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    if not hasattr(z, "__iter__"):  
        z = np.array([z])

    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Source:
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments