
""" 
 This is file is part of the developer code of the AE3212-II project. 
 The class defined in this file 

 Note that comments are largely missing from this code. However, all 
 calculations follow from what is described in the manual accompanying the 
 developer model. Some matrices have been slightly renamed compared to the 
 manual.
 
 Furthermore, due to some of the lengthy expressions in this code, the 
 80-character-per-line limit has been ignored. 

 Author: Sam van Elsloo
 Date:   06/02/2021
"""


import math as m
import numpy as np
import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt
import sys

def monosdint(mono1,mono2):
    i = mono1.n
    j = mono2.n
    if i < 2 or j < 2:
        return 0
    return 1.0*i*(i-1)*j*(j-1)/(i+j-3)*(1+(-1)**(i+j))

def monofdint(mono1,mono2):
    i = mono1.n
    j = mono2.n
    if i < 1 or j < 1:
        return 0
    return 1.0*i*j/(i+j-1)*(1+(-1)**(i+j))

class Mono:
    def __init__(self,n):
        self.n = n
    def eval(self,x):
        return np.power(x,self.n)
    def fd(self,x):
        return self.n*np.power(x,max(0,self.n-1))
    def sd(self,x):
        return self.n*(self.n-1)*np.power(x,max(0,self.n-2))
    def td(self,x):
        return self.n*(self.n-1)*(self.n-2)*np.power(x,max(0,self.n-3))
    def nd(self,x,o):
        return m.factorial(self.n)/m.factorial(self.n-o)*np.power(x,max(0,o))

class Solution:
    def __init__(self,n):
        self.n = n
        self.coef = np.zeros(n)
        self.unitcoef = np.ones(n)
        self.basis = np.empty(n,dtype=object)
        vmono = np.vectorize(Mono)
        irange = np.linspace(0,n-1,n)
        self.basis = vmono(irange)
        self.basis = list(self.basis)*3
    def evalbasis(self,x):
        temp = np.empty((3*self.n,np.size(x)))
        for i in range(3*self.n):
            temp[i,:] = self.basis[i].eval(x)
        return temp
    def fdbasis(self,x):
        temp = np.empty((3*self.n,np.size(x)))
        for i in range(3*self.n):
            temp[i,:] = self.basis[i].fd(x)
        return temp
    def sdbasis(self,x):
        temp = np.empty((2*self.n,np.size(x)))
        for i in range(2*self.n):
            temp[i,:] = self.basis[i].sd(x)
        return temp
    def tdbasis(self,x):
        temp = np.empty((2*self.n,np.size(x)))
        for i in range(2*self.n):
            temp[i,:] = self.basis[i].td(x)
        return temp

    ### Transformed domains
    def eval(self,x):
        return np.dot(self.coef[:self.n],self.evalbasis(x)[:self.n]),np.dot(self.coef[self.n:2*self.n],self.evalbasis(x)[self.n:2*self.n]),np.dot(self.coef[-self.n:],self.evalbasis(x)[-self.n:])
    def fdeval(self,x):
        return np.dot(self.coef[:self.n],self.fdbasis(x)[:self.n]),np.dot(self.coef[self.n:2*self.n],self.fdbasis(x)[self.n:2*self.n]),np.dot(self.coef[-self.n:],self.fdbasis(x)[-self.n:])
    def sdeval(self,x):
        return np.dot(self.coef[:self.n],self.sdbasis(x)[:self.n]),np.dot(self.coef[self.n:2*self.n],self.sdbasis(x)[self.n:2*self.n]),np.dot(self.coef[-self.n:],self.sdbasis(x)[-self.n:])
    def tdeval(self,x):
        return np.dot(self.coef[:self.n],self.tdbasis(x)[:self.n]),np.dot(self.coef[self.n:2*self.n],self.tdbasis(x)[self.n:2*self.n]),np.dot(self.coef[-self.n:],self.tdbasis(x)[-self.n:])
    def uniteval(self,x):
        return np.dot(self.unitcoef,self.evalbasis(x))

class Beam:
    def __init__(self, la, crosssection, N, E, G):
        self.L = la
        self.C = crosssection.Ca
        self.H = crosssection.ha
        self.ysc = crosssection.ysc
        self.zsc = crosssection.zsc
        self.EIyy = crosssection.Iyy*E
        self.EIzz = crosssection.Izz*E
        self.GJ = crosssection.J*G
        self.bc = []
        self.nbc = 0
        self.f = []
        self.n = N
        self.sol = Solution(N)

    def addbcss(self, x,y,z,theta,f):
        bcond = ["ss",x,y,z,theta,f]
        self.bc.append(bcond)

    def addbcfo(self,x,theta,f):
        bcond = ["c",x,theta,f]
        self.bc.append(bcond)

    def addbctw(self,x,f):
        bcond = ["t",x,f]
        self.bc.append(bcond)

    def addfdistt(self,x1,x2,f):
        ### Input parameters:
        # x1: starting position of distributed torque
        # x2: ending position of distributed torque
        # f: function describing distributed torque
        force = ["distt",x1,x2,f]
        self.f.append(force)

    def addfdirectt(self,x,T):
        ### Input parameters:
        # x: location of direct torque
        # T: magnitude of direct torque
        force = ["directt",x,T]
        self.f.append(force)

    def addfddxz(self,x1,x2,z1,z2,f):
        ### Input parameters:
        # x1: starting position of distributed load in x-direction
        # x2: ending position of distributed load in x-direction
        # z1: starting position of distributed load in z-direction
        # z2: ending position of distributed load in z-direction
        force = ["ddxz",x1,x2,z1,z2,f]
        self.f.append(force)

    def addfddxy(self,x1,x2,y1,y2,f):
        ### Input parameters:
        # x1: starting position of distributed load in x-direction
        # x2: ending position of distributed load in x-direction
        # y1: starting position of distributed load in z-direction
        # y2: ending position of distributed load in z-direction
        force = ["ddxz",x1,x2,y1,y2,f]
        self.f.append(force)

    def addfdl(self,x1,x2,yf,zf,thetaf,f):
        ### Input parameters:
        # x1: starting position of distributed load in x-direction
        # x2: ending position of distributed load in x-direction
        # yf: function describing the y-coordinate of the point of application
        # zf: function describing the z-coordinate of the point of application
        # thetaf: function describing the orientation of the distributed load
        # f: function describing distributed load
        force = ["dl",x1,x2,yf,zf,thetaf,f]
        self.f.append(force)

    def addfpl(self,x,y,z,theta,P):
        ### Input parameters:
        # x: x-coordinate point of application
        # y: y-coordinate point of application
        # z: z-coordinate point of application
        # theta: orientation of point load
        # P: magnitude of point load
        force = ["pl",x,y,z,theta,P]
        self.f.append(force)

    def addfcm(self,x,theta,M):
        ### Input parameters:
        # x: x-coordinate point of application
        # theta: orientation of couple moment
        # M: magnitude of couple moment
        force = ["cm",x,theta,M]
        self.f.append(force)

    def cUm(self):
        self.nbc = len(self.bc)
        self.bcmatrixfull = np.zeros((self.nbc,3*self.n))
        self.bcRHS = np.zeros(self.nbc)
        for i,bc in enumerate(self.bc):
            if bc[0] == "ss":
                self.bcmatrixfull[i,:] = self.sol.evalbasis(2*bc[1]/self.L-1).flatten()
                self.bcmatrixfull[i,:self.n] *= m.cos(bc[4])
                self.bcmatrixfull[i,self.n:2*self.n] *= m.sin(bc[4])
                self.bcmatrixfull[i,2*self.n:] *= (-(bc[3]-self.zsc)*m.cos(bc[4])+(bc[2]-self.ysc)*m.sin(bc[4]))
                self.bcRHS[i] = bc[5]
            elif bc[0] == "c":
                self.bcmatrixfull[i, :2*self.n] = self.sol.fdbasis(2*bc[1]/self.L-1)[:2*self.n].flatten()
                self.bcmatrixfull[i,:self.n] *= m.sin(bc[2])
                self.bcmatrixfull[i,self.n:2*self.n] *= -m.cos(bc[2])
                self.bcRHS[i] = bc[3]*self.L/2
            elif bc[0] == "t":
                self.bcmatrixfull[i,2*self.n:] = self.sol.evalbasis(2*bc[1]/self.L-1)[2*self.n:].flatten()
                self.bcRHS[i] = bc[2]
        i = 1
        j = 2
        succes = False

        """
         The code below is used to determine which coefficients should be part 
         of hat(a) and which should be part of bar(a) (essentially being used to
         satisfy the boundary conditions). Choosing which coefficients go where
         is less straightforward than it initially seems - the 'lowest' 
         coefficients should go into bar(a) generally speaking, but in case of
         three deflection curves, it is not straightforward to pick a
         combination of coefficients that guarantees a system of full rank, and
         there are generally multiple combinations possible.

         Therefore, this code tries out a few combinations that meet the basic
         requirements for which coefficients should go into bar(a): at least the
         first two coefficients for the bending curves, and one coefficient for 
         the twist curve, and the total number of coefficients in bar(a) equals
         self.nbc. For each combination it is tried whether the resulting 
         Upsilon matrix can be constructed (requiring that Upsilon135 is 
         invertible). If an error is thrown by Python (indicating that there is
         no solution due to a singular matrix), a new combination is tried, 
         until no new combination can be tried. If there is no combination of
         coefficients that go into bar(a) possible, then the problem is either
         over- or under-constrained. 

         Note that sometimes Python will sometimes not throw an error because
         the Upsilon135 matrix is marginally invertible. However, in that case,
         the values in the Upsilon matrix are very large, so that is another 
         sign that the current combination is not valid.
        """

        while succes == False and i < self.nbc:
            try:
                self.nbct = i
                self.nbcv = j
                self.nbcw = self.nbc-i-j
                self.U1temp = self.bcmatrixfull[:,:self.nbcv]
                self.U2temp = self.bcmatrixfull[:,self.nbcv:self.n]
                self.U3temp = self.bcmatrixfull[:,self.n:self.n+self.nbcw]
                self.U4temp = self.bcmatrixfull[:,self.n+self.nbcw:2*self.n]
                self.U5temp = self.bcmatrixfull[:,2*self.n:2*self.n+self.nbct]
                self.U6temp = self.bcmatrixfull[:,2*self.n+self.nbct:]
                self.U135temp = np.concatenate((self.U1temp,self.U3temp,self.U5temp),axis = 1)
                self.U246temp = np.concatenate((self.U2temp,self.U4temp,self.U6temp),axis = 1)
                self.U = np.linalg.solve(self.U135temp,-self.U246temp)
            except np.linalg.LinAlgError:
                if i == self.nbc - 3:
                    print("Boundary conditions form an over- or underdetermined set of equations; program is aborted")
                    sys.exit(0)
                else:
                    if j < self.nbc - i - 2:
                        j += 1
                    else:
                        i += 1
                        j = 2
            else:
                if np.linalg.norm(self.U) < 1000000000:
                    succes = True
                else:
                    if j < self.nbc - i - 2:
                        j += 1
                    else:
                        i += 1
                        j = 2
        if succes == False:
            print("Boundary conditions form an over- or underdetermined set of equations; program is aborted")
            sys.exit(0)
        self.Ua = self.U[:self.nbcv,:]
        self.Ub = self.U[self.nbcv:self.nbcv+self.nbcw,:]
        self.Uc = self.U[-self.nbct:,:]
        self.Ut = self.U.transpose()
        self.Uat = self.Ua.transpose()
        self.Ubt = self.Ub.transpose()
        self.Uct = self.Uc.transpose()
        self.F = np.linalg.solve(self.U135temp,self.bcRHS)
        self.Na = self.n - self.nbcv
        self.Nb = self.n - self.nbcw
        self.Nc = self.n - self.nbct

    def cHam(self):
        self.Ha = np.zeros((self.Na,3*self.n-self.nbc))
        self.Ha[:,:self.Na] = np.eye(self.Na)
        self.Hat = self.Ha.transpose()

    def cHbm(self):
        self.Hb = np.zeros((self.Nb,3*self.n-self.nbc))
        self.Hb[:,self.Na:self.Na+self.Nb] = np.eye(self.Nb)
        self.Hbt = self.Hb.transpose()

    def cHcm(self):
        self.Hc = np.zeros((self.Nc,3*self.n-self.nbc))
        self.Hc[:,self.Na+self.Nb:] = np.eye(self.Nc)
        self.Hct = self.Hc.transpose()

    def cK1m(self):
        ### Create K1
        self.K1 = np.zeros((self.n,self.n))
        irange = np.arange(2,self.n)
        krange = np.array([1,-1] * self.n)[:self.n-2]
        temp1 = np.einsum('i,j->ij',irange*(irange-1),irange*(irange-1))
        temp2 = irange[:,np.newaxis]+irange[np.newaxis, :]-3
        temp3 = np.abs(krange[:,np.newaxis]+krange[np.newaxis,:])
        self.K1[2:,2:] = temp1 / temp2 * temp3
        self.K1a11 = np.copy(self.K1[:self.nbcv,:self.nbcv])
        self.K1a12 = np.copy(self.K1[:self.nbcv,self.nbcv:])
        self.K1a21 = np.copy(self.K1[self.nbcv:,:self.nbcv])
        self.K1a22 = np.copy(self.K1[self.nbcv:,self.nbcv:])
        self.K1b11 = np.copy(self.K1[:self.nbcw,:self.nbcw])
        self.K1b12 = np.copy(self.K1[:self.nbcw,self.nbcw:])
        self.K1b21 = np.copy(self.K1[self.nbcw:,:self.nbcw])
        self.K1b22 = np.copy(self.K1[self.nbcw:,self.nbcw:])

    def cC1m(self):
        ### Create C1
        self.C1 = np.zeros((self.n,self.n))
        irange = np.arange(1,self.n)
        krange = np.array([1,-1]*self.n)[:self.n-1]
        temp1 = np.einsum('i,j->ij', irange, irange)
        temp2 = irange[:, np.newaxis] + irange[np.newaxis, :] - 1
        temp3 = np.abs(krange[:, np.newaxis] + krange[np.newaxis, :])
        self.C1[1:,1:] = temp1 / temp2 * temp3
        self.C111 = self.C1[:self.nbct, :self.nbct]
        self.C112 = self.C1[:self.nbct, self.nbct:]
        self.C121 = self.C1[self.nbct:, :self.nbct]
        self.C122 = self.C1[self.nbct:, self.nbct:]

    def cK2m(self):
        ### Create K2
        self.K2a = np.zeros((self.n))
        self.K2b = np.zeros((self.n))
        for i,force in enumerate(self.f):
            if force[0] == "distt":
                pass
            elif force[0] == "directt":
                pass
            elif force[0] == "ddxz":
                print("here")
                def g(x,z):
                    return force[5]((x+1)*self.L/2,z)
                for j in range(self.n):
                    self.K2a[j] += sp.integrate.dblquad(lambda x,z: g(x,z)*self.sol.basis[j].eval(x),force[3],force[4],2*force[1]/self.L-1,2*force[2]/self.L-1)[0]
                    print("Currently at " + str(j/self.n*100) + "% of the construction of vector K_2,a and K_2,b")
                print("Completed construction of vector K_2,a and K_2,b")
            elif force[0] == "ddxy":
                def g(x,y):
                    return force[5]((x+1)*self.L/2,y)
                for j in range(self.n):
                    self.K2b[j] += sp.integrate.dblquad(lambda x,y: g(x,y)*self.sol.basis[j].eval(x),force[3],force[4],2*force[1]/self.L-1,2*force[2]/self.L-1)[0]
            elif force[0] == "dl":
                def g(x):
                    return force[6]((x+1)*self.L/2)*np.cos(force[5]((x+1)*self.L/2))
                def h(x):
                    return force[6]((x+1)*self.L/2)*np.sin(force[5]((x+1)*self.L/2))
                for j in range(self.n):
                    self.K2a[j] += sp.integrate.quad(lambda x: g(x)*self.sol.basis[j].eval(x),2*force[1]/self.L-1,2*force[2]/self.L-1)[0]
                    self.K2b[j] += sp.integrate.quad(lambda x: h(x)*self.sol.basis[j].eval(x),2*force[1]/self.L-1,2*force[2]/self.L-1)[0]
            elif force[0] == "pl":
                for j in range(self.n):
                    self.K2a[j] += 2/(self.L)*force[5]*np.cos(force[4])*self.sol.basis[j].eval(2*force[1]/self.L-1)
                    self.K2b[j] += 2/(self.L)*force[5]*np.sin(force[4])*self.sol.basis[j].eval(2*force[1]/self.L-1)
            elif force[0] == "cm":
                for j in range(self.n):
                    self.K2a[j] += 2/(self.L)*force[3]*np.sin(force[2])*self.sol.basis[j].eval(2*force[1]/self.L-1)
                    self.K2b[j] -= 2/(self.L)*force[3]*np.cos(force[2])*self.sol.basis[j].eval(2*force[1]/self.L-1)
            else:
                exit("At least one of your forces contains a typo in its name.")
        self.K2a1 = self.K2a[:self.nbcv]
        self.K2a2 = self.K2a[self.nbcv:]
        self.K2b1 = self.K2b[:self.nbcw]
        self.K2b2 = self.K2b[self.nbcw:]

    def cC2m(self):
        ### Create K2
        self.C2 = np.zeros((self.n))
        for i,force in enumerate(self.f):
            if force[0] == "distt":
                for j in range(self.n):
                    def g(x):
                        return force[3]((x+1)*self.L/2)
                    self.C2[j] += sp.integrate.quad(lambda x: g(x)*self.sol.basis[j].eval(x),2*force[1]/self.L-1,2*force[2]/self.L-1)[0]
            elif force[0] == "directt":
                for j in range(self.n):
                    self.C2[j] += 2/self.L*force[2]*self.sol.basis[j].eval(2*force[1]/self.L-1)
            elif force[0] == "ddxz":
                def g(x,z):
                    return -force[5]((x+1)*self.L/2,z)*(z-self.zsc)
                for j in range(self.n):
                    self.C2[j] += sp.integrate.dblquad(lambda x,z: g(x,z)*self.sol.basis[j].eval(x),force[3],force[4],2*force[1]/self.L-1,2*force[2]/self.L-1)[0]
                    print("Currently at " + str(j / self.n * 100) + "% of the construction of vector K_2,c")
                print("Completed construction of vector K_2,c")
            elif force[0] == "ddxy":
                def g(x,y):
                    return force[5]((x+1)*self.L/2,y)*(y-self.ysc)
                for j in range(self.n):
                    self.C2[j] += sp.integrate.dblquad(lambda x,y: g(x,y)*self.sol.basis[j].eval(x),force[3],force[4],2*force[1]/self.L-1,2*force[2]/self.L-1)[0]
            elif force[0] == "dl":
                def g(x):
                    return -force[6]((x+1)*self.L/2)*np.cos(force[5]((x+1)*self.L/2))*(force[4]((x+1)*self.L/2)-self.zsc)
                def h(x):
                    return force[6]((x+1)*self.L/2)*np.sin(force[5]((x+1)*self.L/2))*(force[3]((x+1)*self.L/2)-self.ysc)
                for j in range(self.n):
                    self.C2[j] += sp.integrate.quad(lambda x: g(x)*self.sol.basis[j].eval(x),2*force[1]/self.L-1,2*force[2]/self.L-1)[0]
                    self.C2[j] += sp.integrate.quad(lambda x: h(x)*self.sol.basis[j].eval(x),2*force[1]/self.L-1,2*force[2]/self.L-1)[0]
            elif force[0] == "pl":
                for j in range(self.n):
                    self.C2[j] += -2/(self.L)*force[5]*np.cos(force[4])*self.sol.basis[j].eval(2*force[1]/self.L-1)*(force[3]-self.zsc)
                    self.C2[j] += 2/(self.L)*force[5]*np.sin(force[4])*self.sol.basis[j].eval(2*force[1]/self.L-1)*(force[2]-self.ysc)
            elif force[0] == "cm":
                pass
            else:
                exit("At least one of your forces contains a typo in its name.")
        self.C21 = self.C2[:self.nbct]
        self.C22 = self.C2[self.nbct:]

    def cLm(self):
        ### Create LHS
        self.LHS = 8*self.EIzz/self.L**3*((self.Uat.dot(self.K1a11)).dot(self.Ua)+((self.Uat).dot(self.K1a12)).dot(self.Ha)+(self.Hat.dot(self.K1a21)).dot(self.Ua)+(self.Hat.dot(self.K1a22)).dot(self.Ha))
        self.LHS += 8*self.EIyy/self.L**3*((self.Ubt.dot(self.K1b11)).dot(self.Ub)+((self.Ubt).dot(self.K1b12)).dot(self.Hb)+(self.Hbt.dot(self.K1b21)).dot(self.Ub)+(self.Hbt.dot(self.K1b22)).dot(self.Hb))
        self.LHS += 2*self.GJ/self.L*((self.Uct.dot(self.C111)).dot(self.Uc)+((self.Uct).dot(self.C112)).dot(self.Hc)+(self.Hct.dot(self.C121)).dot(self.Uc)+(self.Hct.dot(self.C122)).dot(self.Hc))

    def cRm(self):
        ### Create RHS
        self.RHS = self.L/2 * (self.K2a1.dot(self.Ua)+self.K2a2.dot(self.Ha))
        self.RHS += self.L/2 * (self.K2b1.dot(self.Ub)+self.K2b2.dot(self.Hb))
        self.RHS += self.L/2 * (self.C21.dot(self.Uc)+self.C22.dot(self.Hc))
        self.RHS -= 8*self.EIzz/self.L**3*((self.Uat.dot(self.K1a11).dot(self.F[:self.nbcv])+self.Hat.dot(self.K1a21.dot(self.F[:self.nbcv]))))
        self.RHS -= 8*self.EIyy/self.L**3*((self.Ubt.dot(self.K1b11).dot(self.F[self.nbcv:self.nbcv+self.nbcw])+self.Hbt.dot(self.K1b21.dot(self.F[self.nbcv:self.nbcv+self.nbcw]))))
        self.RHS -= 2*self.GJ/self.L*((self.Uct.dot(self.C111).dot(self.F[-self.nbct:])+self.Hct.dot(self.C121.dot(self.F[-self.nbct:]))))

    def solve(self):
        self.solext = np.linalg.solve(self.LHS,self.RHS)
        self.sol.coef = np.concatenate((self.Ua.dot(self.solext)+self.F[:self.nbcv],self.solext[:self.n-self.nbcv],
                                        self.Ub.dot(self.solext)+self.F[self.nbcv:self.nbcv+self.nbcw],self.solext[self.n-self.nbcv:2*self.n-self.nbcv-self.nbcw],
                                        self.Uc.dot(self.solext)+self.F[-self.nbct:],self.solext[2*self.n-self.nbcv-self.nbcw:]))
    def compute_deflections(self):
        self.cUm()
        self.cHam()
        self.cHbm()
        self.cHcm()
        self.cK1m()
        self.cC1m()
        self.cK2m()
        self.cC2m()
        self.cLm()
        self.cRm()
        self.solve()

    ### Physical domains
    def eval(self,x):
        return self.sol.eval(2*x/self.L-1)
    def fdeval(self,x):
        return self.sol.fdeval(2*x/self.L-1)[0]/self.L*2,self.sol.fdeval(2*x/self.L-1)[1]/self.L*2,self.sol.fdeval(2*x/self.L-1)[2]/self.L*2
    def sdeval(self,x):
        return self.sol.sdeval(2*x/self.L-1)[0]/(self.L/2)**2, self.sol.sdeval(2*x/self.L-1)[1]/(self.L/2)**2,self.sol.sdeval(2*x/self.L-1)[2]/(self.L/2)**2
    def tdeval(self,x):
        return self.sol.tdeval(2*x/self.L-1)[0]/(self.L/2)**3, self.sol.tdeval(2*x/self.L-1)[1]/(self.L/2)**3,self.sol.tdeval(2*x/self.L-1)[2]/(self.L/2)**3

    def Mz(self,x):
        return -self.EIzz*self.sol.sdeval(2*x/self.L-1)[0]/(self.L/2)**2
    def My(self,x):
        return -self.EIyy*self.sol.sdeval(2*x/self.L-1)[1]/(self.L/2)**2
    def Sy(self,x):
        return -self.EIzz*self.sol.tdeval(2*x/self.L-1)[0]/(self.L/2)**3
    def Sz(self,x):
        return -self.EIyy*self.sol.tdeval(2*x/self.L-1)[1]/(self.L/2)**3
    def T(self,x):
        return self.GJ*self.sol.fdeval(2*x/self.L-1)[2]/(self.L/2)
    def tau(self,x):
        return self.GJ*self.sol.sdeval(2*x/self.L-1)[2]/(self.L/2)**2

    def plotv(self):
        x = np.linspace(0,self.L,1000)
        fig = plt.figure(1)
        ax = plt.subplot(221)
        ax.plot(x, self.eval(x)[0], 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$v(x)$')
        string = 'Deflection in y, N = ' + str(self.n)
        ax.set_title(string)

        ax = plt.subplot(222)
        ax.plot(x, self.fdeval(x)[0], 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$dv/dx(x)$')
        string = 'Slope in y, N = ' + str(self.n)
        ax.set_title(string)

        ax = plt.subplot(223)
        ax.plot(x, self.Mz(x), 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'$M_z(x)$')
        string = 'Bending moment about z, N = ' + str(self.n)
        ax.set_title(string)

        ax = plt.subplot(224)
        ax.plot(x, self.Sy(x), 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$S_y(x)$')
        string = 'Shear force in y, N = ' + str(self.n)
        ax.set_title(string)
        plt.tight_layout()

        plt.show()
    def plotw(self):
        x = np.linspace(0,self.L,1000)
        fig = plt.figure(2)
        ax = plt.subplot(221)
        ax.plot(x, self.eval(x)[1], 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$w(x)$')
        string = 'Deflection in z, N = ' + str(self.n)
        ax.set_title(string)

        ax = plt.subplot(222)
        ax.plot(x, self.fdeval(x)[1], 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$dw/dx(x)$')
        string = 'Slope in z, N = ' + str(self.n)
        ax.set_title(string)

        ax = plt.subplot(223)
        ax.plot(x, self.My(x), 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'$M_y(x)$')
        string = 'Bending moment about y, N = ' + str(self.n)
        ax.set_title(string)

        ax = plt.subplot(224)
        ax.plot(x, self.Sz(x), 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$S_z(x)$')
        string = 'Shear force in z, N = ' + str(self.n)
        ax.set_title(string)
        plt.tight_layout()
        plt.show()
    def plotphi(self):
        x = np.linspace(0,self.L,1000)
        fig = plt.figure(3)
        ax = plt.subplot(221)
        ax.plot(x, self.eval(x)[2], 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$\phi(x)$')
        string = 'Twist, N = ' + str(self.n)
        ax.set_title(string)

        ax = plt.subplot(222)
        ax.plot(x, self.T(x), 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$T(x)$')
        string = 'Torque, N = ' + str(self.n)
        ax.set_title(string)

        ax = plt.subplot(223)
        ax.plot(x, self.tau(x), 'b', label="Approximate")
        # ax.plot(x,exactsol,'r', label = "Exact")
        ax.legend()
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'$\tau(x)$')
        string = 'Distributed torque, N = ' + str(self.n)
        ax.set_title(string)
        plt.tight_layout()
        plt.show()