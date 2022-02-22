
""" 
 This is the main file of the developer code of the AE3212-II project. 

 Author: Sam van Elsloo
 Date:   07/02/2021
"""

### Importing required packages
import numpy as np
import scipy as sp
import math as m
import matplotlib.pyplot as plt
import Energy
import Stiffness
import Stress
import Polyfit

################### Part I - parameters as in assignment #######################
"""
 The default input parameters are those of the b737-aircraft (the one used in
 the validation data). You should therefore update these parameters to those
 of the aircraft assigned to your group.
"""

datafile = "aerodynamicloadf100.dat" # Replace with file path to your 
                                     # aerodynamic loading.
Ca = 0.505              # m
la = 1.611              # m
x1 = 0.125              # m
x2 = 0.498              # m
x3 = 1.494              # m
xa = 0.245              # m
ha = 0.161              # m
tsk = 1.1/1000          # m
tsp = 2.4/1000          # m
tst = 1.2/1000          # m
hst = 13./1000          # m
wst = 17./1000          # m
nst = 11                # -
d1 = 0.00389            # m
d3 = 0.01245            # m
theta = m.radians(30)   # rad
P = 49.2*1000           # N

################### Part II - bending stiffness calculations ###################
### Create the cross-section object
""" 
 Note that only the cross-sectional geometry is passed on. Furthermore, you
 cannot disable this line, as the cross-section object is used in subsequent 
 calculations 
"""
crosssection = Stiffness.Crosssection(nst,Ca,ha,tsk,tsp,tst,hst,wst)

### Primary functions
""" 
 If you desire, you may disable this line, and manually overwrite the values 
 listed between lines 75-80
"""
crosssection.compute_bending_properties()  

### Auxiliary functions
""" 
 A plot of the cross-section, to inspect that the stringers have been placed 
 correctly, and that the position of the centroid makes sense. Blue cross is
 the centroid, red crosses are stringers.
"""
crosssection.plot_crosssection()    

### Access to important results
""""
 If you desire, you can manually overwrite these values, so that subsequent 
 parts of the program use 'nicer' numbers (e.g for verification purposes). 
"""
_ = crosssection.stcoord            # array containing stringer coordinates
_ = crosssection.totarea            # total cross-section area
_ = crosssection.yc                 # y-coordinate of the centroid
z_c = crosssection.zc                 # z-coordinate of the centroid
I_yy = crosssection.Iyy                # moment of inertia about y-axis
I_zz = crosssection.Izz                # moment of inertia about z-axis
print("I_yy is:",I_yy,"I_zz is:", I_zz)
print("z_c is:",z_c)
################### Part III - Torsional stiffness calculations ################
### Primary functions
""" 
 If you desire, you may disable this line, and manually overwrite the values 
 listed between lines 96-98
"""
crosssection.compute_shearcenter()   # Run the calculations
crosssection.compute_torsionalstiffness()   # Run the calculations

### Access to important results
"""" 
 If you desire, you can manually overwrite these values, so that subsequent 
 parts of the program use 'nicer' numbers (e.g for verification purposes). 
"""
_ = crosssection.ysc                 # y-coordinate of the centroid
z_sc = crosssection.zsc                 # z-coordinate of the centroid
_ = crosssection.J                   # torsional constant
print("z_sc is: ",z_sc) 
################### Part IV - Deflection calculations ##########################
### Definition of additional parameters
N = 15           # Number of basis functions to use in RR method (total number 
                 # of coefficients is 3*N). Note that the value of N = 15 is 
                 # just a place-holder value; it is up to you to check that this
                 # value produces sufficiently accurate results. 
E = 73.1e9       # E-modulus (Pa)
G = 28.0e9       # G-modulus (Pa)

### Create the aileron object
"""
 Merges the cross-sectional properties with the spanwise properties (length and
 material properties)
"""
aileron = Energy.Beam(la,crosssection,N,E,G)

""" 
 Define your boundary conditions; see manual for explanations. The provided 
 boundary conditions are the boundary conditions for the aileron as described in
 the assignment.
"""
aileron.addbcss(x1,0.,-ha/2.,-theta,d1)
aileron.addbcss(x1,0.,-ha/2.,m.pi/2-theta,0)
aileron.addbcss(x2,0.,-ha/2.,0,0)
aileron.addbcss(x2,0.,-ha/2.,m.pi/2,0.)
aileron.addbcss(x3,0.,-ha/2.,-theta,d3)
aileron.addbcss(x3,0.,-ha/2.,m.pi/2-theta,0)
aileron.addbcss(x2-xa/2.,ha/2.,0,m.pi/2.-theta,0)

""""
 Define your applied loading; see manual for explanations. The provided loading
 corresponds to that of the aileron as described in the aileron.

 First the aerodynamic load is imported by creating the rectangular grid as 
 described in the assignment. Then a linear regressor is used to generate a
 2D, continuous function. 

 Note that the kx = 5 and ky = 10 describe the highest-order polynomials that
 appear in the regressor. These are just place-holder values; it is up to you
 to check that these values are sufficiently accurate. Note that inceasing the
 orders will greatly increase the computational cost of the integration of the
 aerodynamic load for the deflections.
"""

### Import aerodynamic load
Nz = 81;
Nx = 41;

## Z-coordinates
thetaz = np.linspace(0,m.pi,Nz+1,endpoint=True)
z = 0.5*(1-np.cos(thetaz))*Ca;
z = -0.5*(z[:-1]+z[1:]);

## X-coordinates
thetax = np.linspace(0,m.pi,Nx+1,endpoint=True)
x = 0.5*(1-np.cos(thetax))*la;
x = 0.5*(x[:-1]+x[1:]);

xx,  zz = np.meshgrid(x, z)
data    = np.genfromtxt(datafile, delimiter = ',')
load    = Polyfit.polyfit(x,z,-data,kx=5,ky=10)     

aileron.addfpl(x2+xa/2.,ha/2.,0,m.pi/2.-theta,-P)
aileron.addfddxz(0,la,0,-Ca,load)

"""
 For demonstration purposes, the commented lines below show the methods you 
 should call to recreate a simple cantilevered beam, subjected to a load in the
 negative y-direction at z = 0, y = 0 within the cross-section. 

 Uncomment the lines above, and comment lines 121 - 163, and the code will run
 for the cantilevered beam. Note that a twist is present unless you set the
 shear center to be at (y,z) = (0,0) in the foregoing.
"""
# aileron.addbcss(0,0,0,0,0)
# aileron.addbcss(0,0,0,m.pi/2,0)
# aileron.addbcfo(0,0,0)
# aileron.addbcfo(0,m.pi/2,0)
# aileron.addbctw(0,0)

# aileron.addfpl(la,0,0,m.pi,P)


### Primary functions
aileron.compute_deflections() 

### Auxiliary functions
"""" A number of auxiliary functions and results are given to you. """

# Simplistic plotting procedures for a first check
aileron.plotv()             # Plot the deflections in y-direction, 
                            # its derivative, the bending moment about the 
                            # z-axis, and the shear force in y.
aileron.plotw()             # Plot the deflections in z-direction, its 
                            # derivative, the bending moment about the y-axis, 
                            # and the shear force in z.
aileron.plotphi()           # Plot the twist distribution, the torque and the 
                            # distributed torque.

## For custom post-processing of the solution
x = np.linspace(0,la,num = 10)  # Subsequent functions accept numpy-arrays

# Compute the deflections
_, _, _ = aileron.eval(x)       # Compute the three deflections
_, _, _ = aileron.fdeval(x)     # Compute their their first order derivative
_, _, _ = aileron.sdeval(x)     # Compute their their second order derivative
_, _, _ = aileron.tdeval(x)     # Compute their their third order derivative
# Compute the loading
_ = aileron.Sy(x)               # Compute the shear force in y
_ = aileron.Sz(x)               # Compute the shear force in z
_ = aileron.My(x)               # Compute the moment around the y-axis
_ = aileron.Mz(x)               # Compute the moment around the z-axis
_ = aileron.T(x)                # Compute the torque
_ = aileron.tau(x)              # Compute the distributed torque

### Access to important results
_ = aileron.Na              # Number of coefficients used to approximate v(x)
_ = aileron.Nb              # Number of coefficients used to approximate w(x)
_ = aileron.Nc              # Number of coefficients used to approximate phi(x)
_ = aileron.nbc             # Total number of boundary conditions
_ = aileron.nbcv            # Number of coefficients used for boundary 
                            # conditions for v; equal to N - Na
_ = aileron.nbcw            # Number of coefficients used for boundary 
                            # conditions for w; equal to N - Nb
_ = aileron.nbct            # Number of coefficients used for boundary 
                            # conditions for phi; equal to N - Nc

_ = aileron.Ha              # H_a matrix
_ = aileron.Hb              # H_b matrix
_ = aileron.Hc              # H_c matrix

_ = aileron.Ua              # Upsilon_a matrix
_ = aileron.Ub              # Upsilon_b matrix
_ = aileron.Uc              # Upsilon_c matrix

_ = aileron.K1              # K_{1,a}/K_{1,b} matrix
_ = aileron.C1              # K_{1,c} matrix
_ = aileron.K2a             # K_{2,a} matrix
_ = aileron.K2b             # K_{2,b} matrix
_ = aileron.C2              # K_{2,c} vector
_ = aileron.F               # F vector

_ = aileron.LHS             # Left-hand-side matrix
_ = aileron.RHS             # Right-hand-side vector

aerg = aileron.sol.coef     # Resulting coefficients.Includes both the 
                            # 'boundary' and 'interior' coefficients.

################### Part V - Stress calculations ###############################
### Create the stress state object. 
""" 
 This object will contain all information pertaining to the the stresses of the 
 aileron.
"""
Stressobject = Stress.Stressstate(crosssection)

### Define the forces and moments for which you want to know the stress state
x = 0.45
Sy = -5.62146633e+04
#Sy=1

Sz = 1.02735949e+05
#Sz=1

My = aileron.My(x)
Mz = aileron.Mz(x)
T = -2.78082390e+03
#T=1


### Primary functions
"""
 The following line should never be disabled, as its results are used in the 
 auxiliary functions
"""
Stressobject.compute_unitstressdistributions()

### Auxiliary functions
Stressobject.compute_stressdistributions(Sy,Sz,My,Mz,T)

### Some plotting functions
Stressobject.plot_shearflowdistributions()
Stressobject.plot_directstressdistributions()
Stressobject.plot_vonmisesstressdistributions()

### Access to important results
theta1 = np.linspace(0,m.pi/2,num = 100)
q1_tot_dev = Stressobject.q1f(theta1)             # Compute the shear flow distribution in
                                        # region 1
b = Stressobject.sigma1f(theta1)         # Compute the direct stress distribution 
                                        # in region 1
c = Stressobject.vm1(theta1)             # Compute the Von Mises stress 
                                        # distribution in region 1
r1_z,r1_y  = Stressobject.coord1(theta1)       # Compute the z,y-coordinates for 
                                        # region 1

y1 = np.linspace(0,ha/2.,num = 100)
q2_tot_dev = Stressobject.q2f(y1)                 # Compute the shear flow distribution in 
                                        # region 3
_ = Stressobject.sigma2f(y1)             # Compute the direct stress distribution 
                                        # in region 3
_ = Stressobject.vm2(y1)                 # Compute the Von Mises stress 
                                        # distribution in region 3
r2_z,r2_y = Stressobject.coord2(y1)           # Compute the z,y-coordinates for 
                                        # region 3

s1 = np.linspace(0,m.sqrt((Ca-ha/2.)**2+(ha/2.)**2),num = 100)
q3_tot_dev= Stressobject.q3f(s1)                 # Compute the shear flow distribution in 

                                        # region 4
_ = Stressobject.sigma3f(s1)             # Compute the direct stress distribution 
                                        # in region 4
_ = Stressobject.vm3(s1)                 # Compute the Von Mises stress 
                                        # distribution in region 4
r3_z,r3_y = Stressobject.coord3(s1)           # Compute the z,y-coordinates for 
                                        # region 4

s2 = np.linspace(0,m.sqrt((Ca-ha/2.)**2+(ha/2.)**2),num = 100)
q4_tot_dev = Stressobject.q4f(s2)                 # Compute the shear flow distribution in 
                                        # region 4
_ = Stressobject.sigma4f(s2)             # Compute the direct stress distribution 
                                        # in region 4
_ = Stressobject.vm4(s2)                 # Compute the Von Mises stress 
                                        # distribution in region 4
r4_z,r4_y = Stressobject.coord4(s2)           # Compute the z,y-coordinates for 
                                        # region 4

y2 = np.linspace(0,-ha/2.,num = 100)
q5_tot_dev = Stressobject.q5f(y2)                # Compute the shear flow distribution in 
                                        # region 5
_ = Stressobject.sigma5f(y2)             # Compute the direct stress distribution 
                                        # in region 5
_ = Stressobject.vm5(y2)                 # Compute the Von Mises stress 
                                        # distribution in region 5
r5_z,r5_y = Stressobject.coord5(y2)           # Compute the z,y-coordinates for 
                                        # region 5

theta2 = np.linspace(-m.pi/2,0,num = 100)
q6_tot_dev = Stressobject.q6f(theta2)             # Compute the shear flow distribution in 
                                        # region 6
_ = Stressobject.sigma6f(theta2)         # Compute the direct stress distribution 
                                        # in region 6
_ = Stressobject.vm6(theta2)             # Compute the Von Mises stress 
                                        # distribution in region 6
r6_z,r6_y = Stressobject.coord6(theta2)       # Compute the z,y-coordinates for 
                                        # region 6
'''                    
print("r1_z is:",r1_z)
print("r1_y is:",r1_y)
print("r2_z is:",r2_z)
print("r1_y is:",r2_y)
print("r3_z is:",r3_z)
print("r3_y is:",r3_y)
print("r4_z is:",r4_z)
print("r4_y is:",r4_y)
print("r5_z is:",r5_z)
print("r5_y is:",r5_y)
print("r46z is:",r6_z)
print("r6_y is:",r6_y)
'''
'''
print("q1_tot:", q1_tot_dev)
print("q2_tot:", q2_tot_dev)
print("q3_tot:", q3_tot_dev)
print("q4_tot:", q4_tot_dev)
print("q5_tot:", q5_tot_dev)
print("q6_tot:", q6_tot_dev)
'''

#Stressobject.compute_stressdistributions(Sy,Sz,My,Mz,T)