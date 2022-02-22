
#This function calculates the base shear flows, the redundant shear flows and then ultimately the shear centre based on a shear force in the vertical direction
#The inputs for this function are geometry, shear forces, these are going to be called in the main code 
#The functions requires integration, it will be done numerically by means of a scipy module
#We have the thin wall assumption and we also assume there are no stringers in the cross-section
#This code is going to be based on Figure 5.2 of the Simulation Plan

#This is where the imports are placed
import numpy as np
import matplotlib
import math as m 
from scipy import integrate
from scipy import interpolate
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt
class Shear_flow_and_centre:
    def __init__(self,data_dict):
        #These values are put here just so that we are able to run the code, they will not be included in the final code as they will be called in main
        self.I_zz=data_dict['I_zz']
        self.l_sk=data_dict['l_sk']
        self.h=data_dict['h'] #this is half the spar
        self.c=data_dict['c']
        self.S_y=data_dict['S_y']
        self.t_sk=data_dict['t_sk']
        self.t_sp=data_dict['t_sp']
        self.alpha=data_dict['alpha'] #this is the angle of the inclination of the skin
        #Values for calculating the next shear flow
        self.I_yy=data_dict['I_yy']
        self.z_c=data_dict['z_c']
        self.C_a=data_dict['C_a']
        self.S_z=data_dict['S_z']
        #Torsional stifness
        self.M_x=data_dict['M_x']
        self.G=data_dict['G']
        #End of the temporary values section


    #First big step in this code is to calculate the base shear flows
    #For this, since we are using numerical integration, for regions 1 through 6 we need to have an array with the regional locations and locations in the system of coordinates
    #Centre of the system of coordinates is in the middle of the spar with z positive towards the leading edge and y positive upwards
    def region_geometry(self):
        #Region 1 - this is the upper half of the semi-cricle- it will be based on theta in radians 
        r1_array=[] #nodes
        r1_coord_z=[]
        r1_coord_y=[]
        for i in range(101):
            r1_array.append(i/100 * np.pi/2)
            r1_coord_z.append(self.h*np.cos(i/100 * np.pi/2))
            r1_coord_y.append(self.h*np.sin(i/100 * np.pi/2))
        #Region 2 - upper half of the spar
        r2_array=[]
        r2_coord_z=[]
        r2_coord_y=[]
        for i in range(101):
            r2_array.append(i/100 * h)
            r2_coord_z.append(0)
            r2_coord_y.append(i/100 * h)
        #Region 3 - upper diagonal
        r3_array=[]
        r3_coord_z=[]
        r3_coord_y=[]
        for i in range(101):
            r3_array.append(i/100 * self.l_sk)
            r3_coord_z.append(-i/100 *(self.c-self.h))
            r3_coord_y.append((100-i)/100*self.h)
        #Region 4 - lower diagonal
        r4_array=[]
        r4_coord_z=[]
        r4_coord_y=[]
        for i in range(101):
            r4_array.append(i/100 * self.l_sk)
            r4_coord_z.append(-(100-i)/100*(self.c-self.h))
            r4_coord_y.append(-i/100*self.h)
        #Region 5 - lower half of the spar
        r5_array=[]
        r5_coord_z=[]
        r5_coord_y=[]
        for i in range(101):
            r5_array.append(-i/100*self.h)
            r5_coord_z.append(0)
            r5_coord_y.append(-i/100*self.h)
        #Region 6 - lower half of the semicircle
        r6_array=[]
        r6_coord_z=[]
        r6_coord_y=[]
        for i in range(101):
            r6_array.append(-(100-i)/100 * np.pi/2)
            r6_coord_z.append(self.h*np.sin(i/100 * np.pi/2))
            r6_coord_y.append(-self.h*np.cos(i/100 * np.pi/2))
        return r1_array,r1_coord_z,r1_coord_y,r2_array,r2_coord_z,r2_coord_y,r3_array,r3_coord_z,r3_coord_y,r4_array,r4_coord_z,r4_coord_y,r5_array,r5_coord_z,r5_coord_y,r6_array,r6_coord_z,r6_coord_y
    def shear_centre_and_flow_y_dir(self):
        r1_array,r1_coord_z,r1_coord_y,r2_array,r2_coord_z,r2_coord_y,r3_array,r3_coord_z,r3_coord_y,r4_array,r4_coord_z,r4_coord_y,r5_array,r5_coord_z,r5_coord_y,r6_array,r6_coord_z,r6_coord_y=Shear_flow_and_centre.region_geometry(self)
        #Base shear flows
        #Region 1
        qb1_persection=[] #this is the base shear flow for region 1 for each discrete length
        qb1_uptopoint=[]  #this is the base shear flow from the beginning of the region up until a discrete length
        for i in range(100):
            qb1_persection.append(- self.S_y * self.t_sk/self.I_zz* self.h**2 * integrate.quadrature(np.sin, r1_array[i], r1_array[i+1])[0])
            qb1_uptopoint.append(-self.S_y*self.t_sk/self.I_zz* self.h**2 * integrate.quadrature(np.sin, r1_array[0], r1_array[i+1])[0])
        #Region 2
        qb2_persection=[]  #this is the base shear flow for region 2 for each discrete length
        qb2_uptopoint=[]   #this is the base shear flow from the beginning of the region up until a discrete length
        f=lambda y: y
        for i in range(100):
            qb2_persection.append(-self.S_y*self.t_sp/self.I_zz *integrate.quadrature(f, r2_array[i],r2_array[i+1])[0])
            qb2_uptopoint.append(-self.S_y*self.t_sp/self.I_zz *integrate.quadrature(f, r2_array[0],r2_array[i+1])[0])
        #Region 3
        qb3_persection=[]
        qb3_uptopoint=[]
        f=lambda s3: self.h-self.h/self.l_sk *s3
        for i in range(100):
            qb3_persection.append(-self.S_y*self.t_sk/self.I_zz*integrate.quadrature(f,r3_array[i],r3_array[i+1])[0]+qb1_uptopoint[99]+qb2_uptopoint[99])
            qb3_uptopoint.append(-self.S_y*self.t_sk/self.I_zz*integrate.quadrature(f,r3_array[0],r3_array[i+1])[0]+qb1_uptopoint[99]+qb2_uptopoint[99])
        #Region 4
        qb4_persection=[]
        qb4_uptopoint=[]
        f=lambda s4: -self.h/self.l_sk * s4
        for i in range(100):
            qb4_persection.append(-self.S_y*self.t_sk/self.I_zz*integrate.quadrature(f,r4_array[i],r4_array[i+1])[0]+qb3_uptopoint[99])
            qb4_uptopoint.append(-self.S_y*self.t_sk/self.I_zz*integrate.quadrature(f,r4_array[0],r4_array[i+1])[0]+qb3_uptopoint[99])
        #Region 5
        qb5_persection=[]
        qb5_uptopoint=[]
        f=lambda y: y
        for i in range(100):
            qb5_persection.append(-self.S_y*self.t_sp/self.I_zz *integrate.quadrature(f, r5_array[i],r5_array[i+1])[0])
            qb5_uptopoint.append(-self.S_y*self.t_sp/self.I_zz *integrate.quadrature(f, r5_array[0],r5_array[i+1])[0])
        #Region 6
        qb6_persection=[]
        qb6_uptopoint=[]
        for i in range(100):
            qb6_persection.append(-self.S_y*self.t_sk/self.I_zz* self.h**2 * integrate.quadrature(np.sin, r6_array[i], r6_array[i+1])[0]+qb4_uptopoint[99]-qb5_uptopoint[99])
            qb6_uptopoint.append(-self.S_y*self.t_sk/self.I_zz* self.h**2 * integrate.quadrature(np.sin, r6_array[0], r6_array[i+1])[0]+qb4_uptopoint[99]-qb5_uptopoint[99])

        #Shear flow system of equations

        #A11 represents the perimeter of cell 1 on which q0,1 acts
        #A12 represents the spar of cell 1 on which q0,2 acts
        #A21 represents the spar of cell 2 on which q0,1 acts
        #A22 represents the perimeter of cell 2 on which q0,2 acts
        f=lambda x:1
        A11=self.h/self.t_sk*integrate.quadrature(f,0,np.pi/2)[0]+1/self.t_sp*integrate.quadrature(f,self.h,0)[0]+1/self.t_sp*integrate.quadrature(f,0,-self.h)[0]+self.h/self.t_sk * integrate.quadrature(f,-np.pi/2,0)[0]
        A12=-1/self.t_sp*integrate.quadrature(f,self.h,0)[0]-1/self.t_sp*integrate.quadrature(f,0,-self.h)[0]
        A21=-1/self.t_sp*integrate.quadrature(f,self.h,0)[0]-1/self.t_sp*integrate.quadrature(f,0,-self.h)[0]
        A22=1/self.t_sp*integrate.quadrature(f,self.h,0)[0]+1/self.t_sk*integrate.quadrature(f,0,self.l_sk)[0]+1/self.t_sk*integrate.quadrature(f,0,self.l_sk)[0]+1/self.t_sp*integrate.quadrature(f,-self.h,0)[0]

        #b1 and b2 now represent the rate of twist from the base shear flow around the 2 cells
        #qb1 through qb6 need to be integrated over their respective regions
        #This means that their values need to be interpolated such that a new function is created.
        qb1_interp=interpolate.interp1d(r1_array[:-1],qb1_uptopoint,fill_value="extrapolate")
        qb2_interp=interpolate.interp1d(r2_array[:-1],qb2_uptopoint,fill_value="extrapolate")
        qb3_interp=interpolate.interp1d(r3_array[:-1],qb3_uptopoint,fill_value="extrapolate")
        qb4_interp=interpolate.interp1d(r4_array[:-1],qb4_uptopoint,fill_value="extrapolate")
        qb5_interp=interpolate.interp1d(r5_array[:-1],qb5_uptopoint,fill_value="extrapolate")
        qb6_interp=interpolate.interp1d(r6_array[:-1],qb6_uptopoint,fill_value="extrapolate")

        #Since the shear flows are now interpolated we can use these functions to have them integrated once more in the rate of twist formula
        b1= self.h/self.t_sk * integrate.quadrature(qb1_interp,0,np.pi/2)[0]-1/self.t_sp*integrate.quadrature(qb2_interp,self.h,0)[0]-1/self.t_sp*integrate.quadrature(qb5_interp,0,-self.h)[0]+self.h/self.t_sk*integrate.quadrature(qb6_interp,-np.pi/2,0)[0]
        b2= 1/self.t_sp * integrate.quadrature(qb2_interp,0,self.h)[0] + 1/self.t_sk * integrate.quadrature(qb3_interp,0,self.l_sk)[0] + 1/self.t_sk * integrate.quadrature(qb4_interp,0,self.l_sk)[0] + 1/self.t_sp*integrate.quadrature(qb5_interp,-self.h,0)[0]
        #Creating the system of equations
        A=np.array([[A11,A12],[A21,A22]])
        B=np.array([[b1],[b2]])
        A_inverted=np.linalg.inv(A)
        #Solving for the redundant shear flow
        Q=-np.dot(A_inverted,B)
        q01=Q[0]
        q02=Q[1]
        #Now it is time to calculate the actual shear flows by adding the redundant shear flow and the base shear flow
        q1=[x + q01 for x in qb1_uptopoint]
        q2=[x - q01 + q02 for x in qb2_uptopoint]
        q3=[x + q02 for x in qb3_uptopoint]
        q4=[x + q02 for x in qb4_uptopoint]
        q5=[x -q01 + q02 for x in qb5_uptopoint]
        q6=[x + q01 for x in qb6_uptopoint]
        #These array are to be interpolated once again such that they are prepared for the equation with integration that comes from the internal moment
        q1_interp=interpolate.interp1d(r1_array[:-1],q1,fill_value="extrapolate")
        q2_interp=interpolate.interp1d(r2_array[:-1],q2,fill_value="extrapolate")
        q3_interp=interpolate.interp1d(r3_array[:-1],q3,fill_value="extrapolate")
        q4_interp=interpolate.interp1d(r4_array[:-1],q4,fill_value="extrapolate")
        q5_interp=interpolate.interp1d(r5_array[:-1],q5,fill_value="extrapolate")
        q6_interp=interpolate.interp1d(r6_array[:-1],q6,fill_value="extrapolate")
        #Internal moment
        r=self.l_sk*np.sin(self.alpha) #this is the perpendicular distance from the middle of the spar to the diagonal skin sections
        M_i= self.h**2 * integrate.quadrature(q1_interp,0,np.pi/2)[0] + r * integrate.quadrature(q3_interp,0,self.l_sk)[0] + r * integrate.quadrature(q4_interp,0,self.l_sk)[0] + self.h**2 * integrate.quadrature(q6_interp,-np.pi/2,0)
        #Then, the distance from the middle of the spar to the shear centre is equal to
        eta=-M_i/self.S_y
        return q1,q2,q3,q4,q5,q6,eta,A11,A12,A21,A22,eta

    def shear_flow_z_dir(self):
        r1_array,r1_coord_z,r1_coord_y,r2_array,r2_coord_z,r2_coord_y,r3_array,r3_coord_z,r3_coord_y,r4_array,r4_coord_z,r4_coord_y,r5_array,r5_coord_z,r5_coord_y,r6_array,r6_coord_z,r6_coord_y=Shear_flow_and_centre.region_geometry(self)
        q1,q2,q3,q4,q5,q6,eta,A11,A12,A21,A22,eta=Shear_flow_and_centre.shear_centre_and_flow_y_dir(self)
        #Calculating the shear flow due to a shear force in the z direction
        #Only the base shear flow are needed to be found as the redundant shear flow is 0 (section is symmetric around the z axis)
        #Since the same sketch is used, the y and z coordinates and the integration region remain the same as from the shear flow from a shear force in the y direction
        #Base shear flows
        #Region 1
        qb1_persection=[] #this is the base shear flow for region 1 for each discrete length
        qb1_uptopoint=[]  #this is the base shear flow from the beginning of the region up until a discrete length
        f= lambda theta: -(1-np.cos(theta))*self.h-self.z_c 
        for i in range(100):
            qb1_persection.append(-self.S_z*self.t_sk/self.I_yy* self.h * integrate.quadrature(f, r1_array[i], r1_array[i+1])[0])
            qb1_uptopoint.append(-self.S_z*self.t_sk/self.I_yy* self.h * integrate.quadrature(f, r1_array[0], r1_array[i+1])[0])
        #Region 2
        qb2_persection=[]  #this is the base shear flow for region 2 for each discrete length
        qb2_uptopoint=[]   #this is the base shear flow from the beginning of the region up until a discrete length
        f=lambda y: -self.h-self.z_c
        for i in range(100):
            qb2_persection.append(-self.S_z*self.t_sp/self.I_yy *integrate.quadrature(f, r2_array[i],r2_array[i+1])[0])
            qb2_uptopoint.append(-self.S_z*self.t_sp/self.I_yy *integrate.quadrature(f, r2_array[0],r2_array[i+1])[0])
        #Region 3
        qb3_persection=[]
        qb3_uptopoint=[]
        f=lambda s3: (-self.h-self.z_c)-(self.C_a-self.h)/self.l_sk*s3
        for i in range(100):
            qb3_persection.append(-self.S_z*self.t_sk/self.I_yy*integrate.quadrature(f,r3_array[i],r3_array[i+1])[0]+qb1_uptopoint[99]+qb2_uptopoint[99])
            qb3_uptopoint.append(-self.S_z*self.t_sk/self.I_yy*integrate.quadrature(f,r3_array[0],r3_array[i+1])[0]+qb1_uptopoint[99]+qb2_uptopoint[99])
        #Region 4
        qb4_persection=[]
        qb4_uptopoint=[]
        f=lambda s4: (-self.C_a-self.z_c)+(self.C_a-self.h)/self.l_sk*s4
        for i in range(100):
            qb4_persection.append(-self.S_z*self.t_sk/self.I_zyy*integrate.quadrature(f,r4_array[i],r4_array[i+1])[0]+qb3_uptopoint[99])
            qb4_uptopoint.append(-self.S_z*self.t_sk/self.I_zyy*integrate.quadrature(f,r4_array[0],r4_array[i+1])[0]+qb3_uptopoint[99])
        #Region 5
        qb5_persection=[]
        qb5_uptopoint=[]
        f=lambda y: -self.h-self.z_c
        for i in range(100):
            qb5_persection.append(-self.S_z*self.t_sp/self.I_yy *integrate.quadrature(f, r5_array[i],r5_array[i+1])[0])
            qb5_uptopoint.append(-self.S_z*self.t_sp/self.I_yy *integrate.quadrature(f, r5_array[0],r5_array[i+1])[0])
        #Region 6
        qb6_persection=[]
        qb6_uptopoint=[]
        f= lambda theta: -(1-np.cos(theta))*self.h-self.z_c 
        for i in range(100):
            qb6_persection.append(-self.S_z*self.t_sk/self.I_yy* self.h**2 * integrate.quadrature(np.sin, r6_array[i], r6_array[i+1])[0]+qb4_uptopoint[99]-qb5_uptopoint[99])
            qb6_uptopoint.append(-self.S_z*self.t_sk/self.I_yy* self.h**2 * integrate.quadrature(np.sin, r6_array[0], r6_array[i+1])[0]+qb4_uptopoint[99]-qb5_uptopoint[99])
        return qb1_uptopoint,qb2_uptopoint,qb3_uptopoint,qb4_uptopoint,qb5_uptopoint,qb6_uptopoint

    def torsional_stiffness(self):
        A1 = np.pi*self.h**2/2.
        A2 = (self.Ca-self.h)*self.h

        X = np.array([[0.,0.],[0.,0.]])
        Y = np.array([self.M_x/2],[0.])

        X[0,0]=A2*(2*self.h + np.pi*self.h) + A1*2*self.h
        X[0,1]=A2*(-2*self.h) + A1*(-2*self.h-self.t_sk-self.t_sk)
        X[1,0]=A1
        X[1,1]=A2
       
        X_inverted=np.linalg.inv(X)
        Q=np.linalg.dot(X_inverted,Y)
        q01_t=Q[0]
        q02_t=Q[1]

        dtheta_dx=(A11*q01_t+A12*q02_t)/self.G
        J=self.M_x/self.G/dtheta_dx
        return J, dtheta_dx,q01_t,q02_t

    def sum_shearflowdistributions(self):
        J, dtheta_dx,q01_t,q02_t = Shear_flow_and_centre.torsional_stiffness(self)
        qb1_uptopoint,qb2_uptopoint,qb3_uptopoint,qb4_uptopoint,qb5_uptopoint,qb6_uptopoint=Shear_flow_and_centre.shear_flow_z_dir(self)
        q1,q2,q3,q4,q5,q6,eta,A11,A12,A21,A22,eta=Shear_flow_and_centre.shear_centre_and_flow_y_dir(self)
        q1_t=[x - q01_t for x in qb1_uptopoint]
        q2_t=[x + q01_t + q02_t for x in qb2_uptopoint]
        q3_t=[x - q02_t for x in qb3_uptopoint]
        q4_t=[x - q02_t for x in qb4_uptopoint]
        q5_t=[x + q01_t + q02_t for x in qb5_uptopoint]
        q6_t=[x - q01_t for x in qb6_uptopoint]
        q1_tot=[a+b for a,b in zip(q1_t,q1)]
        q2_tot=[a+b for a,b in zip(q2_t,q2)]
        q3_tot=[a+b for a,b in zip(q3_t,q3)]
        q4_tot=[a+b for a,b in zip(q4_t,q4)]
        q5_tot=[a+b for a,b in zip(q5_t,q5)]
        q6_tot=[a+b for a,b in zip(q6_t,q6)]
        return q1_tot,q2_tot,q3_tot,q4_tot,q5_tot,q6_tot
    
    def make_segments(self, x, y):
        """
        Source:
        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def colorline(self, x, y, z=None, cmap=plt.get_cmap('copper'), 
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
        segments = Shear_flow_and_centre.make_segments(self, x, y)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                linewidth=linewidth, alpha=alpha)
        ax = plt.gca()
        ax.add_collection(lc)

        return lc

    def plot_shearflowdistributions(self):
        #Coordinate and s arrays
        r1_array,r1_coord_z,r1_coord_y,r2_array,r2_coord_z,r2_coord_y,r3_array,r3_coord_z,r3_coord_y,r4_array,r4_coord_z,r4_coord_y,r5_array,r5_coord_z,r5_coord_y,r6_array,r6_coord_z,r6_coord_y=Shear_flow_and_centre.region_geometry(self)
        q1_tot,q2_tot,q3_tot,q4_tot,q5_tot,q6_tot=Shear_flow_and_centre.sum_shearflowdistributions(self)

        ### Plot region 1
        x1 = r1_coord_z
        y1 = r1_coord_y
        z1 = q1_tot
        path = mpath.Path(np.column_stack([x1, y1]))
        verts = path.interpolated(steps=1).vertices
        x1, y1 = verts[:, 0,], verts[:, 1]
        maxabs = np.max(np.abs(z1))

        ### Plot region 2
        x2 = r2_coord_z
        y2 = r2_coord_y
        z2 = q2_tot
        path = mpath.Path(np.column_stack([x2, y2]))
        verts = path.interpolated(steps=1).vertices
        x2, y2 = verts[:, 0,], verts[:, 1]
        maxabs2 = np.max(np.abs(z2))
        maxabs = max(maxabs2,maxabs)

        ### Plot region 3
        x3 = r3_coord_z
        y3 = r3_coord_y
        z3 = q3_tot
        path = mpath.Path(np.column_stack([x3, y3]))
        verts = path.interpolated(steps=1).vertices
        x3, y3 = verts[:, 0,], verts[:, 1]
        maxabs3 = np.max(np.abs(z3))
        maxabs = max(maxabs3,maxabs)

        ### Plot region 4
        x4 = r4_coord_z
        y4 = r4_coord_y
        z4 = q4_tot
        path = mpath.Path(np.column_stack([x4, y4]))
        verts = path.interpolated(steps=1).vertices
        x4, y4 = verts[:, 0,], verts[:, 1]
        maxabs4 = np.max(np.abs(z4))
        maxabs = max(maxabs4,maxabs)

        ### Plot region 5
        x5 = r5_coord_z
        y5 = r5_coord_y
        z5 = q5_tot
        path = mpath.Path(np.column_stack([x5, y5]))
        verts = path.interpolated(steps=1).vertices
        x5, y5 = verts[:, 0,], verts[:, 1]
        maxabs5 = np.max(np.abs(z5))
        maxabs = max(maxabs5,maxabs)

        ### Plot region 6
        x6 = r6_coord_z
        y6 = r6_coord_y
        z6 = q5_tot
        path = mpath.Path(np.column_stack([x6, y6]))
        verts = path.interpolated(steps=1).vertices
        x6, y6 = verts[:, 0,], verts[:, 1]
        maxabs6 = np.max(np.abs(z6))
        maxabs = max(maxabs6,maxabs)
        fig = plt.figure(4)

        Shear_flow_and_centre.colorline(self,x1, y1, z1, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        Shear_flow_and_centre.colorline(self,x2, y2, z2, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        Shear_flow_and_centre.colorline(self,x3, y3, z3, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        Shear_flow_and_centre.colorline(self,x4, y4, z4, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        Shear_flow_and_centre.colorline(self,x5, y5, z5, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        Shear_flow_and_centre.colorline(self,x6, y6, z6, cmap=plt.get_cmap('jet'), 
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'), 
                                   norm=plt.Normalize(-maxabs, maxabs))
        sm.set_array([])
        plt.colorbar(sm,label=r'$q$ [N/m]',fraction = 0.20, pad = 0.04, 
                     orientation = "horizontal")
        plt.xlim(-self.C_a-0.1, 0.1)
        plt.ylim(-self.h-0.02, self.h+0.02)
        plt.axis('scaled')
        plt.gca().invert_xaxis()
        plt.xlabel(r'$-z$ [m]')
        plt.ylabel(r'$y$ [m]')
        plt.title('Shear flow distribution')
        plt.show()

#HAVENT MULTIPLIED BY THICKNESS TO GET SHEAR FORCE AND DIDNT SET DICTIONARY(EXAMPLE H)