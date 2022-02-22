
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
        
        self.h=data_dict['h_a']/2 #this is half the spar
        self.t_sk=data_dict['t_sk']
        self.t_sp=data_dict['t_sp']
        
        #Values for calculating the next shear flow
        self.C_a=data_dict['C_a']
        #Torsional stifness
        self.G=data_dict['G']
        #Values to be found
        self.eta=None
        self.J=None
        self.l_sk=np.sqrt(self.h**2 + (self.C_a-self.h)**2)
        self.alpha=np.arctan(self.h/(self.C_a-self.h))
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
            r2_array.append(i/100 * self.h)
            r2_coord_z.append(0)
            r2_coord_y.append(i/100 *self.h)
        #Region 3 - upper diagonal
        r3_array=[]
        r3_coord_z=[]
        r3_coord_y=[]
        for i in range(101):
            r3_array.append(i/100 * self.l_sk)
            r3_coord_z.append(-i/100 *(self.C_a-self.h))
            r3_coord_y.append((100-i)/100*self.h)
        #Region 4 - lower diagonal
        r4_array=[]
        r4_coord_z=[]
        r4_coord_y=[]
        for i in range(101):
            r4_array.append(i/100 * self.l_sk)
            r4_coord_z.append(-(100-i)/100*(self.C_a-self.h))
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
    def shear_centre_and_flow_y_dir(self,data_dict,S_y,I_zz):
        r1_array,r1_coord_z,r1_coord_y,r2_array,r2_coord_z,r2_coord_y,r3_array,r3_coord_z,r3_coord_y,r4_array,r4_coord_z,r4_coord_y,r5_array,r5_coord_z,r5_coord_y,r6_array,r6_coord_z,r6_coord_y=Shear_flow_and_centre.region_geometry(self)
        #Base shear flows
        #Region 1
        qb1_persection=[] #this is the base shear flow for region 1 for each discrete length
        qb1_uptopoint=[]  #this is the base shear flow from the beginning of the region up until a discrete length
        f=lambda theta: self.h*np.sin(theta)
        for i in range(100):
            qb1_persection.append(- S_y * self.t_sk/I_zz* self.h * integrate.quadrature(f, r1_array[i], r1_array[i+1])[0])
            qb1_uptopoint.append(-S_y*self.t_sk/I_zz* self.h * integrate.quadrature(f, r1_array[0], r1_array[i+1])[0])
        #Region 2
        qb2_persection=[]  #this is the base shear flow for region 2 for each discrete length
        qb2_uptopoint=[]   #this is the base shear flow from the beginning of the region up until a discrete length
        f=lambda y: y
        for i in range(100):
            qb2_persection.append(-S_y*self.t_sp/I_zz *integrate.quadrature(f, r2_array[i],r2_array[i+1])[0])
            qb2_uptopoint.append(-S_y*self.t_sp/I_zz *integrate.quadrature(f, r2_array[0],r2_array[i+1])[0])
        #Region 3
        qb3_persection=[]
        qb3_uptopoint=[]
        f=lambda s3: self.h-self.h/self.l_sk *s3
        for i in range(100):
            qb3_persection.append(-S_y*self.t_sk/I_zz*integrate.quadrature(f,r3_array[i],r3_array[i+1])[0]+qb1_uptopoint[-1]+qb2_uptopoint[-1])
            qb3_uptopoint.append(-S_y*self.t_sk/I_zz*integrate.quadrature(f,r3_array[0],r3_array[i+1])[0]+qb1_uptopoint[-1]+qb2_uptopoint[-1])
        #Region 4
        qb4_persection=[]
        qb4_uptopoint=[]
        f=lambda s4: -self.h/self.l_sk * s4
        for i in range(100):
            qb4_persection.append(-S_y*self.t_sk/I_zz*integrate.quadrature(f,r4_array[i],r4_array[i+1])[0]+qb3_uptopoint[-1])
            qb4_uptopoint.append(-S_y*self.t_sk/I_zz*integrate.quadrature(f,r4_array[0],r4_array[i+1])[0]+qb3_uptopoint[-1])
        #Region 5
        qb5_persection=[]
        qb5_uptopoint=[]
        f=lambda y: y
        for i in range(100):
            qb5_persection.append(-S_y*self.t_sp/I_zz *integrate.quadrature(f, r5_array[i],r5_array[i+1])[0])
            qb5_uptopoint.append(-S_y*self.t_sp/I_zz *integrate.quadrature(f, r5_array[0],r5_array[i+1])[0])
        #Region 6
        qb6_persection=[]
        qb6_uptopoint=[]
        f=lambda theta: self.h*np.sin(theta)
        for i in range(100):
            qb6_persection.append(-S_y*self.t_sk/I_zz* self.h * integrate.quadrature(f, r6_array[i], r6_array[i+1])[0]+qb4_uptopoint[-1]-qb5_uptopoint[-1])
            qb6_uptopoint.append(-S_y*self.t_sk/I_zz* self.h * integrate.quadrature(f, r6_array[0], r6_array[i+1])[0]+qb4_uptopoint[-1]-qb5_uptopoint[-1])

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
        r1_array=r1_array[:-1]
        r2_array=r2_array[:-1]
        r3_array=r3_array[:-1]
        r4_array=r4_array[:-1]
        r5_array=r5_array[:-1]
        r6_array=r6_array[:-1]
        #qb1 through qb6 need to be integrated over their respective regions
        #This means that their values need to be interpolated such that a new function is created.
        qb1_interp=interpolate.interp1d(r1_array,qb1_uptopoint,fill_value="extrapolate")
        qb2_interp=interpolate.interp1d(r2_array,qb2_uptopoint,fill_value="extrapolate")
        qb3_interp=interpolate.interp1d(r3_array,qb3_uptopoint,fill_value="extrapolate")
        qb4_interp=interpolate.interp1d(r4_array,qb4_uptopoint,fill_value="extrapolate")
        qb5_interp=interpolate.interp1d(r5_array,qb5_uptopoint,fill_value="extrapolate")
        qb6_interp=interpolate.interp1d(r6_array,qb6_uptopoint,fill_value="extrapolate")

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
        #print("q01 is:",q01)
        #print("q02 is:",q02)
        #Now it is time to calculate the actual shear flows by adding the redundant shear flow and the base shear flow
        q1=qb1_uptopoint-q01
        q2=qb2_uptopoint+q01+q02
        q3=qb3_uptopoint +q02
        q4=qb4_uptopoint +q02
        q5=qb5_uptopoint+q01+q02
        q6=qb6_uptopoint-q01
        
        #These array are to be interpolated once again such that they are prepared for the equation with integration that comes from the internal moment
        
        #print(len(q1))
        #print(len(r1_array))
        q1_interp=interpolate.interp1d(r1_array,q1,fill_value="extrapolate")
        q2_interp=interpolate.interp1d(r2_array,q2,fill_value="extrapolate")
        q3_interp=interpolate.interp1d(r3_array,q3,fill_value="extrapolate")
        q4_interp=interpolate.interp1d(r4_array,q4,fill_value="extrapolate")
        q5_interp=interpolate.interp1d(r5_array,q5,fill_value="extrapolate")
        q6_interp=interpolate.interp1d(r6_array,q6,fill_value="extrapolate")
        #Internal moment
        r=(self.C_a-self.h)*np.sin(self.alpha) #this is the perpendicular distance from the middle of the spar to the diagonal skin sections
        M_i= self.h**2 * integrate.quadrature(q1_interp,0,np.pi/2)[0] + self.h**2 * integrate.quadrature(q6_interp,-np.pi/2,0)[0] + r * integrate.quadrature(q3_interp,0,self.l_sk)[0] + r * integrate.quadrature(q4_interp,0,self.l_sk)[0]
        #Then, the distance from the middle of the spar to the shear centre is equal to
        eta=-M_i/S_y
        #print('eta is', eta)
        data_dict["eta"] = eta 
        return q1,q2,q3,q4,q5,q6,eta,A11,A12,A21,A22,data_dict

    def shear_flow_z_dir(self,data_dict,S_z,I_yy,S_y,I_zz,z_c):
        r1_array,r1_coord_z,r1_coord_y,r2_array,r2_coord_z,r2_coord_y,r3_array,r3_coord_z,r3_coord_y,r4_array,r4_coord_z,r4_coord_y,r5_array,r5_coord_z,r5_coord_y,r6_array,r6_coord_z,r6_coord_y=Shear_flow_and_centre.region_geometry(self)
        q1,q2,q3,q4,q5,q6,eta,A11,A12,A21,A22,eta=Shear_flow_and_centre.shear_centre_and_flow_y_dir(self,data_dict,S_y,I_zz)
        #Calculating the shear flow due to a shear force in the z direction
        #Only the base shear flow are needed to be found as the redundant shear flow is 0 (section is symmetric around the z axis)
        #Since the same sketch is used, the y and z coordinates and the integration region remain the same as from the shear flow from a shear force in the y direction
        #Base shear flows
        #Region 1
        qb1_persection=[] #this is the base shear flow for region 1 for each discrete length
        qb1_uptopoint=[]  #this is the base shear flow from the beginning of the region up until a discrete length
        f= lambda theta: (np.cos(theta))*self.h+z_c 
        for i in range(100):
            qb1_persection.append(-S_z*self.t_sk/I_yy* self.h * integrate.quadrature(f, r1_array[i], r1_array[i+1])[0])
            qb1_uptopoint.append(-S_z*self.t_sk/I_yy* self.h * integrate.quadrature(f, r1_array[0], r1_array[i+1])[0])
        #Region 2
        qb2_persection=[]  #this is the base shear flow for region 2 for each discrete length
        qb2_uptopoint=[]   #this is the base shear flow from the beginning of the region up until a discrete length
        f=lambda y: z_c
        for i in range(100):
            qb2_persection.append(-S_z*self.t_sp/I_yy *integrate.quadrature(f, r2_array[i],r2_array[i+1])[0])
            qb2_uptopoint.append(-S_z*self.t_sp/I_yy *integrate.quadrature(f, r2_array[0],r2_array[i+1])[0])
        #Region 3
        qb3_persection=[]
        qb3_uptopoint=[]
        f=lambda s3: z_c-(self.C_a-self.h)/self.l_sk*s3
        for i in range(100):
            qb3_persection.append(-S_z*self.t_sk/I_yy*integrate.quadrature(f,r3_array[i],r3_array[i+1])[0]+qb1_uptopoint[-1]+qb2_uptopoint[-1])
            qb3_uptopoint.append(-S_z*self.t_sk/I_yy*integrate.quadrature(f,r3_array[0],r3_array[i+1])[0]+qb1_uptopoint[-1]+qb2_uptopoint[-1])
        #Region 4
        qb4_persection=[]
        qb4_uptopoint=[]
        f=lambda s4: (-self.C_a+ self.h + z_c)+(self.C_a-self.h)/self.l_sk*s4
        for i in range(100):
            qb4_persection.append(-S_z*self.t_sk/I_yy*integrate.quadrature(f,r4_array[i],r4_array[i+1])[0]+qb3_uptopoint[-1])
            qb4_uptopoint.append(-S_z*self.t_sk/I_yy*integrate.quadrature(f,r4_array[0],r4_array[i+1])[0]+qb3_uptopoint[-1])
        #Region 5
        qb5_persection=[]
        qb5_uptopoint=[]
        f=lambda y: z_c
        for i in range(100):
            qb5_persection.append(-S_z*self.t_sp/I_yy *integrate.quadrature(f, r5_array[i],r5_array[i+1])[0])
            qb5_uptopoint.append(-S_z*self.t_sp/I_yy *integrate.quadrature(f, r5_array[0],r5_array[i+1])[0])
        #Region 6
        qb6_persection=[]
        qb6_uptopoint=[]
        f= lambda theta: (np.cos(theta))*self.h+z_c 
        for i in range(100):
            qb6_persection.append(-S_z*self.t_sk/I_yy* self.h * integrate.quadrature(f, r6_array[i], r6_array[i+1])[0]+qb4_uptopoint[-1]-qb5_uptopoint[-1])
            qb6_uptopoint.append(-S_z*self.t_sk/I_yy* self.h * integrate.quadrature(f, r6_array[0], r6_array[i+1])[0]+qb4_uptopoint[-1]-qb5_uptopoint[-1])
        
        return qb1_uptopoint,qb2_uptopoint,qb3_uptopoint,qb4_uptopoint,qb5_uptopoint,qb6_uptopoint

    def torsional_stiffness(self,data_dict,M_x,S_y,I_zz):
        A1 = np.pi*self.h**2/2.
        A2 = (self.C_a-self.h)*self.h
        X = np.array([[0.,0.],[0.,0.]])
        Y = np.array([[0.],[M_x/2]])

        X[0,0]=A2*(2*self.h/self.t_sp + np.pi*self.h/self.t_sk) + A1*2*self.h/self.t_sp
        X[0,1]=A2*(-2*self.h)/self.t_sp + A1*(-2*self.h/self.t_sp-self.l_sk/self.t_sk-self.l_sk/self.t_sk)
        X[1,0]=A1
        X[1,1]=A2
       
        X_inverted=np.linalg.inv(X)
        Q=X_inverted @ Y
        q01_t=Q[0]
        q02_t=Q[1]
        q1,q2,q3,q4,q5,q6,eta,A11,A12,A21,A22,data_dict=Shear_flow_and_centre.shear_centre_and_flow_y_dir(self,data_dict,S_y,I_zz)
        dtheta_dx=(A11*q01_t+A12*q02_t)/self.G
        #print("q01_t", q01_t)
        #print("q02_t", q02_t)
        return dtheta_dx,q01_t,q02_t

    def calculating_J(self,data_dict):
        E1=np.pi * self.h**2 /2
        E2=(self.C_a-self.h)*self.h
        D = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        E = np.array([0.,0.,0.])
       
        D[0,0] = 2.*E1
        D[0,1] = 2.*E2
        E[0] = 1

        D[1,0] = (self.h*np.pi/self.t_sk + 2*self.h/self.t_sp)/(2*E1)
        D[1,1] = (-2*self.h/self.t_sp)/(2*E1)
        D[1,2] = -1.
        E[1] = 0.

        
        D[2,0] = (-2*self.h/self.t_sp)/(2*E2)
        D[2,1] = (2*self.l_sk/self.t_sk + 2*self.h/self.t_sp)/(2*E2)
        D[2,2] = -1
        E[2] = 0.
        solve=np.linalg.solve(D,E)
        self.J = 1/solve[-1]
        #print('J is', self.J)
        data_dict["J"] =self.J
        return data_dict
        
    def sum_shearflowdistributions(self,data_dict,M_x,S_z,S_y,I_zz,I_yy,z_c):
        dtheta_dx,q01_t,q02_t = Shear_flow_and_centre.torsional_stiffness(self,data_dict,M_x,S_y,I_zz)
        q1,q2,q3,q4,q5,q6,eta,A11,A12,A21,A22,data_dict=Shear_flow_and_centre.shear_centre_and_flow_y_dir(self,data_dict,S_y,I_zz)
        qb1_uptopoint,qb2_uptopoint,qb3_uptopoint,qb4_uptopoint,qb5_uptopoint,qb6_uptopoint=Shear_flow_and_centre.shear_flow_z_dir(self,data_dict,S_z,I_yy,S_y,I_zz,z_c)
        

        q1_t=qb1_uptopoint- q01_t
        q2_t=  q01_t - q02_t + qb2_uptopoint
        q3_t=- q02_t +qb3_uptopoint
        q4_t=- q02_t + qb4_uptopoint
        q5_t= q01_t - q02_t + qb5_uptopoint
        q6_t= -q01_t + qb6_uptopoint
        
        q1_tot=[a+b for a,b in zip(q1_t,q1)]
        q2_tot=[a+b for a,b in zip(q2_t,q2)]
        q3_tot=[a+b for a,b in zip(q3_t,q3)]
        q4_tot=[a+b for a,b in zip(q4_t,q4)]
        q5_tot=[a+b for a,b in zip(q5_t,q5)]
        q6_tot=[a+b for a,b in zip(q6_t,q6)]
        
        '''
        q1_tot=qb1_uptopoint
        q2_tot=qb2_uptopoint
        q3_tot=qb3_uptopoint
        q4_tot=qb4_uptopoint
        q5_tot=qb5_uptopoint
        q6_tot=qb6_uptopoint
        
        q1_tot=q1
        q2_tot=q2
        q3_tot=q3
        q4_tot=q4
        q5_tot=q5
        q6_tot=q6
        
        q1_tot=-q01_t
        q2_tot=q01_t - q02_t
        q3_tot=- q02_t
        q4_tot=- q02_t
        q5_tot=q01_t - q02_t 
        q6_tot=-q01_t
        '''
        return q1_tot,q2_tot,q3_tot,q4_tot,q5_tot,q6_tot
    


    def centroid_z(self,data_dict):
        return (self.t_sk*np.pi*self.h*self.h/2+self.t_sp*2*self.h*self.h+2*self.t_sk*self.l_sk*(self.h+(self.C_a-self.h)/2))/(self.t_sk*np.pi*self.h+self.t_sp*2*self.h+2*self.t_sk*self.l_sk)
