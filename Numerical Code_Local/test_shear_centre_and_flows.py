import numpy as np
import matplotlib
import math as m 
from scipy import integrate
from scipy import interpolate
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt


from shear_centre_and_flows import Shear_flow_and_centre
from _02_numerical_model__main import default_data_dict
from _02_numerical_model_I_moments_of_inertia import MomentsOfInertia
from _02_numerical_model_III_aeroload import Aeroload
from _02_numerical_model_moments import Moments

SC=Shear_flow_and_centre(default_data_dict)

moi=MomentsOfInertia(default_data_dict)
mom=Moments(default_data_dict)

moi.calculate_centroid(default_data_dict)
mom.solve_for_reaction_forces()
mom.get_moment_functions(default_data_dict)
mom.get_displacement_functions(default_data_dict)
S_y_func=mom.get_shear_functions(default_data_dict)[0]
S_z_func=mom.get_shear_functions(default_data_dict)[1]
M_x_func=mom.get_moment_functions(default_data_dict)[0]


data_dict=SC.calculating_J(default_data_dict)


x_arr = np.arange(0, 1.611, 0.001)

S_y=[]
S_z=[]
M_x=[]
for i in range(len(x_arr)):
    S_y.append(S_y_func(x_arr)[i])
    S_z.append(S_z_func(x_arr)[i])
    M_x.append(M_x_func(x_arr[i]))

r1_array,r1_coord_z,r1_coord_y,r2_array,r2_coord_z,r2_coord_y,r3_array,r3_coord_z,r3_coord_y,r4_array,r4_coord_z,r4_coord_y,r5_array,r5_coord_z,r5_coord_y,r6_array,r6_coord_z,r6_coord_y=SC.region_geometry()
q1_span=[]
q2_span=[]
q3_span=[]
q4_span=[]
q5_span=[]
q6_span=[]
qb1_uptopoint_span=[]
qb2_uptopoint_span=[]
qb3_uptopoint_span=[]
qb4_uptopoint_span=[]
qb5_uptopoint_span=[]
qb6_uptopoint_span=[]
dtheta_dx_span=[]
q01_t_span=[]
q02_t_span=[]
q1_tot_span=[]
q2_tot_span=[]
q3_tot_span=[]
q4_tot_span=[]
q5_tot_span=[]
q6_tot_span=[]

'''
for i in range(len(S_y)):
    q1,q2,q3,q4,q5,q6,eta,A11,A12,A21,A22,data_dict=SC.shear_centre_and_flow_y_dir(default_data_dict,S_y[i],I_zz)
    q1_span.append(q1)
    q2_span.append(q2)
    q3_span.append(q3)
    q4_span.append(q4)
    q5_span.append(q5)
    q6_span.append(q6)
    qb1_uptopoint,qb2_uptopoint,qb3_uptopoint,qb4_uptopoint,qb5_uptopoint,qb6_uptopoint= SC.shear_flow_z_dir(default_data_dict,S_z[i],I_yy,S_y[i],I_zz,z_c)
    qb1_uptopoint_span.append(qb1_uptopoint)
    qb2_uptopoint_span.append(qb2_uptopoint)
    qb3_uptopoint_span.append(qb3_uptopoint)
    qb4_uptopoint_span.append(qb4_uptopoint)
    qb5_uptopoint_span.append(qb5_uptopoint)
    qb6_uptopoint_span.append(qb6_uptopoint)
    dtheta_dx,q01_t,q02_t=SC.torsional_stiffness(default_data_dict,M_x[i],S_y[i],I_zz)
    dtheta_dx_span.append(dtheta_dx)
    q01_t_span.append(q01_t)
    q02_t_span.append(q02_t)
    q1_tot,q2_tot,q3_tot,q4_tot,q5_tot,q6_tot=SC.sum_shearflowdistributions(data_dict,M_x[i],S_z[i],S_y[i],I_zz,I_yy,z_c)
    q1_tot_span.append(q1_tot)
    q2_tot_span.append(q2_tot)
    q3_tot_span.append(q3_tot)
    q4_tot_span.append(q4_tot)
    q5_tot_span.append(q5_tot)
    q6_tot_span.append(q6_tot)
'''
#SC.plot_shearflowdistributions(default_data_dict,M_x[500],S_z[500],S_y[500],I_zz,I_yy,z_c)


#This checks if the numerical model was correctly done
#cross section is at x=0.5

S_y_dev=-54316.63

I_zz_dev=3.78927412747565 *10**(-6) # from the dev model
#print("I_zz:",I_zz_dev)

S_z_dev=2163658.91

I_yy_dev=3.631150124937523*10**(-5) #from the dev model
#print("I_yy:",I_yy_dev)

M_x_dev=-3072.23


z_c_dev=SC.centroid_z(default_data_dict)-0.161/2


q1,q2,q3,q4,q5,q6,eta,A11,A12,A21,A22,data_dict=SC.shear_centre_and_flow_y_dir(default_data_dict,S_y_dev,I_zz_dev)
qb1_uptopoint,qb2_uptopoint,qb3_uptopoint,qb4_uptopoint,qb5_uptopoint,qb6_uptopoint= SC.shear_flow_z_dir(default_data_dict,S_z_dev,I_yy_dev,S_y_dev,I_zz_dev,z_c_dev)
dtheta_dx,q01_t,q02_t=SC.torsional_stiffness(default_data_dict,M_x_dev,S_y_dev,I_zz_dev)
q1_tot,q2_tot,q3_tot,q4_tot,q5_tot,q6_tot=SC.sum_shearflowdistributions(default_data_dict,M_x_dev,S_z_dev,S_y_dev,I_zz_dev,I_yy_dev,z_c_dev)
SC.plot_shearflowdistributions(default_data_dict,M_x_dev,S_z_dev,S_y_dev,I_zz_dev,I_yy_dev,z_c_dev)


'''
print("r1_z is:",r1_coord_z)
print("r1_y is:",r1_coord_y)
print("r2_z is:",r2_coord_z)
print("r1_y is:",r2_coord_y)
print("r3_z is:",r3_coord_z)
print("r3_y is:",r3_coord_y)
print("r4_z is:",r4_coord_z)
print("r4_y is:",r4_coord_y)
print("r5_z is:",r5_coord_z)
print("r5_y is:",r5_coord_y)
print("r46z is:",r6_coord_z)
print("r6_y is:",r6_coord_y)
'''

'''
print("q1_tot:", q1_tot)
print("q2_tot:", q2_tot)
print("q3_tot:", q3_tot)
print("q4_tot:", q4_tot)
print("q5_tot:", q5_tot)
print("q6_tot:", q6_tot)
'''
'''
print("q1:", q1)
print("q2:", q2)
print("q3:", q3)
print("q4:", q4)
print("q5:", q5)
print("q6:", q6)
'''

'''
print("qb1_uptopoint:",qb1_uptopoint)
print("qb2_uptopoint:",qb2_uptopoint)
print("qb3_uptopoint:",qb3_uptopoint)
print("qb4_uptopoint:",qb4_uptopoint)
print("qb5_uptopoint:",qb5_uptopoint)
print("qb6_uptopoint:",qb6_uptopoint)
'''
