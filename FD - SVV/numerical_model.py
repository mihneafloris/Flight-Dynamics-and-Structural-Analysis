
import numpy as np
import matplotlib.pyplot as plt
import control.matlab as control
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from math import *
#modes: phugoid, short_period, aperiodic_roll, dutch_roll, spiral
#modes: generic, generic_improved, our_data, our_data_improved
# Aircraft mass
m      =  6167.176           # mass [kg]

# Aerodynamic properties
e      =     0.717        # Oswald factor [ ]
CD0    =        0.021     # Zero lift drag coefficient [ ]
CLa    =         4.377    # Slope of CL-alpha curve [ ]
# Longitudinal stability
C_m_alpha    =    -0.5626         # longitudinal stabilty [ ]
C_m_delta_e   =    -1.1642         # elevator effectiveness [ ]

        # Aircraft geometry

S      = 30.00	          # wing area [m^2]
Sh     = 0.2 * S         # stabiliser area [m^2]
Sh_S   = Sh / S	          # [ ]
lh     = 0.71 * 5.968    # tail length [m]
c      = 2.0569	          # mean aerodynamic cord [m]
lh_c   = lh / c	          # [ ]
b      = 15.911	          # wing span [m]
bh     = 5.791	          # stabiliser span [m]
A      = b ** 2 / S      # wing aspect ratio [ ]
Ah     = bh ** 2 / Sh    # stabiliser aspect ratio [ ]
Vh_V   = 1	          # [ ]
ih     = -2 * np.pi / 180   # stabiliser angle of incidence [rad]

rho0   = 1.2250          # air density at sea level [kg/m^3] 
lambda_grad = -0.0065         # temperature gradient in ISA [K/m]
Temp0  = 288.15          # temperature at sea level in ISA [K]
R      = 287.05          # specific gas constant [m^2/sec^2K]
g      = 9.81            # [m/sec^2] (gravity constant)
  
W      = m * g            # [N]       (aircraft weight)

K_XX    = np.sqrt(0.019)
K_ZZ    = np.sqrt(0.042)
K_XZ    = 0.002
K_YY    = np.sqrt(1.3925)

        # Aerodynamic constants

Cmac   = 0                      # Moment coefficient about the aerodynamic centre [ ]
CNwa   = CLa                    # Wing normal force slope [ ]
CNha   = 2 * pi * Ah / (Ah + 2) # Stabiliser normal force slope [ ]
depsda = 4 / (A + 2)            # Downwash gradient [ ]


C_X_u    = -0.09500
C_X_alpha    = +0.47966		# Positive! (see FD lecture notes) 
C_X_alpha_dot = +0.08330
C_X_q    = -0.28170
C_X_delta_e   = -0.03728

        
C_Z_u    = -0.37616
C_Z_alpha    = -5.74340
C_Z_alpha_dot = -0.00350
C_Z_q    = -5.66290
C_Z_delta_e   = -0.69612

C_m_u    = +0.06990
C_m_alpha_dot = +0.17800
C_m_q    = -8.79415

C_Y_beta    = -0.7500
C_Y_beta_dot =  0     
C_Y_p    = -0.0304
C_Y_r    = +0.8495
C_Y_delta_a   = -0.0400
C_Y_delta_r   = +0.2300

C_l_beta    = -0.10260
C_l_p    = -0.71085
C_l_r    = +0.23760
C_l_delta_a   = -0.23088
C_l_delta_r   = +0.03440

C_n_beta    =  +0.1348
C_n_beta_dot =   0  

C_n_p    =  -0.0602
C_n_r    =  -0.2061
C_n_delta_a   =  -0.0120
C_n_delta_r   =  -0.0939
def root_solver(A,B,C):
    if B**2-4*A*C < 0 :
        Im = np.sqrt(-(B**2-4*A*C))/(2*A)
        Re = -B/(2*A)
        root_1=(Re,Im)
        root_2=(Re,-Im)
    else:
        root_1 = (-B + np.sqrt(B**2-4*A*C))/(2*A)
        root_2 = (-B - np.sqrt(B**2-4*A*C))/(2*A)
    return np.array([root_1,root_2])

def analytical_eigenvalues(mode,mode2):
    vector_bounds,vector,location = data_processing(mode,mode2)
    hp0=(data('Dadc1_bcAlt')[0][location[0]]*0.3048)[0]
    V0=(data('Dadc1_tas')[0][location[0]]*0.5144)[0]
    alpha0=(data('vane_AOA')[0][location[0]]*np.pi /180)[0]
    th0=(data('Ahrs1_Pitch')[0][location[0]]*np.pi/180)[0]
    Vt0=V0
    fuel_consumed = ((data('lh_engine_FU')[0][location[0]])[0] + (data('rh_engine_FU')[0][location[0]])[0])*0.453592
    m_atm= m-fuel_consumed
    rho=rho0 * np.power( ((1+(lambda_grad * hp0 / Temp0))), (-((g / (lambda_grad*R)) + 1)))
    W=m_atm * 9.81
    mu_c=m_atm/(rho*S*c)
    mu_b=m_atm/(rho*S*b)
    C_L = 2 * W / (rho * V0 ** 2 * S)
    C_X_0    = W * np.sin(th0) / (0.5 * rho * V0 ** 2 * S)
    C_Z_0    = -W * np.cos(th0) / (0.5 * rho * V0 ** 2 * S)
    if mode == 'phugoid':
        A = 2*mu_c*(C_Z_alpha * C_m_q - 2* mu_c*C_m_alpha)
        B = 2*mu_c*(C_X_u*C_m_alpha - C_m_u*C_X_alpha) + C_m_q* (C_Z_u*C_X_alpha - C_X_u* C_Z_alpha)
        C= C_Z_0 * (C_m_u * C_Z_alpha -  C_Z_u*C_m_alpha)
        return root_solver(A,B,C) *Vt0/c
    elif mode == 'short_period':
        A= 2*mu_c *K_YY**2*(2* mu_c - C_Z_alpha_dot)
        B= -2*mu_c*K_YY**2 *C_Z_alpha - (2*mu_c + C_Z_q) *C_m_alpha_dot - (2*mu_c - C_Z_alpha_dot) *C_m_q
        C= C_Z_alpha*C_m_q - (2*mu_c + C_Z_q) * C_m_alpha
        return root_solver(A,B,C) * Vt0/c
    elif mode == 'aperiodic_roll':
        return C_l_p / (4*mu_b * K_XX**2) *Vt0/b
    elif mode == 'dutch_roll':
        A= 8 * mu_b**2 *K_ZZ**2
        B= -2 * mu_b * (C_n_r + 2*K_ZZ**2*C_Y_beta)
        C= 4*mu_b*C_n_beta + C_Y_beta * C_n_r
        return root_solver(A,B,C)*Vt0/b
    elif mode == 'spiral':
        return 2*C_L * (C_l_beta*C_n_r - C_n_beta* C_l_r) / (C_l_p* (C_Y_beta * C_n_r + 4*mu_b*C_n_beta) - C_n_p * (C_Y_beta*C_l_r+4 * mu_b *C_l_beta)) * Vt0/b
    else:
        return print('You have not entered a valid mode')
    
def state_system(hp0,V0,alpha0,th0,m):
    Vt0=V0
    W=m * 9.81
    rho=rho0 * np.power( ((1+(lambda_grad * hp0 / Temp0))), (-((g / (lambda_grad*R)) + 1)))
    mu_c=m/(rho*S*c)
    mu_b=m/(rho*S*b)
    C_L= 2* W / (rho*S*V0**2)
    C_X_0    = W * np.sin(th0) / (0.5 * rho * V0 ** 2 * S)
    C_Z_0    = -W * np.cos(th0) / (0.5 * rho * V0 ** 2 * S)
    #Symmetric equations of motion
    S1=np.array([[-2*mu_c*c/Vt0,0,0,0],
                 [0,(C_Z_alpha_dot-2*mu_c)*c/Vt0,0,0],
                 [0,0,-c/Vt0,0],
                 [0,C_m_alpha_dot*c/Vt0,0,-2*mu_c*K_YY**2*c/Vt0]])

    S2=np.array([[C_X_u,C_X_alpha,C_Z_0,C_X_q],
                 [C_Z_u,C_Z_alpha,-C_X_0,C_Z_q+2*mu_c],
                 [0,0,0,1],
                 [C_m_u,C_m_alpha,0,C_m_q]])

    S3=np.array([[C_X_delta_e],
                 [C_Z_delta_e],
                 [0],
                 [C_m_delta_e]])

    #Asymmetric equations of motion
    A1=np.array([[(C_Y_beta_dot-2*mu_b)*b/Vt0,0,0,0],
                 [0,-1/2*b/Vt0,0,0],
                 [0,0,-4*mu_b*K_XX**2*b/Vt0,4*mu_b*K_XZ*b/Vt0],
                 [C_n_beta_dot*b/Vt0,0,4*mu_b*K_XZ*b/Vt0,-4*mu_b*K_ZZ**2*b/Vt0]])

    A2=np.array([[C_Y_beta,C_L,C_Y_p,C_Y_r-4*mu_b],
                 [0,0,1,0],
                 [C_l_beta,0,C_l_p,C_l_r],
                 [C_n_beta,0,C_n_p,C_n_r]])

    A3=np.array([[C_Y_delta_a,C_Y_delta_r],
                 [0,0],
                 [C_l_delta_a,C_l_delta_r],
                 [C_n_delta_a,C_n_delta_r]])
                 
    #State space matrices
    A_S= -np.dot(np.linalg.inv(S1),S2)
    B_S= - np.dot(np.linalg.inv(S1),S3)
    C_S= np.identity(4)
    C_S[0,0]=V0
    C_S[1,1]=1
    C_S[2,2]=1
    C_S[3,3]=V0/c
    D_S=np.zeros([4,1])
    system_S = control.ss(A_S,B_S,C_S,D_S)
    eig_S = np.linalg.eig(A_S)[0]

    A_A=[]
    B_A=[]
    A_A= -np.dot(np.linalg.inv(A1),A2)
    B_A= - np.dot(np.linalg.inv(A1),A3)
    C_A= np.zeros([4,4])
    C_A[0,0]=1
    C_A[1,1]=1
    C_A[2,2]=2*V0/b
    C_A[3,3]=2*V0/b
    D_A=np.zeros([4,2])
    system_A = control.ss(A_A,B_A,C_A,D_A)
    eig_A = np.linalg.eig(A_A)[0]
    return system_S,system_A,eig_S,eig_A

def data_names(mode):
    if mode == 'generic':
        data= loadmat('matlabR.mat')
    elif mode == 'our_data':
        data= loadmat('FTISxprt-20200309_flight1.mat')
    else:
        return print('You have not entered a valid mode')
    flight_data=data['flightdata']
    names = flight_data.dtype.names
    return names    
    
def data(name):
    data=loadmat('FTISxprt-20200309_flight1.mat')
    flight_data=data['flightdata']
    values= flight_data[0][0][name]['data'].flat[0]
    units = flight_data[0][0][name]['units'].flat[0].flat[0].flat[0]
    description = flight_data[0][0][name]['description'].flat[0].flat[0].flat[0]
    return values, units, description

def maneuvers_time(mode1):
    if mode1 == 'generic':
        return {'stationary_beginning' : (0,0),
                'short_period':    (3126,20),
                'phugoid':    (3219, 200),
                'dutch_roll':  (3455,15),
                'aperiodic_roll':    (3050,60),
                'spiral': (3590, 75)}
    else:
        return {'stationary_beginning' : (0,0),
                'short_period':    (3034,20),
                'phugoid':    (3260, 200),
                'dutch_roll':  (3450,15),
                'aperiodic_roll':    (3160,60),
                'spiral': (3690, 75)}

def data_processing(mode1,mode2):
    manuevers_period=maneuvers_time(mode2)[mode1]
    begin = manuevers_period[0]
    length = manuevers_period[1]
    begin_end=(begin, begin+length)
    vector=data(data_names(mode2)[48])[0]
    if mode2 == 'generic':
        vector= vector.reshape((48221,1))
    elif mode2 == 'our_data':
        vector= vector.reshape((49741,1))
    else:
        return print('You have not entered a valid mode')
    
    location=[np.where(vector == begin_end[0])[0][0],np.where(vector == begin_end[1])[0][0]]
    vector=vector[location[0]:location[1]]-begin_end[0]
    vector_bounds=(vector[0],np.round(vector[-1]))
    return vector_bounds,vector,location



def numerical_eigenmotions(mode1,mode2):  
    vector_bounds,vector,location = data_processing(mode1,mode2)
    hp=(data('Dadc1_bcAlt')[0][location[0]]*0.3048)[0]
    V=(data('Dadc1_tas')[0][location[0]]*0.5144)[0]
    if mode2 == 'our_data':
        alpha=(data('vane_AOA')[0][location[0]]*np.pi /180)[0] + 7.3375 *np.pi/180
        th=(data('Ahrs1_Pitch')[0][location[0]]*np.pi/180)[0] - 0.378 * np.pi/180
    else:
        alpha0=(data('vane_AOA')[0][location[0]]*np.pi /180)[0] -0.1737 *np.pi/180
        th0=(data('Ahrs1_Pitch')[0][location[0]]*np.pi/180)[0] - 0.7361 * np.pi/180
    fuel_consumed = ((data('lh_engine_FU')[0][location[0]])[0] + (data('rh_engine_FU')[0][location[0]])[0])*0.453592
    m_atm= m-fuel_consumed
    
    system_S,system_A,eig_S,eig_A = state_system(hp,V,alpha,th,m_atm)


    if mode1 in ['phugoid','short_period']:
        delta_elev=data('delta_e')[0][location[0]:location[1]]*np.pi/180
        y,T,x = control.lsim(system_S,delta_elev,vector)
        #print('Eigenvalues from the numerical model are:', eig_S)
        y[:,1]=y[:,1]*180/np.pi
        y[:,2]=y[:,2]*180/np.pi
        y[:,3]=y[:,3]*180/np.pi
        plot_values_units=['V [m/s]' , 'AoA [deg]', 'theta [deg]' , 'q [deg/s]']
    elif mode1 in ['aperiodic_roll','dutch_roll','spiral']:
        delta_ail=data('delta_a')[0][location[0]:location[1]].transpose() * -np.pi/180
        delta_rud=data('delta_r')[0][location[0]:location[1]].transpose() * -np.pi/180
        U=np.concatenate([delta_ail,delta_rud]).transpose()
        #print(U)
        y,T,x= control.lsim(system_A,U,vector)
        y[:,0] = y[:,0] * 180/ np.pi
        y[:,1] = y[:,1] * 180/ np.pi
        y[:,2] = y[:,2] * 180/ np.pi
        #print('Eigenvalues from the numerical model are:', eig_A)
        plot_values_units=['roll_angle [deg]', 'roll_rate [deg/s]', 'yawrate [deg/s]']
    else:
        return print('You have not entered a valid mode')
    
    
    
    reference=eigenmotion_experiment(mode1,mode2)
    pic, axis = plt.subplots(len(plot_values_units),1)
    print(len(plot_values_units))
    for i in range(len(plot_values_units)):
        axis[i].plot(vector,y[:,i] + reference[i][0])
        axis[i].plot(vector,reference[i])
        axis[i].set_xlim(vector_bounds)
        axis[i].set_ylabel(plot_values_units[i])
        axis[i].grid(True)
        pic.tight_layout()
    plt.show()
    return system_S,system_A,eig_S,eig_A


def eigenmotion_experiment(mode1,mode2):
    vector_bounds,vector,location = data_processing(mode1,mode2)
    if mode1 in ['phugoid','short_period']:
        V_tas = data('Dadc1_tas')[0][location[0]:location[1]]*0.5144 # 0.5144 comes from unit conversion
        alpha = data('vane_AOA')[0][location[0]:location[1]]

        pitch_angle = data('Ahrs1_Pitch')[0][location[0]:location[1]] 

        pitch_rate = data('Ahrs1_bPitchRate')[0][location[0]:location[1]]
        plot_values=[V_tas,alpha,pitch_angle,pitch_rate]
        plot_values_units=['V [m/s]' , 'AoA [deg]', 'theta [deg]' , 'q [deg/s]']
        
    elif mode1 in ['aperiodic_roll','dutch_roll','spiral']:
        roll_angle=data('Ahrs1_Roll')[0][location[0]:location[1]]
        roll_rate=data('Ahrs1_bRollRate')[0][location[0]:location[1]]
        yawrate=data('Ahrs1_bYawRate')[0][location[0]:location[1]]
        plot_values= [roll_angle,roll_rate,yawrate]
        plot_values_units=['roll_angle [deg]', 'roll_rate [deg/s]', 'yawrate [deg/s]']
        
    
    
    pic,axis=plt.subplots(len(plot_values),1)
    for i in range(len(plot_values)):
        axis[i].plot(vector,plot_values[i])
        axis[i].set_xlim(vector_bounds)
        axis[i].set_ylabel(plot_values_units[i])
        axis[i].grid(True)
        pic.tight_layout()
    #plt.show()

    if mode1 in ['short_period','phugoid']:
        return V_tas, alpha, pitch_angle, pitch_rate  
    elif mode1 in ['dutch_roll', 'aperiodic_roll', 'spiral']:
        return yawrate,roll_angle,roll_rate
    else:
        return 'You have not entered a valid mode'
'''
e_phugoid1,e_phugoid2=analytical_eigenvalues('phugoid','generic')
print('Phugoid analytical eigenvalues are:', e_phugoid1,e_phugoid2)
e_shortp1,e_shortp2=analytical_eigenvalues('short_period','generic')
print('Short period analytical eigenvalues are:', e_shortp1,e_shortp2)
e_dutch1,e_dutch2=analytical_eigenvalues('dutch_roll','generic')
print('Dutch roll analytical eigenvalues are:', e_dutch1,e_dutch2)
e_aproll=analytical_eigenvalues('aperiodic_roll','generic')
print('Aperiodic roll analytical eigenvalues are:', e_aproll)
e_spiral=analytical_eigenvalues('spiral','generic')
print('Spiral analytical eigenvalues are:', e_spiral)
'''
'''
exp_phug= eigenmotion_experiment('phugoid','generic')
exp_shortp= eigenmotion_experiment('short_period','generic')
exp_dutch= eigenmotion_experiment('dutch_roll','generic')
exp_aproll= eigenmotion_experiment('aperiodic_roll','generic')
exp_spiral= eigenmotion_experiment('spiral','generic') 
'''

'''
system_S,system_A,eig_S,eig_A= numerical_eigenmotions('phugoid','generic')
print('Phugoid numerical eigenvalues are:', eig_S)

system_S,system_A,eig_S,eig_A= numerical_eigenmotions('short_period','generic')
print('Short period numerical eigenvalues are:', eig_S)


system_S,system_A,eig_S,eig_A= numerical_eigenmotions('dutch_roll','generic')
print('Dutch roll numerical eigenvalues are:', eig_A)


system_S,system_A,eig_S,eig_A= numerical_eigenmotions('aperiodic_roll','generic')
print('Aperiodic roll numerical eigenvalues are:', eig_A)

system_S,system_A,eig_S,eig_A= numerical_eigenmotions('spiral','generic') 
print('Spiral numerical eigenvalues are:', eig_A)
'''

#Have to modify data function to get data from the our_data matlab file

e_phugoid1,e_phugoid2=analytical_eigenvalues('phugoid','our_data')
print('Phugoid analytical eigenvalues are:', e_phugoid1,e_phugoid2)
e_shortp1,e_shortp2=analytical_eigenvalues('short_period','our_data')
print('Short period analytical eigenvalues are:', e_shortp1,e_shortp2)
e_dutch1,e_dutch2=analytical_eigenvalues('dutch_roll','our_data')
print('Dutch roll analytical eigenvalues are:', e_dutch1,e_dutch2)
e_aproll=analytical_eigenvalues('aperiodic_roll','our_data')
print('Aperiodic roll analytical eigenvalues are:', e_aproll)
e_spiral=analytical_eigenvalues('spiral','our_data')
print('Spiral analytical eigenvalues are:', e_spiral)

system_S,system_A,eig_S,eig_A= numerical_eigenmotions('phugoid','our_data')
print('Phugoid numerical eigenvalues are:', eig_S)
system_S,system_A,eig_S,eig_A= numerical_eigenmotions('short_period','our_data')
print('Short period numerical eigenvalues are:', eig_S)
system_S,system_A,eig_S,eig_A= numerical_eigenmotions('dutch_roll','our_data')
print('Dutch roll numerical eigenvalues are:', eig_A)
system_S,system_A,eig_S,eig_A= numerical_eigenmotions('aperiodic_roll','our_data')
print('Aperiodic roll numerical eigenvalues are:', eig_A)
system_S,system_A,eig_S,eig_A= numerical_eigenmotions('spiral','our_data')
print('Spiral numerical eigenvalues are:', eig_A)



