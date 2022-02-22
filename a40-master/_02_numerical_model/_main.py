from I_moments_of_inertia import Moments_of_inertia
from III_aeroload import Aeroload
from shear_centre_and_flows import Shear_flow_and_centre
# from shear_centre_and_flows import Shear_flow_and_centre


"""
------------------------- NUMERICAL MODEL -------------------------------
Group A40
Plane: Fokker 100

"""

class Numerical_model:
    def __init__(self, data_dict):
        """store the data dictionary as a class variable"""
        self.data_dict = data_dict

    #Part I: Calculate the moments of inertia for the z and y axis. Computing the cross-sectional properties
    def moments_of_inertia(self):
        self.moi = Moments_of_inertia(self.data_dict)
        self.moi.calculate_centroid()
        self.data_dict = self.moi.calculate_moments_of_inertia_izz(self.data_dict)  # Append dictionary
        self.data_dict = self.moi.calculate_moments_of_inertia_iyy(self.data_dict)  # Append dictionary
    def torsional_properties(self):
        Shear_flow_and_centre.region_geometry(self)
        
        self.data_dict=Shear_flow_and_centre.shear_centre_and_flow_y_dir(self,self.data_dict)[11]
        self.data_dict=Shear_flow_and_centre.torsional_stiffness(self,self.data_dict)[0]

    #Part II: Shear and normal stress analysis
    def shear_normal_stress(self):
        pass

    #Part III: Aerodynamic loads
    def calculate_aeroload(self):
        self.al = Aeroload(self.data_dict)
        self.data_dict = self.al.get_loadfunction()

    #Part IV: Moments about the x-axis and resulting twist
    def moments_x_axis_twist(self):
        pass

    #Part V: Moments about the y-axis and resulting deflection
    def moments_y_axis_deflection(self):
        pass

    #Part VI: Moments about the z-axis and resulting deflection
    def moments_z_axis_deflection(self):
        pass

    #Part VII: Stress calculations
    def calculate_stresses(self):
        pass


# All starting values stored in a dictionary
default_data_dict = {
    "C_a": 0.505,
    "h_a": 0.161,
    "t_sk": 0.0011,
    "t_sp": 0.0024,
    "t_st": 0.0012,
    "h_st": 0.013,
    "w_st": 0.017,
    "n_st": 11,
    "x_1": 0.125,
    "x_2": 0.498,
    "x_3": 1.494,
    "x_a": 0.245,
    "l_a": 1.611,
    "d_1": 0.00389,
    "d_3": 0.001245,
    "theta": 30,
    "E": 73.1*10**9,
    "G": 28.0*10**9,
    "P": 49.2*10**3,
    "alpha": 0.18741,
    "l_sk":  0.432065,

    #Additional settings for the aerodynamic loads interpolation
    "aerodynamicload_path": "aerodynamicloadf100.dat",
    "interpolation_type_2d": "cubic"
}

#Only excecute the full class when main is run independently
if __name__ == '__main__':
    num_mod = Numerical_model(default_data_dict)
    num_mod.moments_of_inertia()
    num_mod.shear_normal_stress()
    num_mod.calculate_aeroload()
    num_mod.moments_x_axis_twist()
    num_mod.moments_y_axis_deflection()
    num_mod.moments_z_axis_deflection()
    num_mod.calculate_stresses()
