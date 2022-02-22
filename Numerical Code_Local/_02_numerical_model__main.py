from scipy import integrate, interpolate
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
    "d_2": 0.0,
    "d_3": 0.01245,
    "theta": 30,
    "E": 73.1*10**9,
    "G": 28.0*10**9,
    "P": 49.2*10**3,
    "eta": 0.0371,
    "J":7.748548555816593e-06,
    "I_yy":3.631150124937523*10**(-5),
    "I_zz":3.78927412747565 *10**(-6),
    # just because we have to choose something. should be zero at edges
    "q(x,z)": lambda x, z: 10,  # x * (x - 1.61) * 10,  # (z - 0.08) * (z + 0.4)
    "P": 49200,
    "dx": 0.001,

    #Additional settings for the aerodynamic loads interpolation
    "aerodynamicload_path": "aerodynamicloadf100.dat",
    "interpolation_type_2d": "cubic",
    "interpolation_type": interpolate.InterpolatedUnivariateSpline,
    "integration_type": integrate.quad

}
