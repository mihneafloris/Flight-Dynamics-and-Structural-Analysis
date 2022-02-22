import pytest
import numpy as np
from scipy import interpolate, integrate
from VI_moment_z_deflection_y import MomentZDeflectionY

default_data_dict = {
    "x_1": 1,
    "x_2": 2,
    "x_3": 3,
    "x_a": 0.1,
    "l_a": 4,
    "C_a": 2,
    "d_1": 2,
    "d_3": 2,
    "theta": 30,
    "I_zz": 1,
    "I_yy": 1,
    "E": 1,
    "q(x,z)": lambda x, z: z*(z-2) + x*(x-4),  # just because we have to choose something.
    "R_1_y": 1,
    "R_2_y": -2,
    "R_3_y": 1,
    "P_I": 1,
    "P": 1,
    "dx": 0.01,
    "dy": 0.01,
    "dz": 0.01,
    "interpolation_type": interpolate.InterpolatedUnivariateSpline,
    "integration_type": integrate.quad
}


def test_no_default_values():
    # This test ensures there are no default values for the moment, shear force and deflection
    # When we haven't calculated anything yet, it should be impossible to get graphs.
    with pytest.raises(RuntimeError):
        clean_instance = MomentZDeflectionY(data_dict=default_data_dict)
        clean_instance.plot_moment_around_z()

    with pytest.raises(RuntimeError):
        clean_instance = MomentZDeflectionY(data_dict=default_data_dict)
        clean_instance.plot_shear_force_in_y()

    with pytest.raises(RuntimeError):
        clean_instance = MomentZDeflectionY(data_dict=default_data_dict)
        clean_instance.plot_deflection_in_y()


# The following test is commented out because I haven't written the code for it yet.
# def test_deflection_points():
#     # This function will verify that the boundary conditions have been met and some other points.
#     clean_instance = MomentZDeflectionY(data_dict=default_data_dict)
#     m_z = clean_instance.get_moment_around_z()
#
#     assert m_z(default_data_dict['x_2']) < 10**(-2)  # Arbitrarily chosen precision
#     assert m_z(default_data_dict['x_1']) - default_data_dict['d_1'] < 10**(-2)
#     assert m_z(default_data_dict['x_3']) - default_data_dict['d_3'] < 10**(-2)
#
#     # the two following ones are based on the idea that the two sides should both be higher than d_1 and d_3
#     assert m_z(0) - m_z(default_data_dict['x_1']) > 0  # Just need to be positive
#     assert m_z(default_data_dict['l']) - m_z(default_data_dict['x_3']) > 0  # Just need to be positive

# def test_moment_around_z():  # Just testing that it runs. We'll get proper values later
#     clean_instance = MomentZDeflectionY(data_dict=default_data_dict)
#     clean_instance.get_moment_around_z()
#     assert clean_instance.plot_moment_around_z() is True


def test_validity_shear_force_calculation():
    clean_instance = MomentZDeflectionY(data_dict=default_data_dict)
    # To validate the method used, we will compare the actual result for something we know to what is predicted.
    # Also, please don't do this anywhere else than testing: _M_z is a protected variable for a reason!

    # -- CREATING A FICTITIOUS MOMENT FUNCTION M_z(x) = sin(x) --
    # added precision because this doesn't take much time anyway
    fine_x_array = np.arange(0, default_data_dict['l_a'],  default_data_dict['dx'] / 10)
    m_z = [np.sin(x) for x in fine_x_array]
    m_z = default_data_dict['interpolation_type'](fine_x_array, m_z)

    # -- GETTING THE CODE TO FIND THE DERIVATIVE --
    clean_instance._M_z = m_z
    shear_force_func = clean_instance.get_shear_force_in_y()
    x_array = np.arange(0, default_data_dict['x_a'], default_data_dict['dx'])
    assert np.max(np.abs(shear_force_func(x_array) - np.cos(x_array))) < 10**(-10)


if __name__ == "__main__":
    test_validity_shear_force_calculation()
