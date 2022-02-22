import pytest
from I_moments_of_inertia import Moments_of_inertia

default_data_dict = {
    "C_a": 1,
    "h_a": 0.5,
    "t_sk": 0.01,
    "t_sp": 0.01,
    "t_st": 0.01,
    "h_st": 0.05,
    "w_st": 0.05,
    "n_st": 7,
    "x_1": 1,
    "x_2": 2,
    "x_3": 3,
    "x_a": 0.1,
    "l": 4,
    "d_1": 2,
    "d_3": 2,
    "theta": 30,
    "E": 1,
    "P": 1
}

def test_centroid_location():
    """Testing the calculations for the computation of the centroid"""
    moi = Moments_of_inertia(default_data_dict)
    moi.calculate_centroid()
    #Test values obtained from calculating the centroid by hand
    assert moi.spar == 0.005
    assert round(moi.arc,5) == 0.00785
    assert round(moi.sheet,8) == 0.00790569
    assert moi.stringer == 0.001
    assert round(moi.totalarea,4) == 0.0357
    assert round(moi.stringerspacing,4) == 0.3381
    assert moi.centroid_y == 0
    assert round(moi.centroid_z,4) == -0.1696

def test_moments_Izz():
    """Testing moment round z axis with manual calculations"""
    moi = Moments_of_inertia(default_data_dict)
    moi.calculate_centroid()
    moi.calculate_moments_of_inertia_izz(default_data_dict)
    assert round(moi.i_zz,6) == 0.000675

def test_moments_Iyy():
    """Testing moment round y axis with manual calculations"""
    moi = Moments_of_inertia(default_data_dict)
    moi.calculate_centroid()
    moi.calculate_moments_of_inertia_iyy(default_data_dict)
    assert round(moi.i_yy,6) == 0.004885

