from III_aeroload import Aeroload

default_data_dict = {
    "nx": 6,
    "nz": 8,
    "l_a": 3,
    "C_a": 5,
    "h_a": 1
}


def test_calc_xrange():
    al = Aeroload(default_data_dict)
    res = al.calc_xrange(default_data_dict["nx"], default_data_dict["l_a"])
    assert len(res) == default_data_dict["nx"]
    assert min(res) > 0
    assert max(res) < default_data_dict["l_a"]


def test_calc_zrange():
    al = Aeroload(default_data_dict)
    res = al.calc_zrange(default_data_dict["nz"], default_data_dict["C_a"], default_data_dict["h_a"])
    assert len(res) == default_data_dict["nz"]
    assert min(res) > - default_data_dict["C_a"] - default_data_dict["h_a"] / 2
    assert max(res) < default_data_dict["h_a"] / 2
