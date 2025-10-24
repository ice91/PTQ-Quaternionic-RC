from ptq.data.s4g_h_pipeline import K, _norm_name

def test_K_value():
    assert abs(K - 0.00484813681109536) < 1e-15

def test_norm_name():
    assert _norm_name("Ngc  1068") == "NGC1068"
    assert _norm_name("eso444-g084") == "ESO444G084"
    assert _norm_name("IC  2574") == "IC2574"