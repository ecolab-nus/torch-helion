import pytest

from torch_helion.codegen.emit import safe_float_factors


@pytest.mark.parametrize("value", [1e-5, 1e-6, 2.5e-7, 0.125, 3.0, -1e-5, 1e-3, 0.0, 1e12])
def test_safe_float_factors_no_exponent(value):
    factors = safe_float_factors(value)
    assert all("e" not in repr(f) for f in factors)
    prod = 1.0
    for f in factors:
        prod *= f
    assert prod == pytest.approx(value, rel=1e-9)


def test_safe_float_single_factor_for_clean_values():
    assert safe_float_factors(0.5) == [0.5]
    assert safe_float_factors(0.00390625) == [0.00390625]
