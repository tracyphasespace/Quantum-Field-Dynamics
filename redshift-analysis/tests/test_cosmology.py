import numpy as np
import pytest
from qfd_redshift.cosmology import QFDCosmology


@pytest.fixture
def cosmo():
    return QFDCosmology(hubble_constant=70.0)


def test_hubble_distance(cosmo):
    # c = 299792.458 km/s
    # H0 = 70 km/s/Mpc
    # d_H = c/H0
    expected_dist = 299792.458 / 70.0
    assert np.isclose(cosmo.hubble_distance(), expected_dist)


def test_luminosity_distance(cosmo):
    z = 0.5
    comoving_dist = cosmo.comoving_distance(z)
    lum_dist = cosmo.luminosity_distance(z)
    assert lum_dist > comoving_dist
    assert np.isclose(lum_dist, comoving_dist * 1.5)


def test_angular_diameter_distance(cosmo):
    z = 0.5
    comoving_dist = cosmo.comoving_distance(z)
    ang_diam_dist = cosmo.angular_diameter_distance(z)
    assert ang_diam_dist < comoving_dist
    assert np.isclose(ang_diam_dist, comoving_dist / 1.5)


def test_lambda_cdm_distance(cosmo):
    # Test that it runs without error and returns a positive number
    z = 0.5
    dist = cosmo.lambda_cdm_distance(z)
    assert dist > 0

    # Test with an array of redshifts
    z_array = np.array([0.1, 0.5, 1.0])
    dists = cosmo.lambda_cdm_distance(z_array)
    assert len(dists) == len(z_array)
    assert np.all(dists > 0)
