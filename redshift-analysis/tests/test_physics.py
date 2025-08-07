import numpy as np
import pytest
from qfd_redshift.physics import QFDPhysics


@pytest.fixture
def physics():
    return QFDPhysics(qfd_coupling=0.85, redshift_power=0.6)


def test_redshift_dimming(physics):
    z = 0.5
    dimming = physics.calculate_redshift_dimming(z)
    assert dimming > 0

    z_array = np.array([0.1, 0.5, 1.0])
    dimmings = physics.calculate_redshift_dimming(z_array)
    assert len(dimmings) == len(z_array)
    assert np.all(dimmings > 0)


def test_qfd_cross_section(physics):
    z = 0.5
    cross_section = physics.calculate_qfd_cross_section(z)
    assert cross_section > physics.sigma_thomson


def test_optical_depth(physics):
    z = 0.5
    path_length_mpc = 1000.0
    tau = physics.calculate_optical_depth(z, path_length_mpc)
    assert tau > 0


def test_transmission(physics):
    z = 0.5
    path_length_mpc = 1000.0
    transmission = physics.calculate_transmission(z, path_length_mpc)
    assert 0 < transmission < 1
