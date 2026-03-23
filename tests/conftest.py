from __future__ import annotations

import numpy as np
import pytest

from analysis.runtime import (
    build_affine_model,
    build_backend,
    build_physical_models,
    load_controller_config,
    load_sim_defaults,
    load_system_parameters,
)


@pytest.fixture(scope="session")
def system():
    return load_system_parameters()


@pytest.fixture(scope="session")
def controller_config():
    return load_controller_config()


@pytest.fixture(scope="session")
def sim_defaults():
    return load_sim_defaults()


@pytest.fixture(scope="session")
def affine_model(system):
    return build_affine_model(system)


@pytest.fixture(scope="session")
def backend(system):
    return build_backend(system, enable_contact=True)


@pytest.fixture(scope="session")
def physical_models(system):
    return build_physical_models(system)


@pytest.fixture(scope="session")
def zero_velocity():
    return np.zeros(3, dtype=float)

