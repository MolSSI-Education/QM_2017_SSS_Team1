#!/usr/bin/env python3
"""
Tests for our SCF functions
"""

# We will test stuff here
import pytest

import numpy as np
import psi4

from qm import jk_algorithm
from qm import scf


def test_calculate_basic_SCF_energy():
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)

    n_el = 5

    e_conv = 1.e-6
    d_conv = 1.e-6

    basis = "sto-3g"

    # Calculate the energy
    E_total = scf.calculate_basic_SCF_energy(mol, n_el, e_conv, d_conv, basis)

    # Now calculate the psi4 energy and make sure it matches
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)

    assert(np.allclose(psi4_energy, E_total))


def test_calculate_diis_SCF_energy():
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)

    n_el = 5

    e_conv = 1.e-6
    d_conv = 1.e-6

    basis = "sto-3g"

    energy = scf.calculate_basic_SCF_energy(mol, n_el, e_conv,
                                            d_conv, basis, diis=True)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)

    assert(np.allclose(psi4_energy, energy))


def test_calculate_JK_SCF_energy():
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)

    n_el = 5

    e_conv = 1.e-6
    d_conv = 1.e-6

    basis = "sto-3g"

    energy = jk_algorithm.calculate_JK_SCF_energy(mol, n_el, e_conv,
                                                  d_conv, basis)

    psi4.set_options({'basis': basis,
                      'scf_type': 'df',
                      'e_convergence': 1e-10,
                      'd_convergence': 1e-10})
    psi4_energy = psi4.energy("SCF/" + basis, molecule=mol)

    assert(np.allclose(psi4_energy, energy))
