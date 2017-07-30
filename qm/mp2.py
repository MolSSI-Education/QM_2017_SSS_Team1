#!/usr/bin/env python3

import numpy as np
import psi4

from . import scf

np.set_printoptions(suppress=True, precision=4)
psi4.core.set_output_file('output.dat', False)
psi4.set_memory("2 GB")
numpy_memory = 2 # in GB

def calculate_MP2_energy(scf_wfn):
    """
    Calculate the energy of a wave function using MP2 and the conventional
    approach. This assumes that an SCF has already been performed.

    Args:
        scf_wfn: the psi4 wave function from the SCF cycle

    Returns: a tuple: (mp2_os_corr, mp2_ss_corr)
    """

    # Number of Occupied orbitals & MOs
    ndocc = scf_wfn.nalpha()
    nmo = scf_wfn.nmo()

    # Get orbital energies and separate occupied & virtual orbitals
    eps = np.asarray(scf_wfn.epsilon_a())
    e_ij = eps[:ndocc]
    e_ab = eps[ndocc:]

    # Now calculate ERIs
    # Start with MintsHelper
    mints = psi4.core.MintsHelper(scf_wfn.basisset())

    # Memory check for ERI tensor
    I_size = (nmo**4) * 8.e-9
    print('\nSize of the ERI tensor will be %4.2f GB.' % I_size)
    memory_footprint = I_size * 1.5
    if I_size > numpy_memory:
        psi4.core.clean()
        raise Exception("Estimated memory utilization (%4.2f GB) exceeds \
                         allotted memory limit of %4.2f GB." \
                         % (memory_footprint, numpy_memory))

    # Build ERI Tensor
    I = np.asarray(mints.ao_eri())

    # Get MO coefficients from SCF wavefunction
    C = np.asarray(scf_wfn.Ca())
    Cocc = C[:, :ndocc]
    Cvirt = C[:, ndocc:]

    # Transform I -> I_mo @ O(N^5)
    tmp = np.einsum('pi,pqrs->iqrs', Cocc, I)
    tmp = np.einsum('qa,iqrs->iars', Cvirt, tmp)
    tmp = np.einsum('iars,rj->iajs', tmp, Cocc)
    I_mo = np.einsum('iajs,sb->iajb', tmp, Cvirt)

    # Compare our Imo to MintsHelper
    Co = scf_wfn.Ca_subset('AO','OCC')
    Cv = scf_wfn.Ca_subset('AO','VIR')
    MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
    print("Do our transformed ERIs match Psi4's? %s" \
          % np.allclose(I_mo, np.asarray(MO)))

    # Compute MP2 Correlation & MP2 Energy
    # Compute energy denominator array
    e_denom = 1 / (e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(-1, 1, 1)
              + e_ij.reshape(-1, 1) - e_ab)

    # Compute SS & OS MP2 Correlation with Einsum
    mp2_os_corr = np.einsum('iajb,iajb,iajb->', I_mo, I_mo, e_denom)
    mp2_ss_corr = np.einsum('iajb,iajb,iajb->', I_mo,
                            I_mo - I_mo.swapaxes(1,3), e_denom)

    return mp2_os_corr, mp2_ss_corr

if __name__ == "__main__":
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """) # The symmetry is essential for a reason unknown to me...

    n_el = 5

    e_conv = 1.e-8
    d_conv = 1.e-8

    basis = "sto-3g"

    psi4.set_options({'basis':        basis,
                      'scf_type':     'pk',
                      'mp2_type':     'conv', # We'll do conventional here
                      'e_convergence': e_conv,
                      'd_convergence': d_conv})

    scf_e = scf.calculate_basic_SCF_energy(mol, n_el, e_conv, d_conv, basis)

    # We want the psi4 wavefunction, but we will use our own energy
    _, scf_wfn = psi4.energy('SCF', return_wfn=True)

    mp2_os_corr, mp2_ss_corr = calculate_MP2_energy(scf_wfn)

    mp2_e = scf_e + mp2_os_corr + mp2_ss_corr

    # Calculate psi4 energy and compare
    psi4_energy = psi4.energy("mp2", molecule=mol)
    print("Do we match psi4 mp2?", np.allclose(psi4_energy, mp2_e))
