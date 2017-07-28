#!/usr/bin/env python3

import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=4)
psi4.core.set_output_file('output.dat', False)


def diagonalize(F, A):
    """
    Diagonalize the Fock matrix

    Args:
        F: the Fock matrix
        A: a unitary transformation matrix

    Returns: eigenvalues, coefficients
    """
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def calculate_density(F, A, n_el):
    """
    Calculate the electronic density

    Args:
        F: the Fock matrix
        A: a unitary transformation matrix
        n_el: the number of electrons

    Returns: the electronic density
    """
    eps, C = diagonalize(F, A)
    Cocc = C[:, :n_el]
    return Cocc @ Cocc.T


def calculate_JK_SCF_energy(mol, n_el, e_conv, d_conv, basis):
    """
    Calculate the energy of a molecule using a basic SCF algorithm

    Args:
        mol: a psi4 mol with a geometry set
        n_el: number of electrons
        e_conv: energy convergence (usually around 1.e-6 eV)
        d_conv: gradient convergence (usually around 1.e-6)
        basis: a string with the basis set to use (such as 'sto-3g')

    Returns: the total energy
    """

    mol.update_geometry()
    mol.print_out()

    psi4.set_options({'basis': basis,
                      'scf_type': 'df',
                      'e_convergence': 1e-10,
                      'd_convergence': 1e-10})


    bas = psi4.core.BasisSet.build(mol, target=basis)

    mints = psi4.core.MintsHelper(bas)
    nbf = mints.nbf()

    if (nbf > 100):
        raise Exception("More than 100 basis functions!")

    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())

    H = T + V

    S = np.array(mints.ao_overlap())
    g = np.array(mints.ao_eri())

    # Roothaan equations
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)

    print(A @ S @ A)

    D = calculate_density(H, A, n_el)

    #Get orbital basis from a Wavefunction object
    wfn = psi4.core.Wavefunction.build(
            mol,
            psi4.core.get_global_option('basis')
    )
    orb = wfn.basisset()

    #Build the complementary JKFIT basis for the aVDZ basis
    aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other=basis)

    # The zero basis set
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

    # Build instance of MintsHelper
    mints = psi4.core.MintsHelper(orb)

    # Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
    Ppq = mints.ao_eri(zero_bas, aux, orb, orb)
    Ppq = np.squeeze(Ppq) # remove the 1-dimensions

    # Build and invert Coulomb metric, dimension (1, Naux, 1, Naux)
    metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
    metric.power(-0.5, 1.e-14)
    metric = np.squeeze(metric) # remove the 1-dimensions

    Qpq = np.einsum('QP,Ppq->Qpq', metric, Ppq)

    E_old = 0.0
    F_old = None
    iteration = -1
    max_iter = 100
    while iteration < max_iter:
        iteration += 1

        # Two-step build of J with Qpq and D
        X_Q = np.einsum('Qpq,pq->Q', Qpq, D)
        J = np.einsum('Qpq,Q->pq', Qpq, X_Q)

        # Two-step build of K with Qpq and D
        Z_Qqr = np.einsum('Qrs,sq->Qrq', Qpq, D)
        K = np.einsum('Qpq,Qrq->pr', Qpq, Z_Qqr)

        F = H + 2.0 * J - K

        grad = F @ D @ S - S @ D @ F

        grad_rms = np.mean(grad ** 2) ** 0.5

        E_electric = np.sum((F + H) * D)
        E_total = E_electric + mol.nuclear_repulsion_energy()

        E_diff = E_total - E_old
        E_old = E_total
        F_old = F
        if iteration == 0:
            print(" %12s  %16s  %10s  %10s" %
                  ("iteration", "E_total", "E_diff", "grad_rms"))
            print("---------------------------------------------------------")
        print(" %12d  %16.12f  %10.4e  %10.4e" %
              (iteration, E_total, E_diff, grad_rms))

        if abs(E_diff) < e_conv and grad_rms < d_conv:
            print("Convergence reached!")
            break

        D = calculate_density(F, A, n_el)

    if iteration >= max_iter and (E_diff >= e_conv or grad_rms >= d_conv):
        print("Failed to reach convergence!")
    else:
        print("SCF has finished!")

    return E_total

if __name__ == "__main__":
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)

    n_el = 5

    e_conv = 1.e-6
    d_conv = 1.e-6

    basis = "sto-3g"

    energy = calculate_JK_SCF_energy(mol, n_el, e_conv, d_conv, basis)

    psi4.set_options({'basis': basis,
                      'scf_type': 'df',
                      'e_convergence': 1e-10,
                      'd_convergence': 1e-10})
    psi4_energy = psi4.energy("SCF/" + basis, molecule=mol)
    print("Psi4_energy is", psi4_energy)

