#!/usr/bin/env python3

import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=4)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

mol.update_geometry()
mol.print_out()

e_conv = 1.e-6
d_conv = 1.e-6

n_el = 5

bas = psi4.core.BasisSet.build(mol, target="sto-3g")

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


def diagonalize(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def calculate_density(F, A, n_el):
    eps, C = diagonalize(F, A)
    Cocc = C[:, :n_el]
    return Cocc @ Cocc.T

D = calculate_density(H, A, n_el)

E_old = 0.0
F_old = None
iteration = -1
max_iter = 100
while iteration < max_iter:
    iteration += 1

    J = np.einsum("pqrs,rs->pq", g, D)
    K = np.einsum("prqs,rs->pq", g, D)

    F = H + 2.0 * J - K

    grad = F @ D @ S - S @ D @ F

    grad_rms = np.mean(grad ** 2) ** 0.5

    E_electric = np.sum((F + H) * D)
    E_total = E_electric + mol.nuclear_repulsion_energy()

    E_diff = E_total - E_old
    E_old = E_total
    F_old = F
    print(" %3d  %16.12f  %8.4e  %8.4e" %
          (iteration, E_total, E_diff, grad_rms))

    if E_diff < e_conv and grad_rms < d_conv:
        print("Convergence reached!")
        break

    D = calculate_density(F, A, n_el)

if iteration >= max_iter and (E_diff >= e_conv or grad_rms >= d_conv):
    print("Failed to reach convergence!")
else:
    print("SCF has finished!")

psi4.set_options({"scf_type": "pk"})
psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)

print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))
