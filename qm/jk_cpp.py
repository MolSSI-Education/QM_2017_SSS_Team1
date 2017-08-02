#!/usr/bin/env python3

import numpy as np
from timeit import default_timer as timer

import psi4

import tensor_mult_numpy as tmn

# Clean up printing a little bit
np.set_printoptions(suppress=True, precision=4)
psi4.core.set_output_file('output.dat', False)

# Make sure we get the same random array
np.random.seed(0)

# A hydrogen molecule
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a ERI tensor
basisname = "cc-pvtz"
psi4.set_options({'basis': basisname,
                  'scf_type': 'df',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10})

basis = psi4.core.BasisSet.build(mol, target=basisname)
mints = psi4.core.MintsHelper(basis)
I = np.array(mints.ao_eri())

# Symmetric random density
nbf = I.shape[0]
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

# Reference
J_ref = np.einsum("pqrs,rs->pq", I, D)

numAttempts = 1000
average = 0.0
for i in range(numAttempts):
    start = timer()
    J = tmn.tensor_mult_numpy_J(I,D)
    end = timer()
    average += (end - start)

#print("J_ref is\n",J_ref)
#print("J is\n",J)

average *= 1.e6
average /= numAttempts

print("Average time for J out of", numAttempts, "attempts:", average, "ms")

K_ref = np.einsum("prqs,rs->pq", I, D)

average = 0.0
for i in range(numAttempts):
    start = timer()
    K = tmn.tensor_mult_numpy_K(I,D)
    end = timer()
    average += (end - start)

average *= 1.e6
average /= numAttempts

print("Average time for K out of", numAttempts, "attempts:", average, "ms")

#print("K_ref is\n", K_ref)
#print("K is\n", K)

############## OUR STUFF STARTS HERE ##########################
# Your implementation
#Get orbital basis from a Wavefunction object
wfn = psi4.core.Wavefunction.build(
        mol,
        psi4.core.get_global_option('basis')
)
orb = wfn.basisset()

#Build the complementary JKFIT basis for the aVDZ basis
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", fitrole="JKFIT",
                               other=basisname)

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
#print("metric is", metric)
#print("Ppq is", Ppq)
#print("Qpq is", Qpq)

#print("metric.shape is", metric.shape)
#print("Ppq.shape is", Ppq.shape)
#print("Qpq.shape is", Qpq.shape)


#J = np.random.rand(nbf, nbf)
#K = np.random.rand(nbf, nbf)

############## OUR STUFF ENDS HERE ##########################

# Make sure your implementation is correct
print("J is correct: %s" % np.allclose(J, J_ref))
print("K is correct: %s" % np.allclose(K, K_ref))
