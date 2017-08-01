#include <sstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include <math.h>

double two_body_energy(int i, int j)
{
    return ((double)i + (double)j) / 10.0;
}

int main()
{
    const int nbodies = 10000;
    double energy = 0.0;

    double start = omp_get_wtime();

// Reduction clause can prevent a data race...
//#pragma omp parallel for reduction(+:energy)
// Always use dynamic scheduling
#pragma omp parallel for schedule(dynamic) reduction(+:energy)
    for (int i = 0; i < nbodies; ++i) {
        for (int j = i + 1; j < nbodies; ++j) {
            double eij = two_body_energy(i, j);
// atomic operations can prevent a data race...
//#pragma omp atomic
// Only execute one thread at a time. It uses a mutex.
//#pragma omp critical
            energy += eij;
        }
    }

    double stop = omp_get_wtime();
    std::cout << "time: " << stop - start << " s" << std::endl;

    printf("energy = %.5lf\n", energy);
    return 0;
}
