#include <sstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include <math.h>

int main()
{
    const int size = 10000;
    std::vector<double> a(size);

    double start = omp_get_wtime();

    int x = -1;

#pragma omp parallel default(none) shared(a) firstprivate(x)
    {
        printf("x = %d\n", x);
        x = omp_get_thread_num() + 100;
#pragma omp for
        for (int i = 0; i < size; ++i) {
            a[i] = 0;
            for (int j = 0; j < size; ++j) {
                a[i] += sqrt(sqrt(i + j));
            }
        }
    }

    std::cout << "At the end, x = " << x << std::endl;

    double stop = omp_get_wtime();
    std::cout << "time: " << stop - start << " s" << std::endl;

    return 0;
}
