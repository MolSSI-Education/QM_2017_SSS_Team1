
#include <fstream>
#include <iostream>
#include <vector>

#include <lawrap/blas.h>

std::vector<double> readDataFromFile(std::string filename)
{
    std::vector<double> data;

    std::fstream f;
    f.open(filename.c_str());

    if (!f) {
        std::cout << "Failed to open file " << filename << "\n";
        return data;
    }

    double d;
    while (f >> d)
        data.push_back(d);

    std::cout << "size of data is: " << data.size() << "\n";

    return data;
}

void printMatrix(const std::vector<double>& m, int lda)
{
    std::cout << "printing matrix\n";
    for (size_t i = 0; i < m.size(); ++i) {
        std::cout << m[i] << " ";
        if ((i + 1) % lda == 0) {
            std::cout << "\n";
        }
    }
}

int main(int argc, char* argv[])
{
    std::vector<double> H = readDataFromFile("H.data");
    std::vector<double> F = readDataFromFile("F.data");
    std::vector<double> C = readDataFromFile("C.data");
    std::vector<double> S = readDataFromFile("S.data");

    auto C2 = C;

    // Calculate density. D = 2.0 * C * C^T
    std::vector<double> D(49, 0.0);

    char transa = 'N';
    char transb = 'T';
    int m = 7;
    int n = 7;
    int k = 5;
    double alpha = 2.0;
    int lda = 7;
    int ldb = 7;
    double beta = 0.0;
    int ldc = 7;

    LAWrap::gemm(transa, transb, m, n, k, alpha, C.data(), lda, C2.data(), ldb, beta, D.data(), ldc);

    printMatrix(D, lda);

    // Now calculate D * S
    std::vector<double> DS(49, 0.0);

    transa = 'N';
    transb = 'N';
    m = 7;
    n = 7;
    k = 7;
    alpha = 1.0;
    lda = 7;
    ldb = 7;
    beta = 0.0;
    ldc = 7;

    LAWrap::gemm(transa, transb, m, n, k, alpha, D.data(), lda, S.data(), ldb, beta, DS.data(), ldc);


    // Calculate the trace of D * S. This should be equal to the number of electrons
    double trace = 0.0;
    for (size_t i = 0; i < 7; ++i) {
        size_t idx = i * lda + i;
        trace += DS[idx];
    }

    std::cout << "Trace is: " << trace << "\n";


    // Now add together F and H
    std::vector<double> FPlusH = H;

    n = 49;
    alpha = 1.0;
    int incx = 1;
    int incy = 1;

    LAWrap::axpy(n, alpha, F.data(), incx, FPlusH.data(), incy);


    // Now multiply FPlusH with D and divide by 2
    std::vector<double> FPlusH_D(49, 0.0);

    transa = 'N';
    transb = 'N';
    m = 7;
    n = 7;
    k = 7;
    alpha = 0.5;
    lda = 7;
    ldb = 7;
    beta = 0.0;
    ldc = 7;

    LAWrap::gemm(transa, transb, m, n, k, alpha, FPlusH.data(), lda, D.data(), ldb, beta, FPlusH_D.data(), ldc);

    // Now calculate the trace. This should be the energy
    double energy = 0.0;
    for (size_t i = 0; i < 7; ++i) {
        size_t idx = i * lda + i;
        energy += FPlusH_D[idx];
    }
    std::cout << "Energy is: " << energy << " eV\n";
}
