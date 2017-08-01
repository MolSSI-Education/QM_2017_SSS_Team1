#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <iostream>
#include <omp.h>

namespace py = pybind11;

void print_arg(const std::string& s)
{
    std::cout << "String argument: " << s << std::endl;
}

// J = np.einsum("pqrs,rs->pq", g, D)
py::array_t<double> tensor_mult_numpy_J(py::array_t<double>& t1,
                                        py::array_t<double>& t2)
{
    py::buffer_info t1_info = t1.request();
    py::buffer_info t2_info = t2.request();

    const double* t1_data = static_cast<double*>(t1_info.ptr);
    const double* t2_data = static_cast<double*>(t2_info.ptr);

    size_t len_p = t1_info.shape[0];
    size_t len_q = t1_info.shape[1];
    size_t len_r = t1_info.shape[2];
    size_t len_s = t1_info.shape[3];

    std::vector<double> result(len_p * len_q);

#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (size_t p = 0; p < len_p; ++p) {
        for (size_t q = 0; q <= p; ++q) {
            double val = 0.0;
            for (size_t r = 0; r < len_r; ++r) {
                for (size_t s = 0; s < len_s; ++s) {
                    val += t1_data[p * len_q * len_r * len_s + q * len_r * len_s + r * len_s + s] * t2_data[r * len_s + s];
                }
            }
            // Take advantage of symmetry across the diagonal
            result[p*len_q + q] = val;
            result[q*len_q + p] = val;
        }
    }

    py::buffer_info resbuf =
        {
            result.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            { len_p, len_q },
            { len_q * sizeof(double), sizeof(double) }
        };


    return py::array_t<double>(resbuf);
}

// K = np.einsum("prqs,rs->pq", g, D)
py::array_t<double> tensor_mult_numpy_K(py::array_t<double>& t1,
                                        py::array_t<double>& t2)
{
    py::buffer_info t1_info = t1.request();
    py::buffer_info t2_info = t2.request();

    const double* t1_data = static_cast<double*>(t1_info.ptr);
    const double* t2_data = static_cast<double*>(t2_info.ptr);

    size_t len_p = t1_info.shape[0];
    size_t len_r = t1_info.shape[1];
    size_t len_q = t1_info.shape[2];
    size_t len_s = t1_info.shape[3];

    std::vector<double> result(len_p * len_q);

#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (size_t p = 0; p < len_p; ++p) {
        for (size_t q = 0; q <= p; ++q) {
            double val = 0.0;
            for (size_t r = 0; r < len_r; ++r) {
                for (size_t s = 0; s < len_s; ++s) {
                    val += t1_data[p * len_r * len_q * len_s + r * len_q * len_s + q * len_s + s] * t2_data[r * len_s + s];
                }
            }
            // Take advantage of symmetry across the diagonal
            result[p*len_q + q] = val;
            result[q*len_q + p] = val;
        }
    }

    py::buffer_info resbuf =
        {
            result.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            { len_p, len_q },
            { len_q * sizeof(double), sizeof(double) }
        };


    return py::array_t<double>(resbuf);
}

py::array_t<double> dgemm_numpy(double alpha,
                                py::array_t<double> A,
                                py::array_t<double> B)
{
    py::buffer_info A_info = A.request();
    py::buffer_info B_info = B.request();

    if (A_info.ndim != 2)
        throw std::runtime_error("A is not a matrix");
    if (B_info.ndim != 2)
        throw std::runtime_error("B is not a matrix");

    if (A_info.shape[1] != B_info.shape[0])
        throw std::runtime_error("Rows of A != columns of B");

    size_t C_nrows = A_info.shape[0];
    size_t C_ncols = B_info.shape[1];
    size_t n_k = A_info.shape[1]; // same as B_info.shape[0]

    const double* A_data = static_cast<double *>(A_info.ptr);
    const double* B_data = static_cast<double *>(B_info.ptr);

    std::vector<double> C_data(C_nrows * C_ncols);

    for (size_t i = 0; i < C_nrows; ++i) {
        for (size_t j = 0; j < C_ncols; ++j) {
            double val = 0.0;
            for (size_t k = 0; k < n_k; ++k) {
                val += A_data[i * n_k + k] * B_data[k * C_ncols + j];
            }
            val *= alpha;
            C_data[i*C_ncols + j] = val;
        }
    }

    py::buffer_info Cbuf =
        {
            C_data.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            { C_nrows, C_ncols },
            { C_ncols * sizeof(double), sizeof(double) }
        };


    return py::array_t<double>(Cbuf);
}

PYBIND11_PLUGIN(tensor_mult_numpy)
{
    py::module m("tensor_mult_numpy", "Tensor multiplication!");

    m.def("print_arg", &print_arg, "Prints the passed arg");
    m.def("tensor_mult_numpy_J", &tensor_mult_numpy_J);
    m.def("tensor_mult_numpy_K", &tensor_mult_numpy_K);
    m.def("dgemm_numpy", &dgemm_numpy);

    return m.ptr();
}
