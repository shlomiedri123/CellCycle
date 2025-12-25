#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

namespace simulation {
void tau_leap_step(
    double dt,
    double age,
    const double* mRNA_in, // a point to a constant array of double inter rep 
    double* mRNA_out,
    const double* t_rep,
    const double* trans_prop,
    const double* deg_prop,
    const double* gamma_deg,
    std::size_t G,
    int MAX_MRNA_PER_GENE,
    std::uint64_t rng_seed
);
}

namespace {
void tau_leap_step_binding(
    double dt,
    double age,
    py::array_t<double, py::array::c_style | py::array::forcecast> mRNA_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> mRNA_out,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_rep,
    py::array_t<double, py::array::c_style | py::array::forcecast> trans_prop,
    py::array_t<double, py::array::c_style | py::array::forcecast> deg_prop,
    py::array_t<double, py::array::c_style | py::array::forcecast> gamma_deg,
    std::size_t G,
    int MAX_MRNA_PER_GENE,
    std::uint64_t rng_seed
) {
    const auto buf_in = mRNA_in.request();
    const auto buf_out = mRNA_out.request();
    const auto buf_trep = t_rep.request();
    const auto buf_trans = trans_prop.request();
    const auto buf_deg = deg_prop.request();
    const auto buf_gamma = gamma_deg.request();

    if (buf_in.ndim != 1 || buf_out.ndim != 1 || buf_trep.ndim != 1 ||
        buf_trans.ndim != 1 || buf_deg.ndim != 1 || buf_gamma.ndim != 1) {
        throw std::invalid_argument("All arrays must be 1-D");
    }
    if (buf_in.shape[0] != static_cast<py::ssize_t>(G) ||
        buf_out.shape[0] != static_cast<py::ssize_t>(G) ||
        buf_trep.shape[0] != static_cast<py::ssize_t>(G) ||
        buf_trans.shape[0] != static_cast<py::ssize_t>(G) ||
        buf_deg.shape[0] != static_cast<py::ssize_t>(G) ||
        buf_gamma.shape[0] != static_cast<py::ssize_t>(G)) {
        throw std::invalid_argument("Array lengths must equal G");
    }

    simulation::tau_leap_step(
        dt,
        age,
        static_cast<double*>(buf_in.ptr),
        static_cast<double*>(buf_out.ptr),
        static_cast<double*>(buf_trep.ptr),
        static_cast<double*>(buf_trans.ptr),
        static_cast<double*>(buf_deg.ptr),
        static_cast<double*>(buf_gamma.ptr),
        G,
        MAX_MRNA_PER_GENE,
        rng_seed
    );
}
}  // namespace

PYBIND11_MODULE(tau_kernel, m) {
    m.doc() = "Tau-leaping kernel for mRNA birth-death with replication timing";
    m.def(
        "tau_leap_step",
        &tau_leap_step_binding,
        py::arg("dt"),
        py::arg("age"),
        py::arg("mRNA_in"),
        py::arg("mRNA_out"),
        py::arg("t_rep"),
        py::arg("trans_prop"),
        py::arg("deg_prop"),
        py::arg("gamma_deg"),
        py::arg("G"),
        py::arg("MAX_MRNA_PER_GENE"),
        py::arg("rng_seed")
    );
}
