#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <cstdint>
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

void tau_leap_step_batch(
    double dt,
    const double* ages,
    const double* mRNA_in,
    double* mRNA_out,
    const double* t_rep,
    const double* trans_prop,
    const double* deg_prop,
    const double* gamma_deg,
    std::size_t n_cells,
    std::size_t G,
    int MAX_MRNA_PER_GENE,
    const std::uint64_t* rng_seeds
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

void tau_leap_step_batch_binding(
    double dt,
    py::array_t<double, py::array::c_style | py::array::forcecast> ages,
    py::array_t<double, py::array::c_style | py::array::forcecast> mRNA_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> mRNA_out,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_rep,
    py::array_t<double, py::array::c_style | py::array::forcecast> trans_prop,
    py::array_t<double, py::array::c_style | py::array::forcecast> deg_prop,
    py::array_t<double, py::array::c_style | py::array::forcecast> gamma_deg,
    int MAX_MRNA_PER_GENE,
    py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> rng_seeds
) {
    const auto buf_age = ages.request();
    const auto buf_in = mRNA_in.request();
    const auto buf_out = mRNA_out.request();
    const auto buf_trep = t_rep.request();
    const auto buf_trans = trans_prop.request();
    const auto buf_deg = deg_prop.request();
    const auto buf_gamma = gamma_deg.request();
    const auto buf_seeds = rng_seeds.request();

    if (buf_age.ndim != 1 || buf_in.ndim != 2 || buf_out.ndim != 2 ||
        buf_trep.ndim != 1 || buf_trans.ndim != 2 || buf_deg.ndim != 2 ||
        buf_gamma.ndim != 1 || buf_seeds.ndim != 1) {
        throw std::invalid_argument("Batch arrays must have expected dimensions");
    }

    const auto n_cells = buf_in.shape[0];
    const auto G = buf_in.shape[1];
    if (buf_out.shape[0] != n_cells || buf_out.shape[1] != G ||
        buf_trans.shape[0] != n_cells || buf_trans.shape[1] != G ||
        buf_deg.shape[0] != n_cells || buf_deg.shape[1] != G) {
        throw std::invalid_argument("2-D arrays must share (n_cells, G) shape");
    }
    if (buf_age.shape[0] != n_cells || buf_seeds.shape[0] != n_cells) {
        throw std::invalid_argument("ages and rng_seeds must have length n_cells");
    }
    if (buf_trep.shape[0] != G || buf_gamma.shape[0] != G) {
        throw std::invalid_argument("Gene-specific arrays must have length G");
    }

    simulation::tau_leap_step_batch(
        dt,
        static_cast<double*>(buf_age.ptr),
        static_cast<double*>(buf_in.ptr),
        static_cast<double*>(buf_out.ptr),
        static_cast<double*>(buf_trep.ptr),
        static_cast<double*>(buf_trans.ptr),
        static_cast<double*>(buf_deg.ptr),
        static_cast<double*>(buf_gamma.ptr),
        static_cast<std::size_t>(n_cells),
        static_cast<std::size_t>(G),
        MAX_MRNA_PER_GENE,
        static_cast<std::uint64_t*>(buf_seeds.ptr)
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
    m.def(
        "tau_leap_step_batch",
        &tau_leap_step_batch_binding,
        py::arg("dt"),
        py::arg("ages"),
        py::arg("mRNA_in"),
        py::arg("mRNA_out"),
        py::arg("t_rep"),
        py::arg("trans_prop"),
        py::arg("deg_prop"),
        py::arg("gamma_deg"),
        py::arg("MAX_MRNA_PER_GENE"),
        py::arg("rng_seeds")
    );
}
