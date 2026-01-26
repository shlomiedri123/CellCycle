// Pybind11 bindings for the tau-leaping kernel.
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace simulation {
void tau_leap_step_batch(
    double dt,
    const double* ages,
    const double* mRNA_in,
    double* mRNA_out,
    const double* t_rep,
    const double* trans_prop_arr,
    const double* deg_prop_arr,
    const double* gamma_deg,
    std::size_t n_cells,
    std::size_t G,
    int MAX_MRNA_PER_GENE,
    const std::uint64_t* rng_seeds
);
}

static void validate_1d(const py::array& arr, const char* name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be 1-D");
    }
}

static void validate_2d(const py::array& arr, const char* name) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be 2-D");
    }
}

PYBIND11_MODULE(tau_kernel, m) {
    m.doc() = "Tau-leaping kernel for stochastic mRNA birth-death dynamics";

    m.def(
        "tau_leap_step_batch",
        [](
            double dt,
            py::array_t<double, py::array::c_style | py::array::forcecast> ages,
            py::array_t<double, py::array::c_style | py::array::forcecast> mRNA_in,
            py::array_t<double, py::array::c_style | py::array::forcecast> mRNA_out,
            py::array_t<double, py::array::c_style | py::array::forcecast> t_rep,
            py::array_t<double, py::array::c_style | py::array::forcecast> trans_prop,
            py::array_t<double, py::array::c_style | py::array::forcecast> deg_prop,
            py::array_t<double, py::array::c_style | py::array::forcecast> gamma_deg,
            int max_mrna_per_gene,
            py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> rng_seeds
        ) {
            validate_1d(ages, "ages");
            validate_1d(t_rep, "t_rep");
            validate_1d(gamma_deg, "gamma_deg");
            validate_1d(rng_seeds, "rng_seeds");
            validate_2d(mRNA_in, "mRNA_in");
            validate_2d(mRNA_out, "mRNA_out");
            validate_2d(trans_prop, "trans_prop");
            validate_2d(deg_prop, "deg_prop");

            const std::size_t n_cells = static_cast<std::size_t>(mRNA_in.shape(0));
            const std::size_t G = static_cast<std::size_t>(mRNA_in.shape(1));

            if (mRNA_out.shape(0) != n_cells || mRNA_out.shape(1) != G) {
                throw std::invalid_argument("mRNA_out must match mRNA_in shape");
            }
            if (trans_prop.shape(0) != n_cells || trans_prop.shape(1) != G) {
                throw std::invalid_argument("trans_prop must match mRNA_in shape");
            }
            if (deg_prop.shape(0) != n_cells || deg_prop.shape(1) != G) {
                throw std::invalid_argument("deg_prop must match mRNA_in shape");
            }
            if (ages.shape(0) != n_cells) {
                throw std::invalid_argument("ages length must match number of cells");
            }
            if (rng_seeds.shape(0) != n_cells) {
                throw std::invalid_argument("rng_seeds length must match number of cells");
            }
            if (t_rep.shape(0) != G || gamma_deg.shape(0) != G) {
                throw std::invalid_argument("t_rep and gamma_deg length must match number of genes");
            }

            const auto ages_buf = ages.request();
            const auto m_in_buf = mRNA_in.request();
            const auto m_out_buf = mRNA_out.request();
            const auto t_rep_buf = t_rep.request();
            const auto trans_buf = trans_prop.request();
            const auto deg_buf = deg_prop.request();
            const auto gamma_buf = gamma_deg.request();
            const auto seeds_buf = rng_seeds.request();

            py::gil_scoped_release release;
            simulation::tau_leap_step_batch(
                dt,
                static_cast<const double*>(ages_buf.ptr),
                static_cast<const double*>(m_in_buf.ptr),
                static_cast<double*>(m_out_buf.ptr),
                static_cast<const double*>(t_rep_buf.ptr),
                static_cast<const double*>(trans_buf.ptr),
                static_cast<const double*>(deg_buf.ptr),
                static_cast<const double*>(gamma_buf.ptr),
                n_cells,
                G,
                max_mrna_per_gene,
                static_cast<const std::uint64_t*>(seeds_buf.ptr)
            );
        },
        py::arg("dt"),
        py::arg("ages"),
        py::arg("mRNA_in"),
        py::arg("mRNA_out"),
        py::arg("t_rep"),
        py::arg("trans_prop"),
        py::arg("deg_prop"),
        py::arg("gamma_deg"),
        py::arg("max_mrna_per_gene"),
        py::arg("rng_seeds")
    );
}
