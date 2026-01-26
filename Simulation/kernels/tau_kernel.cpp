// Tau-leaping kernel for stochastic mRNA birth-death dynamics.
// Births ~ Poisson(Gamma * g(t) * O(t) * dt), deaths ~ Poisson(gamma * m * dt).
// Promoter occupancy O(t) and gene dosage g(t) are computed by the simulator.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

namespace simulation {

void tau_leap_step(
    double dt,
    double age,
    const double* mRNA_in,
    double* mRNA_out,
    const double* t_rep,
    const double* trans_prop_arr,
    const double* deg_prop_arr,
    const double* gamma_deg,
    std::size_t G,
    int MAX_MRNA_PER_GENE,
    std::uint64_t rng_seed
) {
    std::mt19937_64 rng(rng_seed);

    for (std::size_t g = 0; g < G; ++g) {
        const int m_current = static_cast<int>(mRNA_in[g]);
        const int copies = age < t_rep[g] ? 1 : 2; // Number of copies 
        std::poisson_distribution<int> birth_dist(trans_prop_arr[g] * dt);
        std::poisson_distribution<int> death_dist(deg_prop_arr[g] * dt);
        const int births = birth_dist(rng);
        int deaths = death_dist(rng);
        if (deaths > m_current) {
            deaths = m_current;
        }

        int updated = m_current + births - deaths;
        if (updated < 0) {
            updated = 0;
        } else if (updated > MAX_MRNA_PER_GENE) {
            updated = MAX_MRNA_PER_GENE;
        }
        // Transcription profile of each gene after the tau leap step 
        mRNA_out[g] = static_cast<double>(updated);
    }
}

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
) {
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
        const double age = ages[cell];
        const double* m_in = mRNA_in + (cell * G);
        double* m_out = mRNA_out + (cell * G);
        const double* trans_prop = trans_prop_arr + (cell * G);
        const double* deg_prop = deg_prop_arr + (cell * G);

        tau_leap_step(
            dt,
            age,
            m_in,
            m_out,
            t_rep,
            trans_prop,
            deg_prop,
            gamma_deg,
            G,
            MAX_MRNA_PER_GENE,
            rng_seeds[cell]
        );
    }
}

}  