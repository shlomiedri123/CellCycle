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
        // Stochastic sampling -> Replacing Gillaspie's algorithm with Poisson sampling 
        // So we won't have to go over each event for a exponential growing population of cells 
        std::poisson_distribution<int> birth_dist(trans_prop_arr[g] * dt);
        std::poisson_distribution<int> death_dist(deg_prop_arr[g] * dt);
        // rng should be limited for the amount of births and deaths possible in a single iteartion     
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

}  
