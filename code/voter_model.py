import numpy as np
from collections import Counter
from scipy.stats import rv_discrete

def voter_model(S=1000, J=1000, T=1000, nu=0.0, 
                num_log_steps=100, num_log_steps_count=100,
                IC=None, opinion_counts=None, rng=None):
    """
    Simulates the Voter Model with optional speciation and log-scaled time sampling.

    Parameters
    ----------
    S : int
        Number of possible opinions (i.e., species or types in the neutral theory).
    J : int
        Total number of individuals in the system.
    T : int
        Number of generations to simulate (total number of steps is T * J).
    nu : float
        Speciation probability. With probability nu, a new opinion is introduced.
    num_log_steps : int
        Number of time points (logarithmically spaced) at which richness is stored.
    num_log_steps_count : int
        Number of time points (logarithmically spaced) at which opinion abundances are stored.
    IC : array-like or None
        Initial condition: list of initial opinions for each of the J individuals.
        If None, opinions are randomly assigned.
    opinion_counts : array-like or None
        Optional precomputed opinion counts (length S). If None, computed from IC.
    rng : numpy.random.Generator or None
        Optional random number generator instance.

    Returns
    -------
    opinion_counts_store : ndarray, shape (S, num_log_steps_count + 1)
        Array of opinion abundances at each stored time step.
    richness : ndarray
        Richness (number of distinct opinions) at each stored time step.
    time : ndarray
        Time points (in generations) at which richness was stored.
    time_c : ndarray
        Time points (in generations) at which opinion counts were stored.
    """
    if rng is None:
        rng = np.random.default_rng()

    total_time = T * J  # Total number of update steps

    # Initialize individual opinions
    if IC is None:
        # Random initial assignment of opinions
        IC = rng.integers(0, S, size=J)
    else:
        IC = np.array(IC)

    opinions = IC.copy()

    # Initialize opinion count vector if not provided
    if opinion_counts is None:
        opinion_counts = np.zeros(S, dtype=int)
        for o in opinions:
            opinion_counts[o] += 1

    # Compute logarithmic time points for recording
    log_steps = np.unique(np.round(np.logspace(np.log10(J), np.log10(total_time), num_log_steps)).astype(int))
    log_steps_count = np.unique(np.round(np.logspace(np.log10(J), np.log10(total_time), num_log_steps_count)).astype(int))

    # Containers for outputs
    richness = np.zeros(len(log_steps) + 1, dtype=int)
    opinion_counts_store = np.zeros((S, len(log_steps_count) + 1), dtype=int)
    time = np.zeros(len(log_steps) + 1)
    time_c = np.zeros(len(log_steps_count) + 1)

    # Store initial state
    richness[0] = np.sum(opinion_counts > 0)
    opinion_counts_store[:, 0] = opinion_counts.copy()
    time[0] = 0.0
    time_c[0] = 0.0

    k, k_c = 1, 1  # Logging step indices

    for t in range(1, total_time + 1):
        # Choose a random individual to update
        i = rng.integers(J)

        # With probability (1 - nu), adopt opinion from a random peer
        if rng.random() >= nu:
            opinions[i] = opinions[rng.integers(J)]
        else:
            # With probability nu, adopt a new random opinion (speciation)
            opinions[i] = rng.integers(S)

        # Record richness at log-spaced steps
        if k < len(log_steps) and t == log_steps[k - 1]:
            richness[k] = len(np.unique(opinions))
            time[k] = t / J
            # Early stopping if system reaches consensus (only one opinion left)
            if nu == 0 and richness[k] == 1:
                return opinion_counts_store, richness[:k+1], time[:k+1], time_c[:k_c]
            k += 1

        # Record opinion counts at log-spaced steps
        if k_c < len(log_steps_count) and t == log_steps_count[k_c - 1]:
            counts = Counter(opinions)
            opinion_counts = np.zeros(S, dtype=int)
            for key, val in counts.items():
                opinion_counts[key] = val
            opinion_counts_store[:, k_c] = opinion_counts
            time_c[k_c] = t / J
            k_c += 1
            
        # if last elemnt of richness is 0, then eliminate the last element from richness and time
        if k == len(log_steps) and richness[-1] == 0:
            richness = richness[:-1]
            time = time[:-1]

    return opinion_counts_store, richness, time, time_c



def logseries_distribution(theta=0.99, max_integer=1000):
    """
    Returns a truncated Logarithmic Series distribution as a scipy rv_discrete object.

    The PMF is defined as:
        p(k) = -θ^k / (k * log(1 - θ)), for k in 1, 2, ..., max_integer

    Parameters
    ----------
    theta : float
        Parameter of the distribution, must be in (0,1).
    max_integer : int
        Maximum integer to include in the truncation.

    Returns
    -------
    dist : rv_discrete
        A scipy.stats discrete distribution object that can be sampled from.
    """
    if not (0 < theta < 1):
        raise ValueError("theta must be in the interval (0, 1).")

    # Vector of integers 1 to max_integer
    ks = np.arange(1, max_integer + 1)

    # Compute unnormalized PMF values
    pmf = -theta ** ks / (ks * np.log(1 - theta))

    # Normalize to sum to 1
    pmf /= pmf.sum()

    # Create scipy discrete distribution
    dist = rv_discrete(name='truncated_logseries', values=(ks, pmf))
    return dist