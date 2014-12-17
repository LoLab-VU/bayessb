from bayessb import MCMC, MCMCOpts
import numpy as np
import math

class PT_MCMC(object):
    """Implementation of parallel tempering algorithm.

    See Geyer, "Maximum likelihood Markov Chain Monte Carlo", 1991.

    In this algorithm, a series of chains are run at different temperatures in
    parallel. They execute normally (following the Metropolis algorithm) for n
    steps (defined by the parameter ``swap_period``) and then a swap is
    proposed between two chains that neighbor each other in temperature.  If it
    is accepted, each chain adopts the position of the other. Since the swap is
    symmetric, detailed balance and ergodicity is maintained (when one
    considers that the parameter space is now the Cartesian product of the
    parameter spaces for all chains).

    In this implementation, a :py:class:`PT_MCMC` object is used to contain all
    of the chains (as instances of :py:class:`bayessb.MCMC`) and manage their
    execution, performing swaps when appropriate.

    Temperatures are given as minimum and maximum values; the intermediate
    values are interpolated on a log scale. So if ``min_temp`` is 1, and
    ``max_temp`` is 100, with three temperatures, the temperatures used are 1,
    10, 100.

    Parameters
    ----------
    opts: bayessb.MCMCOpts
        Used to initialize all of the :py:class:`bayessb.MCMC` chains in the
        temperature series.
    num_chains : int
        The number of chains/temperatures to run. Too many temperatures will
        make swapping inefficient; too few temperatures will make swaps
        unlikely to be accepted.
    max_temp : number
        The highest temperature in the series.
    min_temp : number
        The lowest temperature in the series. Should usually be 1 (the default).
    swap_period : int
        Number of steps of "regular" (Metropolis) MCMC to perform for each
        chain before proposing a swap.
    """

    def __init__(self, opts, num_chains, max_temp, min_temp=1, swap_period=20):
        self.options = opts
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.swap_period = swap_period

        self.iter = 0
        """Current step iteration (runs to ``nsteps``)."""
        self.chains = []
        """The set of chains in the temperature series."""

        # Calculate the temperature series
        temps = np.logspace(np.log10(min_temp), np.log10(max_temp),
                                           num_chains)
        # Initialize each of the chains
        for i, temp in enumerate(temps):
            chain = MCMC(opts)
            chain.options.T_init = temp
            chain.initialize()
            chain.iter = chain.start_iter
            self.chains.append(chain)

        # Initialize arrays for storing swap info
        num_swaps = self.options.nsteps / swap_period
        self.swap_proposals = np.zeros((num_swaps, 2))
        """Each swap proposal is stored as [i, j] row in this array."""
        self.pi_xi = np.zeros(num_swaps)
        """The posterior of chain i at the position of i."""
        self.pi_xj = np.zeros(num_swaps)
        """The posterior of chain i at the position of j."""
        self.pj_xi = np.zeros(num_swaps)
        """The posterior of chain j at the position of i."""
        self.pj_xj = np.zeros(num_swaps)
        """The posterior of chain j at the position of j."""
        self.delta_test_posteriors = np.zeros(num_swaps)
        """The posterior probability ratio for swap/noswap."""
        self.swap_alphas = np.zeros(num_swaps)
        """The random number used to accept/reject the swap."""
        self.swap_accepts = np.zeros(num_swaps, dtype=bool)
        """Booleans indicating accepted swaps."""
        self.swap_rejects = np.zeros(num_swaps, dtype=bool)
        """Booleans indicating rejected swaps."""
        self.swap_iter = 0
        """Current swap iteration (runs to ``nsteps / swap_period``)"""

    def estimate(self):
        """Parallel tempering MCMC algorithm (see Geyer, 1991)."""
        while self.iter < self.options.nsteps:
            # Check if it's time to propose a swap
            swap_index = None
            if self.iter >= self.swap_period and \
               self.iter % self.swap_period == 0:
                swap_index = self.propose_swap()

            # Perform Metropolis step for each chain
            for i, chain in enumerate(self.chains):
                # If we have swapped (or tried to swap) chains, skip the
                # Metropolis step for the swapped chains for this round
                if swap_index is not None and \
                   (i == swap_index or i == swap_index+1):
                    continue
                else:
                    self.metropolis_step(chain)

            # Call user-callback step function on the first chain in the series
            # to track execution
            if self.chains[0].options.step_fn:
                self.chains[0].options.step_fn(self.chains[0])

            self.iter += 1

    def propose_swap(self):
        """Performs the temperature-swapping step of the PT algorithm.

        Returns
        -------
        If there is an accepted swap, returns the index of the lower of the
        two chains. Returns None if there is no swap accepted.
        """

        # Idea is to pick two neighboring chains to swap, so we
        # randomly pick the one with the lower index
        i = np.random.randint(len(self.chains)-1)
        j = i+1
        # First, we introduce a slight shorthand
        chain_i = self.chains[i]
        chain_j = self.chains[j]
        # We treat the swap as just another way of making a jump proposal.
        chain_i.test_position = chain_j.position
        chain_j.test_position = chain_i.position
        (chain_i.test_posterior, chain_i.test_prior, chain_i.test_likelihood) =\
                    chain_i.calculate_posterior(position=chain_j.position)
        (chain_j.test_posterior, chain_j.test_prior, chain_j.test_likelihood) =\
                    chain_j.calculate_posterior(position=chain_i.position)
        # We determine the "net" change in posterior for the two chains
        # in the swapped position vs. the unswapped position.
        # Since calculations are in logspace, the ratio becomes a difference:
        pi_xi = chain_i.accept_posterior
        pj_xj = chain_j.accept_posterior
        pi_xj = chain_i.test_posterior
        pj_xi = chain_j.test_posterior
        delta_posterior = (pi_xj + pj_xi) - (pi_xi + pj_xj)

        # Decide whether to accept the swap: similar to Metropolis
        # If the swap is an increase in probability, do it.
        if delta_posterior < 0:
            self.accept_swap(i, j)
        # Otherwise, choose a random sample to decide whether to swap:
        else:
            swap_alpha = self.chains[0].random.rand()
            self.swap_alphas[self.swap_iter] = swap_alpha
            # Accept the swap
            if math.e ** -delta_posterior > swap_alpha:
                self.accept_swap(i, j)
            # Reject the swap
            else:
                #print "Reject %d, %d" % (i, j)
                self.swap_rejects[self.swap_iter] = 1
                chain_i.reject_move()
                chain_j.reject_move()

        # Log some interesting variables
        for chain in (chain_i, chain_j):
            chain.positions[chain.iter,:] = chain.position #chain.test_position
            chain.priors[chain.iter] = chain.accept_prior
            chain.likelihoods[chain.iter] = chain.accept_likelihood
            chain.posteriors[chain.iter] = chain.accept_posterior
            chain.delta_test_posteriors[chain.iter] = delta_posterior
            chain.sigmas[chain.iter] = chain.sig_value
            chain.ts[chain.iter] = chain.T

        # Log some more interesting info just about the swaps.
        # Ultimately, it could be possible to extract most of this information
        # from the chains themselves, but this is certainly easier.
        self.swap_proposals[self.swap_iter,:] = (i, j)
        self.pi_xi[self.swap_iter] = pi_xi
        self.pi_xj[self.swap_iter] = pi_xj
        self.pj_xi[self.swap_iter] = pj_xi
        self.pj_xj[self.swap_iter] = pj_xj
        self.delta_test_posteriors[self.swap_iter] = chain.delta_posterior

        # Increment the swap count
        self.swap_iter += 1
        chain_i.iter += 1
        chain_j.iter += 1

        return i

    def accept_swap(self, i, j):
        """Do necessary bookkeeping for accepting the proposed swap."""
        #print "Accept %d, %d" % (i, j)
        self.chains[i].accept_move()
        self.chains[j].accept_move()
        self.swap_accepts[self.swap_iter] = 1

    def metropolis_step(self, chain):
        """Perform a Metropolis update step on the given chain."""
        # Get a new position
        chain.test_position = chain.generate_new_position()
        # Choose test position and calculate posterior there
        (chain.test_posterior, chain.test_prior, chain.test_likelihood)\
                = chain.calculate_posterior(chain.test_position)

        # Decide whether to accept the step
        chain.delta_posterior = chain.test_posterior - chain.accept_posterior
        if chain.delta_posterior < 0:
            chain.accept_move()
        else:
            alpha = chain.random.rand()
            chain.alphas[chain.iter] = alpha;  # log the alpha value
            if math.e ** -chain.delta_posterior > alpha:
                chain.accept_move()
            else:
                chain.reject_move()

        # Log some interesting variables
        chain.positions[chain.iter,:] = chain.position #chain.test_position
        chain.priors[chain.iter] = chain.accept_prior
        chain.likelihoods[chain.iter] = chain.accept_likelihood
        chain.posteriors[chain.iter] = chain.accept_posterior
        chain.delta_test_posteriors[chain.iter] = chain.delta_posterior
        chain.sigmas[chain.iter] = chain.sig_value
        chain.ts[chain.iter] = chain.T

        chain.iter += 1

