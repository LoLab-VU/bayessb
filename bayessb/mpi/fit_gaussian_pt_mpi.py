"""
Example template for how to implement parallel tempering algorithm using MPI.

Run with, e.g.::

    mpiexec -np 5 python hello_mpi.py
"""

from mpi4py import MPI
import numpy as np
from bayessb.tests import TwoDGaussianFit
from bayessb import MCMCOpts, MCMC
from pt_mpi import PT_MPI_Master, PT_MPI_Worker

if __name__ == '__main__':
    # The total number of MCMC steps to run
    nsteps = 40000
    # Frequency for proposing swaps
    swap_period = 20
    # Temperature range
    min_temp = 1
    max_temp = 100

    # The communicator to use
    comm = MPI.COMM_WORLD
    # Number of chains/workers in the whole pool
    num_chains = comm.Get_size()
    # The rank of this chain (0 is the master, others are workers)
    rank = comm.Get_rank()

    # Set up the 2-D gaussian model
    means_x = [ 0.1, 0.5, 0.9,
                0.1, 0.5, 0.9,
                0.1, 0.5, 0.9]
    means_y = [0.1, 0.1, 0.1,
               0.5, 0.5, 0.5,
               0.9, 0.9, 0.9]
    sd = 0.01
    tdg = TwoDGaussianFit(means_x, means_y, sd ** 2)

    # Create temperature array based on number of workers (excluding master)
    temps = np.logspace(np.log10(min_temp), np.log10(max_temp), num_chains-1)

    # Create the options
    opts = MCMCOpts()
    opts.model = tdg
    opts.estimate_params = tdg.parameters
    opts.initial_values = [1.001, 1.001]
    opts.nsteps = nsteps
    opts.anneal_length = 0 # necessary so cooling does not occur
    opts.T_init = temps[rank - 1] # Use the temperature for this worker
    opts.use_hessian = False
    opts.seed = 1
    opts.norm_step_size = 0.1
    opts.likelihood_fn = tdg.likelihood

    mcmc = MCMC(opts)
    mcmc.initialize()

    # The master coordinates when swaps occur ---------
    if rank == 0:
        pt = PT_MPI_Master(comm, rank, opts, swap_period, num_chains)
        pt.run()
    # Everyone else runs MCMC steps and swaps when told -----------
    else:
        pt = PT_MPI_Worker(comm, rank, mcmc, swap_period)
        pt.run()



