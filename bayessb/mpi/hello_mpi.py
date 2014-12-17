"""
Example template for how to implement parallel tempering algorithm using MPI.

Run with, e.g.::

    mpiexec -np 5 python hello_mpi.py
"""

from mpi4py import MPI
import numpy as np
from StringIO import StringIO

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()
num_cycles = 3

log = StringIO()
log.write("rank = %d --------------------------\n" % rank)

# The master coordinates when swaps occur ---------
if rank == 0:
    # Iterate for a certain number of swap cycles
    for cycle_num in range(num_cycles):
        # Pick two guys to swap
        i = np.random.randint(1, nprocs - 1)
        j = i+1
        # Tell everyone except i and j to proceed
        for target_id in range(1, nprocs):
            if target_id == i or target_id == j:
                log.write("Master tells %d and %d to swap!\n" % (i, j))
                comm.send(i, dest=target_id, tag=1)
            else:
                log.write("Master tells %d to continue!\n" % (target_id))
                comm.send("continue", dest=target_id, tag=1)

    # Tell everyone to stop
    for target_id in range(1, nprocs):
        comm.send("stop", dest=target_id, tag=1)
# Everyone else runs MCMC steps and swaps when told -----------
else:
    iter = 0
    while True:
        # Run five "steps"
        for i in range(5):
            log.write("Rank %d, iter %d\n" % (rank, iter))
            iter += 10 ** (rank - 1)
        # Check if we should proceed
        data = comm.recv(source=0, tag=1)
        if data == "continue":
            log.write("Rank %d continuing\n" % rank)
        elif data == "stop":
            log.write("Rank %d stopping\n" % rank)
            break
        else:
            log.write("Rank %d swapping\n" % rank)
            i = int(data)
            # First, handle the case where we are the lower index, and will
            # manage the swap
            if rank == i:
                i_pos = iter
                # 1. The j chain will need the i position to calculate the j test
                #    posterior
                comm.send(i_pos, dest=i+1, tag=1)
                # 2. From the j chain, we'll need: the j position, the j
                #    accept_posterior, and the j test_posterior
                # FIXME for now, receive j_iter
                j_pos = comm.recv(source=i+1, tag=2)
                # Do the swap
                iter = j_pos
                log.write("Rank %d swapped from pos %d to %d\n" %
                          (rank, i_pos, j_pos))
            # Now, handle the case where we are the higher index, and will
            # participate in (but not manage) the swap
            elif rank == i+1:
                j_pos = iter
                # 1. The j chain will receive i position to calculate the j
                #    test posterior
                i_pos = comm.recv(source=i, tag=1)
                # 2. The j chain sends the test_posterior
                # FIXME for now, send the iter
                comm.send(j_pos, dest=i, tag=2)
                # Do the swap
                iter = i_pos
                log.write("Rank %d swapped from pos %d to %d\n" %
                          (rank, j_pos, i_pos))
            else:
                raise Exception("Invalid data message received.")

with open('hello_log_%d.log' % rank, 'w') as f:
    f.write(log.getvalue())
