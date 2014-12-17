from bayessb import MCMC, MCMCOpts
from bayessb.parallel_tempering import PT_MCMC
import numpy as np
from scipy.stats import norm, beta
from scipy.special import gamma
from matplotlib import pyplot as plt
import math
import collections

# DUMMY MODELS ######

Parameter = collections.namedtuple('Parameter', 'name value')

class GaussianFit():
    """Dummy model for fitting multivariate gaussians."""
    def __init__(self, means, variances):
        if len(means) != len(variances):
            raise Exception("Mismatch between means and variances")
        num_dimensions = len(means)
        self.means = means
        self.variances = variances
        self.parameters = []
        for i in range(num_dimensions):
            #self.norm = norm(loc=means[i], scale=sqrt(variances[i]))
            self.parameters.append(Parameter('p%d' % i, means[i]))

    def likelihood(self, mcmc, position):
        return np.sum(((position - self.means)**2) / (2 * self.variances))
        #return self.norm.pdf(position[0])

class TwoDGaussianFit():
    """Dummy model for fitting a set of two-D gaussians."""
    def __init__(self, means_x, means_y, variance):
        if len(means_x) != len(means_y):
            raise Exception("Mismatch between means and variances.")
        self.means_x = means_x
        self.means_y = means_y
        self.variance = variance
        self.parameters = [Parameter('x', 0), Parameter('y', 0)]

    def likelihood(self, mcmc, position):

        lkl = 0
        for x, y in zip(self.means_x, self.means_y):
            lkl += math.e ** -((position[0] - x)**2 / (2 * self.variance) +
                               (position[1] - y)**2 / (2 * self.variance))
        # If very low values for the likelihood can lead to divide by zero
        # errors when taking the negative log. We catch these and return
        # np.inf instead
        if lkl == 0:
            return np.inf
        else:
            return -np.log10(lkl)

class BetaFit():
    """Dummy model for fitting a beta distribution."""
    def __init__(self, alpha, beta):
        self.parameters = [Parameter('p', 0)]
        self.a = alpha
        self.b = beta
        self.scaling = 10

    def likelihood(self, mcmc, position):
        #beta.pdf(x, a, b) = gamma(a+b)/(gamma(a)*gamma(b)) * x**(a-1) *
        #(1-x)**(b-1),
        if position[0] >= 1 or position[0] <= 0:
            return np.inf

        const = gamma(self.a + self.b) / (gamma(self.a)*gamma(self.b))
        return self.scaling * -np.log10(const * (position ** (self.a - 1)) *
                         ((1 - position)**(self.b - 1)))

# FITTING SCRIPTS #####

def step(mcmc):
    """Useful step function."""
    if mcmc.iter % 1000 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f ' \
              'glob_acc=%-.3f  lkl=%g  prior=%g  post=%g' % \
              (mcmc.iter, mcmc.sig_value, mcmc.T,
               mcmc.acceptance/(mcmc.iter+1.), mcmc.accept_likelihood,
               mcmc.accept_prior, mcmc.accept_posterior)

def fit_gaussian(nsteps):
    num_dimensions = 1
    # Create the dummy model
    means = 10 ** np.random.rand(num_dimensions)
    variances = np.random.rand(num_dimensions)
    g = GaussianFit(means, variances)

    # Create the options
    opts = MCMCOpts()
    opts.model = g
    opts.estimate_params = g.parameters
    opts.initial_values = [1]
    opts.nsteps = nsteps
    opts.anneal_length = nsteps/10
    opts.T_init = 1
    opts.use_hessian = False
    opts.seed = 1
    opts.norm_step_size = 1
    opts.likelihood_fn = g.likelihood
    opts.step_fn = step

    # Create the MCMC object
    mcmc = MCMC(opts)
    mcmc.initialize()
    mcmc.estimate()
    mcmc.prune(nsteps/10, 1)

    plt.ion()
    for i in range(mcmc.num_estimate):
        mean_i = np.mean(mcmc.positions[:,i])
        var_i = np.var(mcmc.positions[:, i])
        print "True mean: %f" % means[i]
        print "Sampled mean: %f" % mean_i
        print "True variance: %f" % variances[i]
        print "Sampled variance: %f" % var_i
        plt.figure()
        (heights, points, lines) = plt.hist(mcmc.positions[:,i], bins=50,
                                        normed=True)
        plt.plot(points, norm.pdf(points, loc=means[i],
             scale=np.sqrt(variances[i])), 'r')

    mean_err = np.zeros(len(mcmc.positions))
    var_err = np.zeros(len(mcmc.positions))
    for i in range(len(mcmc.positions)):
        mean_err[i] = (means[0] - np.mean(mcmc.positions[:i+1,0]))
        var_err[i] = (variances[0] - np.var(mcmc.positions[:i+1,0]))
    plt.figure()
    plt.plot(mean_err)
    plt.plot(var_err)

    return mcmc

def fit_beta(nsteps):
    """Fit a beta distribution by MCMC."""
    num_dimensions = 1
    # Create the dummy model
    b = BetaFit(0.5, 0.5)

    # Create the options
    opts = MCMCOpts()
    opts.model = b
    opts.estimate_params = b.parameters
    opts.initial_values = [1.001]
    opts.nsteps = nsteps
    opts.anneal_length = nsteps/10
    opts.T_init = 100
    opts.use_hessian = False
    opts.seed = 1
    opts.norm_step_size = 0.01
    opts.likelihood_fn = b.likelihood
    opts.step_fn = step

    # Create the MCMC object
    mcmc = MCMC(opts)
    mcmc.initialize()
    mcmc.estimate()
    mcmc.prune(nsteps/10, 1)

    plt.ion()
    for i in range(mcmc.num_estimate):
        plt.figure()
        (heights, points, lines) = plt.hist(mcmc.positions[:,i], bins=100,
                                        normed=True)
        plt.plot(points, beta.pdf(points, b.a, b.b), 'r')
    return mcmc

def fit_beta_by_pt(nsteps):
    """Fit a beta distribution by parallel tempering."""
    num_dimensions = 1
    # Create the dummy model
    b = BetaFit(0.5, 0.5)

    # Create the options
    opts = MCMCOpts()
    opts.model = b
    opts.estimate_params = b.parameters
    opts.initial_values = [10 ** 0.5]
    opts.nsteps = nsteps
    opts.anneal_length = 0
    opts.T_init = 1
    opts.use_hessian = False
    opts.seed = 1
    opts.norm_step_size = 0.5
    opts.likelihood_fn = b.likelihood
    opts.step_fn = step

    # Create the MCMC object
    num_temps = 8
    pt = PT_MCMC(opts, num_temps, 10)
    pt.estimate()

    plt.ion()
    for chain in pt.chains:
        fig = plt.figure()
        chain.prune(nsteps/10, 1)
        (heights, points, lines) = plt.hist(chain.positions, bins=100,
                                            normed=True)
        plt.plot(points, beta.pdf(points, b.a, b.b), 'r')
        plt.ylim((0,10))
        plt.xlim((0, 1))
    return pt

def fit_twod_gaussians(nsteps):
    means_x = [ 0.1, 0.5, 0.9,
                0.1, 0.5, 0.9,
                0.1, 0.5, 0.9]
    means_y = [0.1, 0.1, 0.1,
               0.5, 0.5, 0.5,
               0.9, 0.9, 0.9]
    sd = 0.01
    tdg = TwoDGaussianFit(means_x, means_y, sd ** 2)

    # Create the options
    opts = MCMCOpts()
    opts.model = tdg
    opts.estimate_params = tdg.parameters
    opts.initial_values = [1.001, 1.001]
    opts.nsteps = nsteps
    opts.anneal_length = nsteps/10
    opts.T_init = 1
    opts.use_hessian = False
    opts.seed = 1
    opts.norm_step_size = 0.1
    opts.likelihood_fn = tdg.likelihood
    opts.step_fn = step

    mcmc = MCMC(opts)
    mcmc.initialize()
    mcmc.estimate()

    plt.ion()
    fig = plt.figure()
    mcmc.prune(0, 20)
    plt.scatter(mcmc.positions[:,0], mcmc.positions[:,1])
    ax = fig.gca()
    for x, y in zip(means_x, means_y):
        circ = plt.Circle((x, y), radius=2*sd, color='r', fill=False)
        ax.add_patch(circ)
    plt.xlim((-0.5, 1.5))
    plt.ylim((-0.5, 1.5))
    plt.show()

    return mcmc

def fit_twod_gaussians_by_pt(nsteps):
    means_x = [ 0.1, 0.5, 0.9,
                0.1, 0.5, 0.9,
                0.1, 0.5, 0.9]
    means_y = [0.1, 0.1, 0.1,
               0.5, 0.5, 0.5,
               0.9, 0.9, 0.9]
    sd = 0.01
    tdg = TwoDGaussianFit(means_x, means_y, sd ** 2)

    # Create the options
    opts = MCMCOpts()
    opts.model = tdg
    opts.estimate_params = tdg.parameters
    opts.initial_values = [1.001, 1.001]
    opts.nsteps = nsteps
    opts.anneal_length = 0 # necessary so cooling does not occur
    opts.T_init = 1
    opts.use_hessian = False
    opts.seed = 1
    opts.norm_step_size = 0.1
    opts.likelihood_fn = tdg.likelihood
    opts.step_fn = step

    # Create the PT object
    num_temps = 8
    pt = PT_MCMC(opts, num_temps, 100)
    pt.estimate()

    plt.ion()
    for chain in pt.chains:
        fig = plt.figure()
        chain.prune(nsteps/10, 1)
        plt.scatter(chain.positions[:,0], chain.positions[:,1])
        ax = fig.gca()
        for x, y in zip(means_x, means_y):
            circ = plt.Circle((x, y), radius=2*sd, color='r', fill=False)
            ax.add_patch(circ)
        plt.xlim((-0.5, 1.5))
        plt.ylim((-0.5, 1.5))
        plt.title('Temp = %.2f' % chain.options.T_init)
        plt.show()

    return pt

# TESTS #######

def test_GaussianFit_class():
    """Smoke test to see if the GaussianFit class can be instantiated."""
    means = np.random.rand(5)
    variances = np.random.rand(5)
    g = GaussianFit(means, variances)
    # check that the model has parameters and each param has value attribute
    for p in g.parameters:
        a = p.value
    assert True

def test_GaussianFit_likelihood():
    """Check if the likelihood method in the GaussianFit class runs."""
    num_dimensions = 5
    means = np.random.rand(num_dimensions)
    variances = np.random.rand(num_dimensions)
    g = GaussianFit(means, variances)
    assert True

def test_fit_gaussian():
    mcmc = fit_gaussian(1000)

def test_fit_beta():
    mcmc = fit_beta(1000)

def test_fit_beta_by_pt():
    mcmc = fit_beta_by_pt(1000)

def test_fit_twod_gaussians():
    mcmc = fit_twod_gaussians(1000)

def test_fit_twod_gaussians_by_pt():
    pt = fit_twod_gaussians_by_pt(1000)


