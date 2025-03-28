import pymc as pm
import numpy as np
import arviz as az
from matplotlib import pyplot as plt
from scipy.special import binom

n1, n2, n12 = 125, 110, 15
N0 = n1 + n2 - n12
Nmax = 5000

do_model_1 = True

if do_model_1:
# partie 1 
    # méthode 1: simulation MCMC
    with pm.Model() as model:
        N = pm.Uniform("N", N0, Nmax)
        n12_var = pm.HyperGeometric("n12", N, n1, n2, observed=n12)
        trace = pm.sample(5000)

    print(az.summary(trace))
    az.plot_posterior(trace)
    plt.show()


    # méthode 2: calcul direct de la moyenne a posteriori
    def V(N):
        numerator = binom(n1, n12) * binom(N-n1, n2 - n12)
        denominator = binom(N, n2)
        return numerator / denominator

    probas = np.array([V(N) for N in np.arange(N0, Nmax)])

    probas /= probas.sum()

    # sum_N (N x proba(N))
    bayes = (np.arange(N0, Nmax) * probas).sum()
    print(f"Bayes estimator = {bayes}")
else:
    # to be done friday
    pass