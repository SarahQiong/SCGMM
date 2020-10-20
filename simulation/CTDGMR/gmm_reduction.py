import ot
import time
import warnings
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from .utils import GMM_CWD, GMM_sampler, Gaussian_barycenter

"""
Minimum composite Wasserstein distance for GMR

"""


def entropy(log_ot_plan):
    """
    The entropy of a coupling matrix
    """

    return 1 - np.sum(np.exp(log_ot_plan) * log_ot_plan)


class GMR_CTD:
    """Find a GMM with n_components that is closest
    to a GMM parameterized by means, covs, weights in
    the composite transportation distance sense.

    Parameters
    ----------
    reg: strength of entropic regularization

    Returns
    -------
    weights and support points of reduced GMM.
    """
    def __init__(self,
                 means,
                 covs,
                 weights,
                 n,
                 n_pseudo=1,
                 init_method="kmeans",
                 tol=1e-5,
                 max_iter=100,
                 ground_distance="W2",
                 reg=0,
                 means_init=None,
                 covs_init=None,
                 weights_init=None,
                 random_state=0):

        self.means = means
        self.covs = covs
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.origin_n = self.weights.shape[0]
        self.new_n = n
        self.n_pseudo = n_pseudo
        self.random_state = random_state
        self.ground_distance = ground_distance
        self.converged_ = False
        if reg >= 0:
            self.reg = reg
        else:
            raise ValueError("The regularization term should be non-negative.")
        self.init_method = init_method
        self.means_init = np.copy(means_init)
        self.covs_init = np.copy(covs_init)
        self.weights_init = np.copy(weights_init)

    def _initialize_parameter(self):
        """Initializatin of the clustering barycenter"""
        if self.init_method == "kmeans":
            total_sample_size = 10000
            X = GMM_sampler(self.means, self.covs, self.weights,
                            total_sample_size, self.random_state)[0]
            kmeans = KMeans(n_clusters=self.new_n,
                            n_init=1,
                            random_state=self.random_state).fit(X)
            self.reduced_means = kmeans.cluster_centers_
            self.reduced_covs = np.tile(np.mean(self.covs, 0),
                                        (self.new_n, 1, 1))
            self.reduced_weights = np.array([
                np.sum(kmeans.labels_ == i) / total_sample_size
                for i in range(self.new_n)
            ])
        elif self.init_method == "user":
            self.reduced_means = self.means_init
            self.reduced_covs = self.covs_init
            self.reduced_weights = self.weights_init
        else:
            self.reduced_means, self.reduced_covs, self.reduced_weights = GMR_greedy(
                self.means, self.covs, self.weights, self.new_n,
                self.init_method)

        self.cost_matrix = GMM_CWD([self.means, self.reduced_means],
                                   [self.covs, self.reduced_covs],
                                   [self.weights, self.reduced_weights],
                                   ground_distance=self.ground_distance,
                                   matrix=True,
                                   N=self.n_pseudo)
       
    def _obj(self):
        if self.reg == 0:
            return np.sum(self.cost_matrix * self.ot_plan)
        elif self.reg > 0:
            return np.sum(self.cost_matrix *
                          self.ot_plan) - self.reg * entropy(self.log_ot_plan)

    def _weight_update(self):
        if self.reg == 0:
            self.clustering_matrix = (self.cost_matrix.T == np.min(
                self.cost_matrix, 1)).T
            self.ot_plan = self.clustering_matrix * self.weights.reshape(
                (-1, 1))
            self.reduced_weights = self.ot_plan.sum(axis=0)
        elif self.reg > 0:
            lognum = -self.cost_matrix / self.reg
            logtemp = (lognum.T - logsumexp(lognum, axis=1)).T
            self.log_ot_plan = (logtemp.T + np.log(self.weights)).T
            self.ot_plan = np.exp(self.log_ot_plan)
            self.reduced_weights = self.ot_plan.sum(axis=0)
        return self._obj()

    def _support_update(self):
        for i in range(self.new_n):
            self.reduced_means[i], self.reduced_covs[i] = Gaussian_barycenter(
                self.means,
                self.covs,
                self.ot_plan[:, i],
                ground_distance=self.ground_distance)
        self.cost_matrix = GMM_CWD([self.means, self.reduced_means],
                                   [self.covs, self.reduced_covs],
                                   [self.weights, self.reduced_weights],
                                   ground_distance=self.ground_distance,
                                   matrix=True,
                                   N=self.n_pseudo)
        return self._obj()

    def iterative(self):
        self._initialize_parameter()
        obj = np.Inf
        proc_time = time.time()
        for n_iter in range(1, self.max_iter + 1):
            prev_obj = obj
            obj1 = self._weight_update()
            if min(self.ot_plan.sum(axis=0)) == 0:
                self.new_n -= 1
                self.iterative()
            else:
                obj2 = self._support_update()
                change = obj2 - obj1
                obj = obj2
                if abs(change) < self.tol:
                    self.converged_ = True
                    self.obj = obj
                    self.n_iter_ = n_iter
                    break
                if obj2 > obj1:
                    raise ValueError(
                        "Warning: The objective function is increasing!")
        self.time = time.time() - proc_time
        if not self.converged_:
            print('Algorithm did not converge. '
                  'Try different init parameters, '
                  'or increase max_iter, tol ')
