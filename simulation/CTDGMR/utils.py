import ot
import numpy as np
from scipy import linalg
import scipy.stats as ss
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture.base import _check_shape
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
import time


def _check_weights(weights):
    # check range
    if (np.less(weights, 0.).any() or np.greater(weights, 1.).any()):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got min value %.5f, max value %.5f" % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))


def log_normal_pdf(x, mean, cov):
    d = x.shape[0]
    # cholesky decomposition of precision matrix
    cov_chol = linalg.cholesky(cov, lower=True)
    prec_chol = linalg.solve_triangular(cov_chol, np.eye(d), lower=True).T
    # log determinant of cholesky matrix
    log_det = np.sum(np.log(np.diag(prec_chol)))
    y = np.dot(x - mean, prec_chol)
    log_prob = np.sum(np.square(y))
    return -.5 * (d * np.log(2 * np.pi) + log_prob) + log_det


def fEij(mean, cov, reduced_mean, reduced_cov):
    return log_normal_pdf(mean, reduced_mean,
                          reduced_cov) - 0.5 * np.trace(linalg.inv(reduced_cov) @ cov)


def compute_precision_cholesky(covariances):
    """Compute the Cholesky decomposition of the precisions"""
    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol[k] = linalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T
    return precisions_chol


def compute_log_det_cholesky(matrix_chol, n_features):
    n_components, _, _ = matrix_chol.shape
    log_det_chol = (np.sum(np.log(matrix_chol.reshape(n_components, -1)[:, ::n_features + 1]), 1))
    return log_det_chol


def compute_resp(X, means, covs):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    precisions_chol = compute_precision_cholesky(covs)
    log_det = compute_log_det_cholesky(precisions_chol, n_features)
    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)
    log_resp = -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
    return log_resp


def log_GMM_pdf(X, means, covs, weights):
    resp = compute_resp(X, means, covs)
    return np.log(np.sum(np.exp(resp) * weights, axis=1))


def GMM_sampler(means, covs, weights, n_samples, random_state=0):
    """Sample from a Gaussian mixture

    Parameters
    ----------
    means : array-like, shape (n, d)
    covs :  array-like, shape (n, d, d)
    weights : array-like, shape (n,)

    Returns
    -------
    # n_sampels of samples from the Gaussian mixture
    """
    if n_samples < 1:
        raise ValueError("Invalid value for 'n_samples': %d . The sampling requires at "
                         "least one sample." % (n_components))
    weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    # check range
    if (any(np.less(weights, 0.)) or any(np.greater(weights, 1.))):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got max value %.5f, min value %.5f" % (np.min(weights), np.max(weights)))
    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    rng = check_random_state(random_state)
    n_samples_comp = rng.multinomial(n_samples, weights)
    X = np.vstack([
        rng.multivariate_normal(mean, cov, int(sample))
        for (mean, cov, sample) in zip(means, covs, n_samples_comp)
    ])

    y = np.concatenate([np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)])

    return (X, y)


"""
Distance functions

Part I: distance between Gaussians
Part II: distance between Gaussian mixtures

"""


def Gaussian_distance(mu1, mu2, Sigma1, Sigma2, which="W2"):
    """Compute distance between Gaussians.

    Parameters
    ----------
    mu1 : array-like, shape (d, )
    mu2 : array-like, shape (d, )
    Sigma1 :  array-like, shape (d, d)
    Sigma2 :  array-like, shape (d, d)
    which : string, "W2" or "KL"


    Returns
    -------
    2-Wasserstein distance between Gaussians.

    """
    if which == "KL":
        d = mu1.shape[0]
        Sigma2_inv = np.linalg.inv(Sigma2)
        # log_det = -(np.log(np.linalg.det(Sigma2_inv)) +
        #             np.log(np.linalg.det(Sigma1)))
        # the above approach for computing the log-determinant has
        # numerical issues when the dimension is high, we therefore
        # first get the eigenvalues and the log determinant is the
        # summation of the log eigenvalues
        log_det = -(np.log(np.linalg.eigvals(Sigma2_inv)).sum() +
                    np.log(np.linalg.eigvals(Sigma1)).sum())

        trace = np.matrix.trace(Sigma2_inv.dot(Sigma1))
        quadratic_term = (mu2 - mu1).T.dot(Sigma2_inv).dot(mu2 - mu1)
        return .5 * (log_det + trace + quadratic_term - d)

    elif which == "W2":
        # 1 dimensional
        if mu1.shape[0] == 1 or mu2.shape[0] == 1:
            W2_squared = (mu1 - mu2)**2 + (np.sqrt(Sigma1) - np.sqrt(Sigma2))**2
            W2_squared = np.asscalar(W2_squared)
        # multi-dimensional
        else:
            sqrt_Sigma1 = linalg.sqrtm(Sigma1)
            Sigma = Sigma1 + Sigma2 - 2 * linalg.sqrtm(sqrt_Sigma1 @ Sigma2 @ sqrt_Sigma1)
            W2_squared = np.linalg.norm(mu1 - mu2)**2 + np.trace(Sigma) + 1e-13
        return np.sqrt(W2_squared)
    else:
        raise ValueError("This ground distance is not implemented!")


def log_normals(diffs, covs):
    """
    log normal density of a matrix X
    evaluated for multiple multivariate normals
    =====
    input:
    diffs: array-like (N, M, d)
    covs: array-like (N,M,d,d)
    """
    n, m, d, _ = covs.shape
    if d == 1:
        precisions_chol = (np.sqrt(1 / covs)).reshape((n * m, d, d))
    else:
        precisions_chol = np.empty((n * m, d, d))
        for k, cov in enumerate(covs.reshape((-1, d, d))):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                raise ValueError("Precision chol is wrong.")
            precisions_chol[k] = linalg.solve_triangular(cov_chol, np.eye(d), lower=True).T
    log_det = (np.sum(np.log(precisions_chol.reshape(n * m, -1)[:, ::d + 1]), 1))
    diffs = diffs.reshape((-1, d))
    y = np.einsum('ij,ijk->ik', diffs, precisions_chol)
    log_prob = np.sum(np.square(y), axis=1)
    probs = -.5 * (d * np.log(2 * np.pi) + log_prob) + log_det
    return probs.reshape((n, m))


def GMM_L2(means, covs, weights, normalized=False):
    """Compute the squared L2 distance between Gaussian mixtures.

    Parameters
    ----------
    means : list of numpy arrays, length 2, (k1,d), (k2,d)
    covs :  list of numpy arrays , length 2, (k1, d, d), (k2, d, d)
    weights: list of numpy arrays 
    Returns
    -------
    Squared L2 distance between Gaussian mixtures.
    """
    w1, w2 = weights[0], weights[1]
    mus1, mus2 = means[0], means[1]
    Sigmas1, Sigmas2 = covs[0], covs[1]
    # normalization of the weights
    w1 /= w1.sum()
    w2 /= w2.sum()

    # M11
    diff = mus1[np.newaxis, :] - mus1[:, np.newaxis]
    covs = Sigmas1[np.newaxis, :] + Sigmas1[:, np.newaxis]
    M11 = np.exp(log_normals(diff, covs))

    # M12
    diff = mus2[np.newaxis, :] - mus1[:, np.newaxis]
    covs = Sigmas2[np.newaxis, :] + Sigmas1[:, np.newaxis]
    M12 = np.exp(log_normals(diff, covs))

    # M22
    diff = mus2[np.newaxis, :] - mus2[:, np.newaxis]
    covs = Sigmas2[np.newaxis, :] + Sigmas2[:, np.newaxis]
    M22 = np.exp(log_normals(diff, covs))

    return w1.T.dot(M11).dot(w1) - 2 * w1.T.dot(M12).dot(w2) + w2.T.dot(M22).dot(w2)


def GMM_CWD(means, covs, weights=None, ground_distance="W2", matrix=False, N=1):
    """Compute the 2 Wasserstein distance between Gaussian mixtures.

    Parameters
    ----------
    means : list of numpy arrays, length 2, (k1,d), (k2,d)
    covs :  list of numpy arrays , length 2, (k1, d, d), (k2, d, d)
    weights: list of numpy arrays 
    Returns
    -------
    Composite Wasserstein distance.
    """

    mus1, mus2 = means[0], means[1]
    Sigmas1, Sigmas2 = covs[0], covs[1]

    k1, k2 = mus1.shape[0], mus2.shape[0]
    cost_matrix = np.zeros((k1, k2))

    w1, w2 = weights[0], weights[1]
    w1 /= w1.sum()
    w2 /= w2.sum()
    _check_weights(w1)
    _check_weights(w2)

    # A lot of redundant work is done here for computing the cost matrix
    # which makes the computation very slow. If the ground distance is
    # KL divergence, we can simplify the computation a lot
    start_time = time.time()
    if ground_distance == "KL":
        # compute the precision matrix of the covariances matrices of the
        # Gaussian components of the second mixture
        n, d, _ = Sigmas2.shape
        if d == 1:
            precisions_chol = np.sqrt(1 / Sigmas2)
        else:
            precisions_chol = np.empty((n, d, d))
            for k, cov in enumerate(Sigmas2):
                try:
                    cov_chol = linalg.cholesky(cov, lower=True)
                except linalg.LinAlgError:
                    raise ValueError("Precision chol is wrong.")
                precisions_chol[k] = linalg.solve_triangular(cov_chol, np.eye(d), lower=True).T
        # the difference term
        for k, (mu, prec_chol) in enumerate(zip(mus2, precisions_chol)):
            y = np.dot(mus1, prec_chol) - np.dot(mu, prec_chol)
            cost_matrix[:, k] = 0.5 * np.sum(np.square(y), axis=1)
        # each column minus the inv_log_det
        inv_log_det = (np.sum(np.log(precisions_chol.reshape(n, -1)[:, ::d + 1]), 1))
        cost_matrix = cost_matrix - inv_log_det
        m, d, _ = Sigmas1.shape
        if d == 1:
            chol = np.sqrt(Sigmas1)
        else:
            chol = np.empty((m, d, d))
            for k, cov in enumerate(Sigmas1):
                try:
                    chol[k] = linalg.cholesky(cov, lower=True)
                except linalg.LinAlgError:
                    raise ValueError("Chol is wrong.")
        log_det = np.sum(np.log(chol.reshape(m, -1)[:, ::d + 1]), 1)
        cost_matrix = (cost_matrix.T - log_det).T

        for i in range(k1):
            for j in range(k2):
                cost_matrix[i, j] += 0.5 * np.trace(precisions_chol[j].dot(
                    precisions_chol[j].T).dot(Sigmas1[i]))
        cost_matrix -= d / 2
    else:
        for i in range(k1):
            for j in range(k2):
                if ground_distance == "W2":
                    cost_matrix[i, j] = Gaussian_distance(mus1[i], mus2[j], Sigmas1[i], Sigmas2[j],
                                                          "W2")**2
                elif ground_distance == "KL":
                    cost_matrix[i, j] = Gaussian_distance(mus1[i], mus2[j], Sigmas1[i], Sigmas2[j],
                                                          "KL")
                elif ground_distance == "WKL":
                    cost_matrix[i, j] = -(np.log(w2[j]) +
                                          N * fEij(mus1[i], Sigmas1[i], mus2[j], Sigmas2[j]))

                elif ground_distance == "W1":
                    cost_matrix[i, j] = np.linalg.norm(mus1[i] - mus2[j]) + np.linalg.norm(
                        linalg.sqrtm(Sigmas1[i]) - linalg.sqrtm(Sigmas2[j]))
                else:
                    raise ValueError("This ground distance is not implemented!")

    if matrix:
        return cost_matrix
    else:
        GMM_Wdistance = ot.emd2(w1, w2, cost_matrix)
        if ground_distance == "W2":
            return np.sqrt(GMM_Wdistance)
        else:
            return GMM_Wdistance


def Gaussian_barycenter(means, covs, weights=None, tol=1e-5, ground_distance="W2"):
    """Compute the Wasserstein or KL barycenter of Gaussian measures.

    Parameters
    ----------
    means : array-like, shape (n, d)
    covs :  array-like, shape (n, d, d)
    weights : array-like, shape (n,)
        The weight in front of the Wasserstein distance.
    ground_distance: string. One of "W2" and "KL"

    Returns
    -------
    mean and covariance of the Gaussian Wasserstein barycenter.

    """
    means = np.copy(means)
    covs = np.copy(covs)
    weights = np.copy(weights)

    m, d = means.shape
    if weights is None:
        weights = np.ones((m, 1)) / m
    else:
        # weight standardization
        weights = weights / weights.sum()
        weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
        _check_shape(weights, (m, ), 'weights')
        # check range
        if (any(np.less(weights, 0.)) or any(np.greater(weights, 1.))):
            raise ValueError("The parameter 'weights' should be in the range "
                             "[0, 1], but got max value %.5f, min value %.5f" %
                             (np.min(weights), np.max(weights)))
        # check normalization
        if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
            raise ValueError("The parameter 'weights' should be normalized, "
                             "but got sum(weights) = %.5f" % np.sum(weights))

    barycenter_means = np.sum(weights.reshape((-1, 1)) * means, axis=0)

    if ground_distance == "KL" or ground_distance == "WKL":
        barycenter_covs = np.sum(covs * weights.reshape((-1, 1, 1)), axis=0)
        diff = means - barycenter_means
        barycenter_covs += np.dot(weights * diff.T, diff)

    elif ground_distance == "W2":
        if d == 1:
            barycenter_covs = np.sum(np.sqrt(covs) * weights.reshape((-1, 1, 1)))**2
        else:
            #Fixed point iteration for Gaussian barycenter
            barycenter_covs = np.zeros((d, d))
            barycenter_covs_next = np.identity(d)
            while np.linalg.norm(barycenter_covs_next - barycenter_covs, 'fro') > tol:
                barycenter_covs = barycenter_covs_next
                sqrt_barycenter_covs = linalg.sqrtm(barycenter_covs)
                barycenter_covs_next = np.zeros((d, d))
                for k in range(m):
                    barycenter_covs_next = barycenter_covs_next + \
                    weights[k] * linalg.sqrtm(sqrt_barycenter_covs@covs[k]@sqrt_barycenter_covs)

    else:
        raise ValueError("This ground_distance %s is no implemented." % ground_distance)
    return barycenter_means, barycenter_covs


def label_predict(weights, means, covs, X):
    resp = compute_resp(X, means, covs)
    # add weight
    resp = np.exp(resp) * weights
    return np.argmax(resp, 1)


def miscls_rate(means, covs, weights, true_means, true_covs, true_weights, true_label, X):
    """
    Inputs:
    weights, means, covs: estimated parameter value
    true_weights, true_means, true_covs: true parameter value
    true_label: the true membership indicator where the sample is from
    X: the sample 
    """
    # step 1: compute the posterior probability to classify X
    predicted_label = label_predict(weights, means, covs, X)
    # step 2: compute distance matrix to align components
    distance = GMM_CWD([true_means, means], [true_covs, covs], [true_weights, weights],
                       ground_distance="KL",
                       matrix=True)
    order1 = np.argmin(distance, 1)
    order2 = np.argmin(distance, 0)

    predicted_label1 = order1[predicted_label]
    predicted_label2 = order2[predicted_label]

    # step 3: compute misclassification rate
    error1 = np.mean(true_label != predicted_label1)
    error2 = np.mean(true_label != predicted_label2)
    # print(order1, order2)
    # print(predicted_label)
    return np.min((error1, error2))
    # return np.mean(predicted_label!=true_label)


def GMM_pairwise_dist(list_of_GMM_means, list_of_GMM_covs, list_of_GMM_weights):
    assert len(list_of_GMM_means) == len(list_of_GMM_covs) & len(list_of_GMM_covs) == len(
        list_of_GMM_weights)
    n = len(list_of_GMM_means)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            dist[i, j] = GMM_L2([list_of_GMM_means[i], list_of_GMM_means[j]],
                                [list_of_GMM_covs[i], list_of_GMM_covs[j]],
                                [list_of_GMM_weights[i], list_of_GMM_weights[j]],
                                normalized=False)
    dist = dist + dist.T
    return dist