import os
import pickle
import argparse
import numpy as np
from CTDGMR.utils import *
from CTDGMR.gmm_reduction import *


def aggregate(seed, X, order=4, nsplit=128):
    subset_means = []
    subset_covs = []
    subset_weights = []

    save_folder = 'output'

    for split in range(nsplit):
        save_file = os.path.join(
            save_folder,
            'seed_' + str(seed) + 'machine_' + str(split) + 'order_' + str(order) + '.pickle')
        with open(save_file, 'rb') as f:
            gmmk = pickle.load(f)
        means, covs, weights = gmmk.means_, gmmk.covariances_, gmmk.weights_

        subset_means.append(means)
        subset_covs.append(covs)
        subset_weights.append(weights)

    obj = np.Inf
    for idx in range(nsplit):
        mean_init = subset_means[idx]
        cov_init = subset_covs[idx]
        weight_init = subset_weights[idx]
        reduced_gmm = GMR_CTD(np.vstack(subset_means),
                              np.vstack(subset_covs),
                              np.concatenate(subset_weights),
                              order,
                              ground_distance="KL",
                              init_method="user",
                              means_init=mean_init,
                              covs_init=cov_init,
                              weights_init=weight_init)
        reduced_gmm.iterative()
        if reduced_gmm._obj() < obj:
            gmr_means, gmr_covs, gmr_weights = reduced_gmm.reduced_means, reduced_gmm.reduced_covs, reduced_gmm.reduced_weights
            obj = reduced_gmm._obj()

    predicted_cluster = label_predict(gmr_weights, gmr_means, gmr_covs, X)

    output = {'gmr': (gmr_means, gmr_covs, gmr_weights), 'cluster': predicted_cluster}
    save_file = os.path.join(save_folder,
                             'gmr_aggregation_case_' + str(seed) + 'order' + str(order) + '.pickle')

    f = open(save_file, 'wb')
    pickle.dump(output, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split-and-conquer for CAM')
    parser.add_argument('--seed', type=int, default=1, help='index of repetition')
    parser.add_argument('--order', type=int, default=4, help='number of local machines')
    parser.add_argument('--bs_sample_size',
                        type=int,
                        default=1000,
                        help='number of Monte Carlo sample for KL averaging')

    args = parser.parse_args()
    seed = args.seed
    order = args.order
    bs_sample_size = args.bs_sample_size
    PRECL = np.load('PRECL.npy')
    OMEGA = np.load('OMEGA.npy')
    Q = np.load('Q.npy')
    T = np.load('T.npy')
    X = np.concatenate((PRECL, OMEGA, Q, T), 1)

    aggregate(seed, X, order)
