import os
import time
import argparse
import numpy as np
import _pickle as pickle
from scipy import optimize
from scipy.special import logit, expit
from scipy.stats import wishart
from GMMpMLE import pMLEGMM
from CTDGMR.utils import *
from CTDGMR.gmm_reduction import *
import pyreadr


def GMM_estimate_comparison(inputs):
    random_state = inputs[0]
    sample_size = inputs[1]
    true_means = inputs[2]
    true_covs = inputs[3]
    true_weights = inputs[4]
    save_folder = inputs[5]
    equal_ss = inputs[6]

    # compute true precision matrix
    n_components, n_feature, _ = true_covs.shape
    true_precisions = np.empty((n_components, n_feature, n_feature))
    for k, cov in enumerate(true_covs):
        true_precisions[k, :, :] = np.linalg.inv(cov)

    # Sample from true mixture
    GMM_sample, true_label = GMM_sampler(true_means, true_covs, true_weights, sample_size,
                                         random_state)

    ll_true = log_GMM_pdf(GMM_sample, true_means, true_covs, true_weights).mean()

    # global pMLE
    gmm = pMLEGMM(n_components=n_components,
                  cov_reg=1. / np.sqrt(sample_size),
                  covariance_type="full",
                  max_iter=10000,
                  n_init=1,
                  tol=1e-6,
                  weights_init=true_weights,
                  means_init=true_means,
                  precisions_init=true_precisions,
                  random_state=0,
                  verbose=0,
                  verbose_interval=1)

    start_time = time.time()
    gmm.fit(GMM_sample)
    global_time = time.time() - start_time
    global_iter = gmm.n_iter_
    global_converg = gmm.converged_

    pmle_mean, pmle_cov, pmle_weights = gmm.means_, gmm.covariances_, gmm.weights_
    global2true_W1 = GMM_CWD([pmle_mean, true_means], [pmle_cov, true_covs],
                             [pmle_weights, true_weights], "W1")

    global_ll = gmm.score(GMM_sample)
    global_miscls = miscls_rate(pmle_mean, pmle_cov, pmle_weights, true_means, true_covs,
                                true_weights, true_label, GMM_sample)

    # split and combine
    for nsplit in [4, 16, 64]:
        subset_means = [np.empty((n_components, n_feature))] * nsplit
        subset_covs = [np.empty((n_components, n_feature, n_feature))] * nsplit
        subset_weights = [np.empty((n_components, ))] * nsplit

        subsets2true_W1 = [None] * nsplit
        subsets2true_L2 = [None] * nsplit
        subset_ll = [None] * nsplit

        local_time = [None] * nsplit
        local_iter = [None] * nsplit
        local_converg = [None] * nsplit
        subset_miscls = [None] * nsplit

        # random divide dataset into folds
        np.random.seed(1)
        index = np.random.permutation(sample_size)
        GMM_sample = GMM_sample[index]
        true_label = true_label[index]

        if equal_ss:
            # equal sample size on local machines
            subset_length = sample_size // nsplit
        else:
            # unequal sample size
            subset_ratio = np.arange(nsplit) * 2 + 1
            subset_ratio = subset_ratio / subset_ratio.sum()
            subset_length = subset_ratio * sample_size
            subset_length = subset_length.astype('int')
            subset_length = np.insert(subset_length, 0, 0)
            subset_length = np.cumsum(subset_length)

        # subsampling
        subsampling_sample = []

        for split in range(nsplit):
            if equal_ss:
                subset = GMM_sample[(split * subset_length):((split + 1) * subset_length)]
                # subsampling approach
                subsampling_sample.append(subset[np.random.choice(subset.shape[0], 1000)])
            else:
                subset = GMM_sample[subset_length[split]:subset_length[split + 1]]
                # print(subset.shape[0])
                # subsampling approach
                subsampling_sample.append(subset[np.random.choice(
                    subset.shape[0], int(subset_ratio[split] * 1000 * nsplit))])

            # Local pMLE
            gmmk = pMLEGMM(
                n_components=n_components,
                cov_reg=1. / np.sqrt(subset.shape[0]),
                covariance_type="full",
                max_iter=10000,
                n_init=1,
                # tol=1 / (subset.shape[0] * 1e6),
                tol=1e-6,
                weights_init=true_weights,
                means_init=true_means,
                precisions_init=true_precisions,
                random_state=0,
                verbose=0,
                verbose_interval=1)

            start_time = time.time()
            gmmk.fit(subset)
            local_timek = time.time() - start_time
            local_iterk = gmmk.n_iter_
            local_convergk = gmmk.converged_
            subset_pmle_mean, subset_pmle_cov, subset_pmle_weights = gmmk.means_, gmmk.covariances_, gmmk.weights_

            subset_means[split] = subset_pmle_mean
            subset_covs[split] = subset_pmle_cov
            subset_weights[split] = subset_pmle_weights

            subsets2true_W1[split] = GMM_CWD([subset_pmle_mean, true_means],
                                             [subset_pmle_cov, true_covs],
                                             [subset_pmle_weights, true_weights], "W1")

            subset_ll[split] = gmmk.score(GMM_sample)

            local_time[split] = local_timek
            local_iter[split] = local_iterk
            local_converg[split] = local_convergk
            subset_miscls[split] = miscls_rate(subset_pmle_mean, subset_pmle_cov,
                                               subset_pmle_weights, true_means, true_covs,
                                               true_weights, true_label, GMM_sample)

        #################
        # pool estimator
        #################

        pool2true_W1 = GMM_CWD([np.concatenate(subset_means), true_means],
                               [np.concatenate(subset_covs), true_covs],
                               [np.concatenate(subset_weights) / nsplit, true_weights], "W1")

        pool_ll = log_GMM_pdf(GMM_sample, np.concatenate(subset_means), np.concatenate(subset_covs),
                              np.concatenate(subset_weights) / nsplit).mean()

        #################
        # Return median estimator
        #################

        dist = GMM_pairwise_dist(subset_means, subset_covs, subset_weights)
        which_GMM = np.argmin(dist.sum(0))
        """
        Aggregate local estimators by different approaches

        1. KL-averaging
        2. GMR
        """

        ###############
        # Estimator 1
        ###############
        # Aggregatioin by KL averaging
        start_time = time.time()
        new_GMM_sample = []

        for split in range(nsplit):
            if equal_ss:
                local_sample_size = 1000
            else:
                local_sample_size = int(subset_ratio[split] * 1000 * nsplit)

            subset_GMM_sample, _ = GMM_sampler(subset_means[split], subset_covs[split],
                                               subset_weights[split], local_sample_size)
            new_GMM_sample.append(subset_GMM_sample)

        new_GMM_sample = np.vstack(new_GMM_sample)

        gmm_kl = pMLEGMM(
            n_components=n_components,
            cov_reg=1. / np.sqrt(new_GMM_sample.shape[0]),
            covariance_type="full",
            max_iter=10000,
            n_init=1,
            # tol=1 / (new_GMM_sample.shape[0] * 1e6),
            tol=1e-6,
            weights_init=true_weights,
            means_init=true_means,
            precisions_init=true_precisions,
            random_state=0,
            verbose=0,
            verbose_interval=1)
        start_time = time.time()
        gmm_kl.fit(new_GMM_sample)
        kl_time = time.time() - start_time
        kl_iter = gmm_kl.n_iter_
        kl_converg = gmm_kl.converged_
        kl_means, kl_covs, kl_weights = gmm_kl.means_, gmm_kl.covariances_, gmm_kl.weights_
        kl2true_W1 = GMM_CWD([kl_means, true_means], [kl_covs, true_covs],
                             [kl_weights, true_weights], "W1")
        kl_ll = gmm_kl.score(GMM_sample)
        kl_miscls = miscls_rate(kl_means, kl_covs, kl_weights, true_means, true_covs, true_weights,
                                true_label, GMM_sample)

        ###############
        # Estimator 2
        ###############
        # Aggregatioin by reduction
        start_time = time.time()
        if equal_ss:
            reduced_gmm = GMR_CTD(np.concatenate(subset_means),
                                  np.concatenate(subset_covs),
                                  np.concatenate(subset_weights),
                                  n_components,
                                  ground_distance="KL",
                                  init_method="user",
                                  means_init=true_means,
                                  covs_init=true_covs,
                                  weights_init=true_weights)
        else:
            reduced_gmm = GMR_CTD(
                np.concatenate(subset_means),
                np.concatenate(subset_covs),
                np.concatenate(subset_weights) * np.repeat(subset_ratio, n_components),
                n_components,
                ground_distance="KL",
                init_method="user",
                means_init=mean_init,
                covs_init=cov_init,
                weights_init=weight_init)
        reduced_gmm.iterative()
        gmr_elapsed_time = time.time() - start_time
        gmr_iter = reduced_gmm.n_iter_
        gmr_means, gmr_covs, gmr_weights = reduced_gmm.reduced_means, reduced_gmm.reduced_covs, reduced_gmm.reduced_weights
        gmr2true_W1 = GMM_CWD([gmr_means, true_means], [gmr_covs, true_covs],
                              [gmr_weights, true_weights], "W1")
        gmr_ll = log_GMM_pdf(GMM_sample, gmr_means, gmr_covs, gmr_weights).mean()
        gmr_miscls = miscls_rate(gmr_means, gmr_covs, gmr_weights, true_means, true_covs,
                                 true_weights, true_label, GMM_sample)

        output_data = {
            "ll_true": ll_true,
            "global2true_W1": global2true_W1,
            "global_ll": global_ll,
            "global_miscls": global_miscls,
            "subset2true_W1": subsets2true_W1,
            "subset_ll": subset_ll,
            "subset_miscls": subset_miscls,
            "kl2true_W1": kl2true_W1,
            "kl_ll": kl_ll,
            "kl_miscls": kl_miscls,
            "gmr2true_W1": gmr2true_W1,
            "gmr_ll": gmr_ll,
            "gmr_miscls": gmr_miscls,
            "median2true_W1": subsets2true_W1[which_GMM],
            "median_ll": subset_ll[which_GMM],
            "median_miscls": subset_miscls[which_GMM],
            "local_time": local_time,
            "local_iter": local_iter,
            "local_converg": local_converg,
            "pool2true_W1": pool2true_W1,
            "pool_ll": pool_ll,
            "global_time": global_time,
            "global_iter": global_iter,
            "global_converg": global_converg,
            "kl_time": kl_time,
            "kl_iter": kl_iter,
            "kl_converg": kl_converg,
            "global": (pmle_mean, pmle_cov, pmle_weights),
            "subset": (subset_means, subset_covs, subset_weights),
            "kl": (kl_means, kl_covs, kl_weights),
            "gmr": (gmr_means, gmr_covs, gmr_weights),
        }
        if equal_ss:
            save_file = os.path.join(
                save_folder,
                'case_' + str(random_state) + '_nsplit_' + str(nsplit) + '_ncomponents_' +
                str(n_components) + '_samplesize_' + str(sample_size) + '_equal_ss.pickle')
        else:
            save_file = os.path.join(
                save_folder,
                'case_' + str(random_state) + '_nsplit_' + str(nsplit) + '_ncomponents_' +
                str(n_components) + '_samplesize_' + str(sample_size) + '_unequal_ss.pickle')
        f = open(save_file, 'wb')
        pickle.dump(output_data, f)
        f.close()


def main(seed, sample_size, equal_ss):
    # fix the number of components to be 5
    num_components = 5
    dimension = 50
    overlap = 0.05

    base_dir = './generated_pop'
    result = pyreadr.read_r(
        os.path.join(
            base_dir, 'weights_order_' + str(num_components) + 'dimension_' + str(dimension) +
            'maxomega_' + str(overlap) + 'seed_' + str(seed) + '.Rds'))
    # also works for .RData
    true_weights = np.array(result[None]).reshape(-1, )

    result = pyreadr.read_r(
        os.path.join(
            base_dir, 'means_order_' + str(num_components) + 'dimension_' + str(dimension) +
            'maxomega_' + str(overlap) + 'seed_' + str(seed) + '.Rds'))
    true_means = np.array(result[None]).reshape((-1, num_components)).T

    result = pyreadr.read_r(
        os.path.join(
            base_dir, 'covs_order_' + str(num_components) + 'dimension_' + str(dimension) +
            'maxomega_' + str(overlap) + 'seed_' + str(seed) + '.Rds'))
    true_covs = np.array(result[None]).reshape((-1, dimension, dimension))

    params = [seed, sample_size, true_means, true_covs, true_weights, save_folder, equal_ss]
    GMM_estimate_comparison(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset split GMM estimator comparison')
    parser.add_argument('--seed', type=int, default=1, help='index of repetition')
    parser.add_argument('--ss', type=int, default=65536, help='Total sample size from a GMM')
    parser.add_argument('--equal_ss',
                        action='store_true',
                        help='Equal sample size on local machine')

    args = parser.parse_args()
    sample_size = int(args.ss)
    seed = args.seed
    equal_ss = args.equal_ss
    # print(args)

    save_folder = './output/save_data/effect_of_M_and_N_equal_overlap_0.5'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    main(seed, sample_size, equal_ss)
