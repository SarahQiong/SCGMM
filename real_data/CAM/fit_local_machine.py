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


def fit_local(X, order):
    gmm = pMLEGMM(n_components=order,
                  covariance_type='full',
                  tol=1 / (X.shape[0] * 1e6),
                  max_iter=20,
                  n_init=10,
                  cov_reg=1 / np.sqrt(X.shape[0]),
                  warm_start=True,
                  random_state=0,
                  verbose=0, verbose_interval=1)
    gmm.fit(X)
    gmm.max_iter = 10000
    gmm.fit(X)
    return gmm


def main(seed, order, machine):
    # load dataset
    PRECL = np.load('PRECL.npy')
    OMEGA = np.load('OMEGA.npy')
    Q = np.load('Q.npy')
    T = np.load('T.npy')
    X = np.concatenate((PRECL, OMEGA, Q, T), 1)

    # load permutation
    idx = np.load(os.path.join('split', 'seed_' + str(seed) + '.npy'))

    # get the subset on the corresponding machine
    nsplit = 128
    subset_length = X.shape[0] // nsplit

    X = X[(machine * subset_length):((machine + 1) * subset_length)]

    # save_folder = 'output_larger_regularization'
    save_folder = 'output'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    save_file = os.path.join(
        save_folder,
        'seed_' + str(seed) + 'machine_' + str(machine) + 'order_' + str(order) + '.pickle')

    gmm = fit_local(X, order)
    f = open(save_file, 'wb')
    pickle.dump(gmm, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split-and-conquer for CAM')
    parser.add_argument('--seed', type=int, default=1, help='index of repetition')
    parser.add_argument('--machine', type=int, default=1, help='index of machine')
    parser.add_argument('--order', type=int, default=4, help='number of local machines')
    

    args = parser.parse_args()
    seed = args.seed
    order = args.order
    machine = args.machine

    start_time = time.time()
    main(seed, order, machine)
    print(time.time() - start_time)
