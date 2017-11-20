from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split

import numpy as np
import pickle
from multiprocessing import Pool

# Trying several preprocessing techniques before cross validation

xs_all_ = np.array( [[float(xi) for xi in x.split()] for x in open("xtrain.txt").readlines()] )
ys_all = np.array( [int(float(y)) for y in open("ytrain.txt").readlines()] )

test_size = 300

xs_train, xs_test = train_test_split(xs_all_, test_size=test_size, random_state=123)

def tryKPCA(pms):
    g, d = pms
    kpca = KernelPCA(kernel="poly", gamma=g, degree=d, fit_inverse_transform=True, max_iter=1000000)
    xs_kpca = kpca.fit_transform(xs_train)
    xs_kpcainv = kpca.inverse_transform( xs_kpca )
    error_pca = xs_train - xs_kpcainv
    return (error_pca.mean(), error_pca.std(), xs_kpca.shape[1])

pool = Pool(processes=6)
feats_kpca = np.zeros( (6, 20) )
error_mean = np.zeros( (6, 20) )
error_std  = np.zeros( (6, 20) )

gammas = np.zeros( (20) )
g = 1.5e-9
for i in range(40):
    gammas[i] = g

    results = pool.map( tryKPCA, [(g, d) for d in range(1,7)] )
    for d in range(6):
        error_mean[d, i] = results[d][0]
        error_std [d, i] = results[d][1]
        feats_kpca[d, i] = results[d][2]

    print("tried with gamma {:.01e}".format(g))

    g *= 2.2

np.savez( "kpca_grid_search.npz", gammas=gammas, feats_kpca=feats_kpca, error_mean=error_mean, error_std=error_std )
