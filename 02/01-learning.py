#from lib.get_ssk_from_indices import ssk_from_indices

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import svm

import numpy as np
from multiprocessing import Pool
import pickle

#import pyximport; pyximport.install()
#from ssk.string_kernel import string_kernel
#from random import randint

# loading data
news_raw = open("./news_subset.txt").read().split("\n")
labels_raw = open("./labels_subset.txt").read().split("\n")
del news_raw[-1] # last one is empty
del labels_raw[-1]

n = len(news_raw)
assert len(news_raw) == len(labels_raw)

news = np.array( news_raw ).reshape( (n, 1) )
labels = np.array( [(-1 if l=='us' else 1) for l in labels_raw] )

test_size = 1640 # makes possible to partition the set in six almost equal parts,
                 # thus k-fold cross has a similar precision on the error estimation as the final one

## testing the matrix precomputed gram as not faulty
#rnd_idx1 = [randint(0,n-1) for i in range(50)]
#rnd_idx2 = [randint(0,n-1) for i in range(40)]
#assert (
#        (string_kernel(news[rnd_idx1], news[rnd_idx2], 5, .8)
#         - ssk_from_indices( rnd_idx1, rnd_idx2 )) < 1e-5
#       ).all(), \
#  "A test validating the loaded matrix as the string kernel gramm matrix precomputed hasn't passed"

#def strker(il,ir):
#    print("Shape of gramm matrix to create ({},{})".format(len(il), len(ir)))
#    l = np.array([news[int(i),0] for i in il]).reshape( (len(il), 1) )
#    r = np.array([news[int(i),0] for i in ir]).reshape( (len(ir), 1) )
#    return string_kernel(l,r,5,.8)

def ssk_from_indices_from_mat(mat):
    def ssk_from_indices( indices_l, indices_r ):
        """Uses a global variable, mat"""
        return mat[ [[int(il)] for il in indices_l], [int(ir) for ir in indices_r] ]
    return ssk_from_indices

grids_params = [
#    { # BAD, very bad idea! it's too damn slow (several hours)
#        'preprocessing': 'no-preprocessing',
#        'kernel_params': {'kernel': strker, 'max_iter': 1000000, 'verbose': True},
#        'axes': {
#            'nu': np.arange(.02,.8,.02)
#        },
#        'xs_all': np.arange(n).reshape( (n,1) )
#    },
    {
        'preprocessing': 'no-preprocessing',
        'name': "news",
        'kernel_params': {'max_iter': 1400000},#, 'verbose': True},
        'axes': {
            'lambda': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
            'nu': np.arange(.02,.8,.02)
        },
        'xs_all': np.arange(n).reshape( (n,1) )
    },
    {
        'preprocessing': 'tokenized_leximized',
        'name': "news_as_intlists",
        'kernel_params': {'max_iter': 1400000},#, 'verbose': True},
        'axes': {
            'lambda': [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1.],
            'nu': np.arange(.02,.8,.02)
        },
        'xs_all': np.arange(n).reshape( (n,1) )
    },
]

def cross_validate_with_params( params ):
    """Uses global variables: xs_train, ys_train, axes_keys"""

    axes_coord, name, kernel_params = params

    lbda_set = False
    if 'lambda' in kernel_params:
        lbda_set = True
        lbda = kernel_params['lambda']
        del kernel_params['lambda']
        mat = np.load( "./{}-ssk_{}.npy".format(name, lbda) )
        kernel = ssk_from_indices_from_mat(mat)

    clf = svm.NuSVC(kernel=kernel, **kernel_params)
    scores = cross_validate(clf, xs_train, ys_train, cv=5)#, n_jobs=-1)

    clfmdl = clf.fit(xs_train, ys_train)

    if lbda_set:
        kernel_params['lambda'] = lbda

    for axis_k in axes_keys:
        print("{}: {:<9.3g}".format(axis_k, kernel_params[axis_k]), end=" ")

    print("Accuracy: {:.5g} (+/- {:.5g})\tNum support: {:d}"
          .format(scores["test_score"].mean(),
                  scores["test_score"].std() * 2,
                  clfmdl.support_.shape[0]))

    return (scores, clfmdl.support_.shape[0])

if __name__ == '__main__':
    for params in grids_params:
        # partitioning
        xs_train, xs_test, ys_train, ys_test = \
                train_test_split(params['xs_all'], labels, test_size=test_size, random_state=123)

        # Cross-validation
        print("Cross-validation processing")

        axes = params['axes']
        axes_keys = list(axes.keys())
        axes_values = [axes[k] for k in axes_keys]

        grid_size = 1
        axes_sizes = []
        for axis_v in axes.values():
            axis_size = len(axis_v)
            grid_size *= axis_size
            axes_sizes.append( axis_size )

        kernels_params = []
        for idx in range(grid_size):
            kernel_params = params['kernel_params'].copy()
            axes_coord = []
            for i in range(len(axes)-1, -1, -1):
                axis_size = axes_sizes[i]
                axis_idx = idx%axis_size
                idx = int(idx/axis_size)

                axis_value = axes_values[i][axis_idx]

                axes_coord.append(axis_value)
                kernel_params[axes_keys[i]] = axis_value

            axes_coord = tuple(reversed(axes_coord))
            kernels_params.append( (axes_coord, params['name'], kernel_params) )

            #cross_results[axes_coord] = None # initializing value to prevent weird behaivor when running the code in parallel
            #print(axes_coord)
            #print(kernel_params)

        pool = Pool(processes=10)
        grid_results = pool.map( cross_validate_with_params, kernels_params )
        cross_results = { kernels_params[i][0]:grid_results[i] for i in range(grid_size) }

        #grid_results = []
        #for p in kernels_params:
        #    grid_results.append( cross_validate_with_params(p) )

        with open("cross_validation-{}.dat"
                  .format(params['preprocessing']), "wb") as f:
            pickle.dump( tuple(axes_keys), f )
            pickle.dump( cross_results, f )
