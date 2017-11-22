from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pickle
from collections import namedtuple
import os

PlotData = namedtuple('PlotData', ['axes_keys', 'shape', 'n_nus', 'n_gammas', 'nus', 'gammas', 'gammas_labels', 'cross_results', 'gammas_idx'])

def load_data(name_file):
    with open(name_file, "rb") as f:
        axes_keys = pickle.load(f)
        #axes_keys = ('nu', 'gamma')
        cross_results = pickle.load(f)

    nus = set()
    gammas = set()

    for nu, gamma in cross_results.keys():
        nus.add(nu)
        gammas.add(gamma)

    nus_ = sorted(nus)
    gammas_ = sorted(gammas)
    n_nus = len(nus)
    n_gammas = len(gammas)

    nus, gammas = np.meshgrid(nus_, gammas_)

    gammas_idx = np.ones( (1,n_nus) ) * np.arange(n_gammas).reshape( (n_gammas, 1) )

    return PlotData(axes_keys, nus.shape, n_nus, n_gammas, nus, gammas, gammas_, cross_results, gammas_idx)

# creating surface
def accuracy(data, ax):
    scores = np.zeros( data.shape )
    for i in range(data.n_gammas):
        for j in range(data.n_nus):
            scores[i,j] = data.cross_results[(data.nus[i,j], data.gammas[i,j])][0]['test_score'].mean()

    #idx = np.unravel_index( scores.argmax(), scores.shape )
    #maxPoint = ax.scatter( data.gammas_idx[idx], data.nus[idx], scores[idx]+.02 , zorder=1)
    surf = ax.plot_surface(data.gammas_idx, data.nus, scores, cmap=cm.coolwarm, linewidth=0, antialiased=False)#, zorder=2)
    return surf, scores

def support_vectors(data, ax):
    svs = np.zeros( data.shape, dtype='int64' )
    for i in range(data.n_gammas):
        for j in range(data.n_nus):
            svs[i,j] = data.cross_results[(data.nus[i,j], data.gammas[i,j])][1]

    surf = ax.plot_surface(data.gammas_idx, data.nus, svs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return surf, svs

def accuracy_std(data, ax):
    devs = np.zeros( data.shape )
    for i in range(data.n_gammas):
        for j in range(data.n_nus):
            devs[i,j] = data.cross_results[(data.nus[i,j], data.gammas[i,j])][0]['test_score'].std()

    surf = ax.plot_surface(data.gammas_idx, data.nus, devs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return surf, devs

def accuracy_training(data, ax):
    scores_train = np.zeros( data.shape )
    for i in range(data.n_gammas):
        for j in range(data.n_nus):
            scores_train[i,j] = data.cross_results[(data.nus[i,j], data.gammas[i,j])][0]['train_score'].mean()

    surf = ax.plot_surface(data.gammas_idx, data.nus, scores_train, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return surf, scores_train

measures = {
    "accuracy":          "Accuracy",
    "support_vectors":   "# Support Vectors",
    "accuracy_std":      "Accuracy standard dev",
    "accuracy_training": "Accuracy on training"
}

create_surfaces = {
    "accuracy":          accuracy,
    "support_vectors":   support_vectors,
    "accuracy_std":      accuracy_std,
    "accuracy_training": accuracy_training
}

def plot_3d_figure(data, measure, name=None, dist=1, force_save=False):
    #fig = plt.figure(figsize=(9,5), tight_layout=True)
    fig = plt.figure(0, figsize=(7,4), tight_layout=True)
    fig.clf() # cleaning current figure, so we don't use a lot of memory without any reason

    ax = fig.gca(projection='3d')

    surf, scores = create_surfaces[measure](data, ax)

    # Adjusting log scale and ticks for plot
    #ax.set_zlim(.3, 1)
    ax.set_xticks( range(0, data.n_gammas, dist) )
    ax.set_xticklabels("{:.3g}".format(data.gammas_labels[i]) for i in range(0, data.n_gammas, dist))

    ax.set_xlabel(data.axes_keys[1])
    ax.set_ylabel(data.axes_keys[0])
    ax.set_zlabel(measures[measure])

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt_show(name, measure)

    return scores

def plot_2d_figure_accuracy_errorbar(data, gamma_i, name=None, measure="Accuracy", testing_set='test', force_save=False):
    """testing_set can either be 'test' or 'train'"""
    i = gamma_i
    scores = np.zeros( data.n_nus )
    devs = np.zeros( data.n_nus )
    for j in range(data.n_nus):
        scores[j] = data.cross_results[(data.nus[i,j], data.gammas[i,j])][0]['{}_score'.format(testing_set)].mean()
        devs[j]   = data.cross_results[(data.nus[i,j], data.gammas[i,j])][0]['{}_score'.format(testing_set)].std()

    fig = plt.figure(figsize=(7,4), tight_layout=True)
    #fig = plt.figure(0, figsize=(7,4), tight_layout=True)
    #fig.clf() # cleaning current figure, so we don't use a lot of memory without any reason

    ax = fig.gca()
    ax.errorbar(x=data.nus[0], y=scores, yerr=devs)

    # plotting highest value
    j = scores.argmax()
    ax.scatter(data.nus[0,j], scores[j])

    ax.set_xlabel(data.axes_keys[0])
    ax.set_ylabel(measure)
    plt_show(name, '{}-accuracy_errorbar'.format(testing_set), force_save)
    return scores, devs

def plt_show(name=None, measure=None, force_save=False):
    if name is None:
        plt.show()
    else:
        path = "{}_{}.svg".format(name, measure)

        if not os.path.isfile(path) or force_save:
            print('Saving "{}"'.format(path))
            plt.savefig(path, transparent=True)

            print('Postprocessing "{}"'.format(path))
            post_processing(path)
        else:
            print('Plot "{}" is already saved'.format(path))

def post_processing(path):
    svg = open(path, 'r').readlines()
    # Detecting transparent background and removing it (for 3d figures)
    if     svg[11] == '  <g id="patch_1">\n' \
       and svg[26] == '  </g>\n':
        print('Removing background image')
        with open(path, "w") as f:
            f.write( ''.join( svg[:11]+svg[27:] ) )
    # Detecting transparent background and removing it (for 2d figures)
    if     svg[11] == '  <g id="patch_1">\n' \
       and svg[17] == '" style="fill:none;"/>\n' \
       and svg[18] == '  </g>\n':
        print('Removing background image')
        with open(path, "w") as f:
            f.write( ''.join( svg[:11]+svg[19:] ) )
    # No transparent background could be found :S
    else:
        print('Uncomplete post processing of svg image "{}", it may not look as expected'.format(path))

    import distutils.spawn

    if distutils.spawn.find_executable('inkscape'):
        from subprocess import Popen
        import os
        FNULL = open(os.devnull, 'w')

        print('Trimming image with Inkscape ...', end=" ", flush=True)
        to_exec = ['inkscape', '--verb=FitCanvasToDrawing', '--verb=FileSave', '--verb=FileQuit', path]
        print(' {}'.format(to_exec), end=" ", flush=True)
        inkscape = Popen(to_exec, stdout=FNULL, stderr=FNULL)
        inkscape.wait()
        print('done')

        print('Converting image to pdf with Inkscape ...', end=" ", flush=True)
        to_exec = ['inkscape', '--without-gui', '--export-pdf={}.pdf'.format(os.path.splitext(path)[0]), path]
        print(' {}'.format(to_exec), end=" ", flush=True)
        inkscape = Popen(to_exec, stdout=FNULL, stderr=FNULL)
        inkscape.wait()
        print('done')
    else:
        print("Inkscape is not installed, no further post-processing can be done")

plots_params = []
#for name_proc in ['poly-no-preprocessing', 'poly-scaling', 'poly-robust-scaling', 'poly-normalization', 'poly-autoencoder', 'poly-kernelPCA_gamma2.2_poly2']:
#    plots_params.append({
#        'name': '01/plots/{}'.format(name_proc),
#        'path': '01/cross_validation/cross_validation-{}.dat'.format(name_proc),
#        'measures': ['accuracy', 'support_vectors', 'accuracy_std', 'accuracy_training'],
#        'dist': 1,
#        #'force_save': True
#    })
for name_proc in ['rbf-autoencoder']:
#for name_proc in ['rbf-no-preprocessing',  'rbf-scaling',  'rbf-robust-scaling',  'rbf-normalization', 'rbf-autoencoder',  'rbf-kernelPCA_gamma2.2_poly2']:
    plots_params.append({
        #'name': '01/plots/{}'.format(name_proc),
        'path': '01/cross_validation/cross_validation-{}.dat'.format(name_proc),
        #'measures': ['accuracy', 'support_vectors', 'accuracy_std', 'accuracy_training'],
        'measures': ['accuracy'],
        'dist': 1,
        #'force_save': True
    })
#for name_proc in ['no-preprocessing', 'tokenized_leximized']:
#    plots_params.append({
#        'name': '02/plots/{}'.format(name_proc),
#        'path': '02/cross_validation-{}.dat'.format(name_proc),
#        'measures': ['accuracy', 'support_vectors', 'accuracy_std', 'accuracy_training'],
#        'dist': 5
#    })

plots_2d_params = []
for name_proc, i in [('poly-no-preprocessing', 1),
#                     ('poly-scaling', 0),
#                     ('poly-robust-scaling', 0),
                     ('poly-normalization', 2),
#                     ('poly-autoencoder', 0),
                     ('poly-kernelPCA_gamma2.2_poly2', 0),
#                     ('rbf-no-preprocessing', 0),
#                     ('rbf-scaling', 0),
#                     ('rbf-robust-scaling', 0),
#                     ('rbf-normalization', 0),
#                     ('rbf-autoencoder', 0),
#                     ('rbf-kernelPCA_gamma2.2_poly2',0)
                     ]:
    plots_2d_params.append({
        'name': '01/plots/{}'.format(name_proc),
        'path': '01/cross_validation/cross_validation-{}.dat'.format(name_proc),
        'gamma_i': i,
        #'testing_set': 'test',
        #'force_save': True
    })

if __name__ == '__main__':
    # showing each plot individualy
    for pparams in plots_params:
        data = load_data( pparams['path'] )
        path = pparams['path']
        del pparams['path']

        for measure in pparams['measures']:
            params = pparams.copy()
            del params['measures']
            params['measure'] = measure

            scores = plot_3d_figure(data, **params)

            print( 'Crossvalidation file: "{}"'.format( path ) )
            print( " Max value: {}".format(scores.max()) )
            (i,j) = np.unravel_index( scores.argmax(), scores.shape )
            print(" Axis values for max value {}: ({:.3g}, {:.3g})".format( data.axes_keys, data.nus[0,j], data.gammas_labels[i] ) )

    #for pparams in plots_2d_params:
    #    data = load_data( pparams['path'] )
    #    path = pparams['path']
    #    del pparams['path']

    #    plot_2d_figure_accuracy_errorbar(data, **pparams)
