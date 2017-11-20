from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pickle
from collections import namedtuple

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

    #idx = scores.argmax()
    #(i,j) = (int(idx/n_nus), idx%n_nus)
    #maxPoint = ax.scatter( gammas_idx[i,j], nus[i,j], scores[i,j]+.02 , zorder=1)
    surf = ax.plot_surface(data.gammas_idx, data.nus, scores, cmap=cm.coolwarm, linewidth=0, antialiased=False)#, zorder=2)
    return surf

def support_vectors(data, ax):
    svs = np.zeros( data.shape, dtype='int64' )
    for i in range(data.n_gammas):
        for j in range(data.n_nus):
            svs[i,j] = data.cross_results[(data.nus[i,j], data.gammas[i,j])][1]

    surf = ax.plot_surface(data.gammas_idx, data.nus, svs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return surf

def accuracy_std(data, ax):
    devs = np.zeros( data.shape )
    for i in range(data.n_gammas):
        for j in range(data.n_nus):
            devs[i,j] = data.cross_results[(data.nus[i,j], data.gammas[i,j])][0]['test_score'].std()

    surf = ax.plot_surface(data.gammas_idx, data.nus, devs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return surf

def accuracy_training(data, ax):
    scores_train = np.zeros( data.shape )
    for i in range(data.n_gammas):
        for j in range(data.n_nus):
            scores_train[i,j] = data.cross_results[(data.nus[i,j], data.gammas[i,j])][0]['train_score'].mean()

    surf = ax.plot_surface(data.gammas_idx, data.nus, scores_train, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return surf

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

def plotfigure(data, measure, name=None, dist=1):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = create_surfaces[measure](data, ax)

    # Adjusting log scale and ticks for plot
    #ax.set_zlim(.3, 1)
    ax.set_xticks( range(0, data.n_gammas, dist) )
    ax.set_xticklabels("{:.3g}".format(data.gammas_labels[i]) for i in range(0, data.n_gammas, dist))

    ax.set_xlabel(data.axes_keys[1])
    ax.set_ylabel(data.axes_keys[0])
    ax.set_zlabel(measures[measure])

    fig.colorbar(surf, shrink=0.5, aspect=5)
    if name is None:
        plt.show()
    else:
        path = "{}_{}.svg".format(name, measure)

        print('Saving "{}"'.format(path))
        plt.savefig(path, transparent=True)

        print('Postprocessing "{}"'.format(path))
        post_processing(path)

def post_processing(path):
    svg = open(path, 'r').readlines()
    if svg[11] == '  <g id="patch_1">\n' and svg[26] == '  </g>\n':
        print('Removing background image')
        with open(path, "w") as f:
            f.write( ''.join( svg[:11]+svg[27:] ) )
    else:
        print('No post_processing in svg file "{}" could be done :S'.format(path))
        return

    import distutils.spawn

    if distutils.spawn.find_executable('inkscape'):
        from subprocess import Popen
        import os
        FNULL = open(os.devnull, 'w')

        print('Trimming image with Inkscape ...', end=" ", flush=True)
        inkscape = Popen(['inkscape', '--verb=FitCanvasToDrawing', '--verb=FileSave', '--verb=FileQuit', path], stdout=FNULL, stderr=FNULL)
        inkscape.wait()
        print('done')
    else:
        print("Inkscape is not installed, no further post-processing can be done")

plots_params = [
#    {
#        'name': '01/plots/poly-no-preprocessing',
#        'path': '01/cross_validation/cross_validation-poly-no-preprocessing.dat',
#        'measures': ['accuracy', 'support_vectors', 'accuracy_std', 'accuracy_training'],
#        'dist': 1
#    },
#    {
#        'name': '01/plots/poly-scaling',
#        'path': '01/cross_validation/cross_validation-poly-scaling.dat',
#        'measures': ['accuracy', 'support_vectors', 'accuracy_std', 'accuracy_training'],
#        'dist': 1
#    },
    {
        #'name': '01/plots/poly-robust-scaling',
        'path': '01/cross_validation/cross_validation-poly-robust-scaling.dat',
        'measures': ['accuracy', 'support_vectors', 'accuracy_std', 'accuracy_training'],
        'dist': 1
    },
]

if __name__ == '__main__':
    # showing each plot individualy
    for pparams in plots_params:
        data = load_data( pparams['path'] )
        del pparams['path']

        for measure in pparams['measures']:
            params = pparams.copy()
            del params['measures']
            params['measure'] = measure

            plotfigure(data, **params)
