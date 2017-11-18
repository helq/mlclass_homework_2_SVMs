from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pickle

#measure = "Accuracy"
#measure = "# Support Vectors"
#measure = "Accuracy standard dev"
measure = "Accuracy on training"

dist = 1

with open("./cross_validation-poly-kernelPCA_gamma1_poly1.dat", "rb") as f:
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
fig = plt.figure()
ax = fig.gca(projection='3d')

# creating surface
if measure == "Accuracy":
    scores = np.zeros( nus.shape )
    for i in range(n_gammas):
        for j in range(n_nus):
            scores[i,j] = cross_results[(nus[i,j], gammas[i,j])][0]['test_score'].mean()

    #idx = scores.argmax()
    #(i,j) = (int(idx/n_nus), idx%n_nus)
    #maxPoint = ax.scatter( gammas_idx[i,j], nus[i,j], scores[i,j]+.02 , zorder=1)
    surf = ax.plot_surface(gammas_idx, nus, scores, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=2)

elif measure == "# Support Vectors":
    svs = np.zeros( nus.shape, dtype='int64' )
    for i in range(n_gammas):
        for j in range(n_nus):
            svs[i,j] = cross_results[(nus[i,j], gammas[i,j])][1]

    surf = ax.plot_surface(gammas_idx, nus, svs, cmap=cm.coolwarm, linewidth=0, antialiased=False)

elif measure == "Accuracy standard dev":
    devs = np.zeros( nus.shape )
    for i in range(n_gammas):
        for j in range(n_nus):
            devs[i,j] = cross_results[(nus[i,j], gammas[i,j])][0]['test_score'].std()

    surf = ax.plot_surface(gammas_idx, nus, devs, cmap=cm.coolwarm, linewidth=0, antialiased=False)

elif measure == "Accuracy on training":
    scores_train = np.zeros( nus.shape )
    for i in range(n_gammas):
        for j in range(n_nus):
            scores_train[i,j] = cross_results[(nus[i,j], gammas[i,j])][0]['train_score'].mean()

    surf = ax.plot_surface(gammas_idx, nus, scores_train, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Adjusting log scale and ticks for plot
#ax.set_zlim(.3, 1)
ax.set_xticks(range(0,n_gammas,dist))
ax.set_xticklabels("{:.3g}".format(gammas_[i]) for i in range(0,n_gammas,dist))

ax.set_xlabel(axes_keys[1])
ax.set_ylabel(axes_keys[0])
ax.set_zlabel(measure)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
