from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm

import numpy as np

test_size = 300

## POLYNOMIAL KERNEL ##
xs_all_ = np.array( [[float(xi) for xi in x.split()] for x in open("xtrain.txt").readlines()] )
xs_all = preprocessing.normalize(xs_all_, norm='l2')
ys_all = np.array( [int(float(y)) for y in open("ytrain.txt").readlines()] )

xs_train, xs_test, ys_train, ys_test = \
   train_test_split(xs_all, ys_all, test_size=test_size, random_state=245)

clf = svm.NuSVC(.62, degree=3, kernel='poly', gamma=1).fit(xs_train, ys_train)
#clf.score(xs_train, ys_train) # score from trainig
print("Training accuracy for poly kernel: {:.3g}".format(clf.score(xs_test, ys_test)))

xs_to_label_ = np.array( [[float(xi) for xi in x.split()] for x in open("xtest.txt").readlines()] )
xs_to_label = preprocessing.normalize(xs_to_label_, norm='l2')
ys_labeled = clf.predict( xs_to_label )
with open("ytest_poly.txt", "w") as f:
    for l in ys_labeled:
        f.write( "{}\n".format(l) )

## GAUSSIAN KERNEL ##
xs_all = np.array( [[float(xi) for xi in x.split()] for x in open("xtrain.txt").readlines()] )
ys_all = np.array( [int(float(y)) for y in open("ytrain.txt").readlines()] )

xs_train, xs_test, ys_train, ys_test = \
   train_test_split(xs_all, ys_all, test_size=test_size, random_state=245)

clf = svm.NuSVC(.52, kernel='rbf', gamma=1.3e-5).fit(xs_train, ys_train)
#clf.score(xs_train, ys_train) # score from trainig
print("Training accuracy for rbf kernel: {:.3g}".format(clf.score(xs_test, ys_test)))

xs_to_label = np.array( [[float(xi) for xi in x.split()] for x in open("xtest.txt").readlines()] )
ys_labeled = clf.predict( xs_to_label )
with open("ytest_rbf.txt", "w") as f:
    for l in ys_labeled:
        f.write( "{}\n".format(l) )

#nu = 0.32
#gamma = 0.00463
#
#clf = svm.NuSVC(nu, kernel='rbf', gamma=gamma)
#adaclf = AdaBoostClassifier(clf, algorithm='SAMME', learning_rate=.8)
#adaclfmdl = adaclf.fit(xs_train, ys_train)
#adaclfmdl.score(xs_train, ys_train)
#adaclfmdl.score(xs_test, ys_test)
#adaclfmdl.n_estimators
#clfmdl = clf.fit( xs_train, ys_train )
#clfmdl.score(xs_train, ys_train)
#clfmdl.score(xs_test, ys_test)
