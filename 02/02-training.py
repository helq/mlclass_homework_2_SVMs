from sklearn.model_selection import train_test_split
from sklearn import svm

import numpy as np

test_size = 1640

def ssk_from_indices_from_mat(mat):
    def ssk_from_indices( indices_l, indices_r ):
        """Uses a non_local variable, mat"""
        return mat[ [[int(il)] for il in indices_l], [int(ir) for ir in indices_r] ]
    return ssk_from_indices

news_raw = open("./news_subset.txt").read().split("\n")
labels_raw = open("./labels_subset.txt").read().split("\n")
del news_raw[-1] # last one is empty
del labels_raw[-1]

n = len(news_raw)
assert len(news_raw) == len(labels_raw)

news = np.array( news_raw ).reshape( (n, 1) )
labels = np.array( [(-1 if l=='us' else 1) for l in labels_raw] )

xs_all = np.arange(n).reshape( (n,1) )

xs_train, xs_test, ys_train, ys_test = \
    train_test_split(xs_all, labels, test_size=test_size, random_state=123)

name = "news_as_intlists"
lbda = 0.55
mat = np.load( "./{}-ssk_{}.npy".format(name, lbda) )
kernel = ssk_from_indices_from_mat(mat)
clf = svm.NuSVC(.2, kernel=kernel).fit(xs_train, ys_train)
print("Testing accuracy for ssk (with tokenizing): {:.4g}".format(clf.score(xs_test, ys_test)))
