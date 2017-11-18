import pyximport; pyximport.install()
from ssk.string_kernel import ssk, string_kernel

import numpy as np
from math import ceil
from os import path
from multiprocessing import Pool
import os

class GrammMatrixCreator(object):
    def __init__(self, xs, partition_size, name="news", lbda=.8, substring_length=10, into_rows=False):#, processes=10):
        self.xs = xs
        self.name = name
        self.lbda = lbda
        self.news_ssk_dir = "{}-ssk_{}".format(name, lbda)
        self.substring_length = substring_length
        self.into_rows = into_rows
        #self.processes = processes

        self.n = len(xs)
        self.partition_size = partition_size
        self.total_partitions = ceil(self.n/partition_size)

        print("Lambda (for ssk): {}".format(lbda))
        print( "Total datapoints: {}".format(self.n) )
        print( "Text partitions: {}".format(self.total_partitions) )
        print( "Total partitions: {}".format(self.total_partitions**2) )
        print( "Total (partition) computations: {}".format( int(self.total_partitions*(self.total_partitions+1)/2.) ) )

    def isSaved(self):
        if self.into_rows:
            return path.isfile(self.news_ssk_dir+"/mat_00.npy")
        else:
            return path.isfile(self.news_ssk_dir+".npy")

    def computeGramm(self):
        if self.isSaved():
            return

        if not path.isdir(self.news_ssk_dir):
            os.mkdir(self.news_ssk_dir)

        ptsize = self.partition_size
        xs_partitioned = [ self.xs[i*ptsize : (i+1)*ptsize] for i in range(self.total_partitions) ]

        xs_s = []
        for i, xs_l in enumerate(xs_partitioned):
            feats_l = np.array(xs_l).reshape( (len(xs_l), 1) )
            for j, xs_r in enumerate(xs_partitioned):
                if j < i:
                    continue
                if i==j:
                    feats_r = feats_l
                else:
                    feats_r = np.array(xs_r).reshape( (len(xs_r), 1) )
                xs_s.append( (i, j, feats_l, feats_r) )

        #pool = Pool(processes=self.processes) # sadly, it doesn't work :(
        #pool.map( self.process_string_list(), xs_s )

        for inp in xs_s:
            self.process_string_list()(inp)

        print( "Finished computing 'dot product' (kernel application) between all pairs of news" )

    def process_string_list(self):
        def to_ret( input_ ):
            (i, j, feats_l, feats_r) = input_
            name = self.news_ssk_dir+"/mat_{}_{}.npy".format(str(i).rjust(2,"0"), str(j).rjust(2,"0"))

            if not path.isfile(name):
                mat = string_kernel(feats_l, feats_r, self.substring_length, self.lbda)
                np.save(name, mat)

            print(".", end="", flush=True)
        return to_ret

    # TODO: ensure all submatrices are already computed
    def save_ssk_mat(self):
        if self.isSaved():
            return

        if self.into_rows:
            for i in range(self.total_partitions):
                self.__save_ssk_row(i)
        else:
            self.__save_ssk_whole()


    def __save_ssk_row( self, i ):
        name_i = self.news_ssk_dir+"/mat_{}.npy".format(str(i).rjust(2,"0"))

        if path.isfile(name_i):
            return

        ptsize = self.partition_size
        n = self.n

        print( "Saving matrix row {}".format( name_i ) )
        if i < self.total_partitions - 1:
            mat_i = np.zeros( (ptsize, n) )
        else:
            mat_i = np.zeros( (n % ptsize, n) )

        for j in range(self.total_partitions):
            if j < i:
                name_ij = self.news_ssk_dir+"/mat_{}_{}.npy".format(str(j).rjust(2,"0"), str(i).rjust(2,"0"))
                mat_ij = np.load( name_ij ).T
            else:
                name_ij = self.news_ssk_dir+"/mat_{}_{}.npy".format(str(i).rjust(2,"0"), str(j).rjust(2,"0"))
                mat_ij = np.load( name_ij )
            mat_i[:, j*ptsize:(j+1)*ptsize] = mat_ij

        np.save( name_i, mat_i )

    def __save_ssk_whole(self):
        mat = np.zeros( (self.n, self.n) )

        ptsize = self.partition_size

        for i in range(self.total_partitions):
            for j in range(i, self.total_partitions):
                name_ij = self.news_ssk_dir+"/mat_{}_{}.npy".format(str(i).rjust(2,"0"), str(j).rjust(2,"0"))
                mat_ij = np.load( name_ij )

                i_start, i_end = i*ptsize, (i+1)*ptsize
                j_start, j_end = j*ptsize, (j+1)*ptsize
                mat[i_start:i_end, j_start:j_end] = mat_ij

                if i!=j:
                    mat[j_start:j_end, i_start:i_end] = mat_ij.T

        np.save(self.news_ssk_dir+".npy", mat)

def save_string_list( name, list_ ):
    with open(name, "w") as f:
        for s in list_:
            f.write(s)
            f.write("\n")

into_rows = False # saves the matrix as multiple rows, useful if the matrix is too big
partition_size = 1000

if __name__ == '__main__':

    if path.isfile("./news_subset.txt") and path.isfile("./labels_subset.txt"):
        news = open("./news_subset.txt").read().split("\n")
        labels = open("./labels_subset.txt").read().split("\n")
        del news[-1] # last one is empty
        del labels[-1]
        assert len(news) == len(labels)
    else:
        news_all = open("./news.txt").read().split('\n')
        del news_all[-1] # last one is empty

        labels_all = open("./labels.txt").read().split('\n')
        del labels_all[-1]

        news = []
        labels = []
        #indices = []
        for i, l in enumerate(labels_all):
            if l == "entertainment" or l == "us" or l == "health":
                news.append( news_all[i] )
                labels.append( l )
                #indices.append( i )

        save_string_list("news_subset.txt", news)
        save_string_list("labels_subset.txt", labels)

    #gramm_ssk = GrammMatrixCreator(news, partition_size, name="news", lbda=.8, into_rows=into_rows)
    #gramm_ssk.computeGramm()
    #gramm_ssk.save_ssk_mat()

    def computeNSaveSSK(lbda):
        gramm_ssk = GrammMatrixCreator(news, partition_size, name="news", lbda=lbda, into_rows=into_rows)
        gramm_ssk.computeGramm()

    pool = Pool(processes=10)
    pool.map( computeNSaveSSK, [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0] )

    for lbda in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
        gramm_ssk = GrammMatrixCreator(news, partition_size, name="news", lbda=lbda, into_rows=into_rows)
        gramm_ssk.save_ssk_mat()

    print("Preprocessing finished")
