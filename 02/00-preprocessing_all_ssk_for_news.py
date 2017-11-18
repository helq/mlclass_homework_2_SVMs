import pyximport; pyximport.install()
from ssk.string_kernel import ssk, string_kernel

import numpy as np
from math import ceil
from sys import stdout
from os import path
from multiprocessing import Pool
import os

# Running with a single value of lambda (eg, 0.8)
#sed -e "s/.8/$i/" 00-preprocessing_all_ssk_for_news.py | python -
# How to make many runs
#for i in {.1,.2.,.3,.4,.5,.6,.7,.8,.9,1.0}; do sed -e "s/LAMBDA/$i/" 00-preprocessing_all_ssk_for_news.py | python -; done

lbda = LAMBDA
print("Lambda (for ssk): {}".format(lbda))
news_ssk_dir = "news-ssk_{}".format(lbda)

def process_string_list( input_ ):
    (i, j, feats_l, feats_r) = input_
    name = news_ssk_dir+"/mat_{}_{}.npy".format(str(i).rjust(2,"0"), str(j).rjust(2,"0"))

    #print( "Calculating {}".format( name ) )

    if not path.isfile(name):
        mat = string_kernel(feats_l, feats_r, 10, .8)

        np.save(name, mat)

    stdout.write(".")
    stdout.flush()

def create_ssk_rows( input_ ):
    i, total_partitions, n, partition_size = input_
    name_i = news_ssk_dir+"/mat_{}.npy".format(str(i).rjust(2,"0"))

    if path.isfile(name_i):
        return

    print( "Saving matrix row {}".format( name_i ) )
    if i < total_partitions - 1:
        mat_i = np.zeros( (partition_size, n) )
    else:
        mat_i = np.zeros( (n % partition_size, n) )

    for j in range(total_partitions):
        if j < i:
            name_ij = news_ssk_dir+"/mat_{}_{}.npy".format(str(j).rjust(2,"0"), str(i).rjust(2,"0"))
            mat_ij = np.load( name_ij ).T
        else:
            name_ij = news_ssk_dir+"/mat_{}_{}.npy".format(str(i).rjust(2,"0"), str(j).rjust(2,"0"))
            mat_ij = np.load( name_ij )
        mat_i[:, j*partition_size:(j+1)*partition_size] = mat_ij

    np.save( name_i, mat_i )

def save_ssk_mat(total_partitions, n, partition_size):
    mat = np.zeros( (n, n) )

    for i in range(total_partitions):
        for j in range(i, total_partitions):
            name_ij = news_ssk_dir+"/mat_{}_{}.npy".format(str(i).rjust(2,"0"), str(j).rjust(2,"0"))
            mat_ij = np.load( name_ij )

            i_start, i_end = i*partition_size, (i+1)*partition_size
            j_start, j_end = j*partition_size, (j+1)*partition_size
            mat[i_start:i_end, j_start:j_end] = mat_ij

            if i!=j:
                mat[j_start:j_end, i_start:i_end] = mat_ij.T

    np.save(news_ssk_dir+".npy", mat)

def save_string_list( name, list_ ):
    with open(name, "w") as f:
        for s in list_:
            f.write(s)
            f.write("\n")

into_rows = False

if __name__ == '__main__':
    partition_size = 1000

    if path.isfile("./news_subset.txt") and path.isfile("./labels_subset.txt"):
        news = open("./news_subset.txt").read().split("\n")
        labels = open("./labels_subset.txt").read().split("\n")
        del news[-1] # last one is empty
        del labels[-1]
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

    n = len(news)
    total_partitions = int(ceil(n/float(partition_size)))

    print( "Total datapoints: {}".format(n) )
    print( "Text partitions: {}".format(total_partitions) )
    print( "Total partitions: {}".format(total_partitions*total_partitions) )
    print( "Total (partition) computations: {}".format( int(total_partitions*(total_partitions+1)/2.) ) )

    if    (    into_rows and not path.isfile(news_ssk_dir+"/mat_00.npy")) \
       or (not into_rows and not path.isfile(news_ssk_dir+".npy")):
        news_partitioned = [ news[i*partition_size:(i+1)*partition_size] for i in range(total_partitions) ]

        news_s = []
        for i, news_l in enumerate(news_partitioned):
            feats_l = np.array(news_l).reshape( (len(news_l), 1) )
            for j, news_r in enumerate(news_partitioned):
                if j < i:
                    continue
                if i==j:
                    feats_r = feats_l
                else:
                    feats_r = np.array(news_r).reshape( (len(news_r), 1) )
                news_s.append( (i, j, feats_l, feats_r) )

        if not path.isdir(news_ssk_dir):
            os.mkdir(news_ssk_dir)

        pool = Pool(processes=10)
        pool.map( process_string_list, news_s )

        print( "Finished computing 'dot product' (kernel application) between all pairs of news" )

        if into_rows:
            for input_ in [(i, total_partitions, n, partition_size) for i in range(total_partitions)]:
                create_ssk_rows(input_)
            #pool.map(create_ssk_rows, [(i, total_partitions, n, partition_size) for i in range(total_partitions)])
        else:
            save_ssk_mat(total_partitions, n, partition_size)

    print("Preprocessing finished")
