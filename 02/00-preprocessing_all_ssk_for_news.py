import pyximport; pyximport.install()
from ssk.string_kernel import ssk, string_kernel

import numpy as np
from math import ceil
from sys import stdout
from os import path
from multiprocessing import Pool
import os

def process_string_list( input_ ):
    (i, j, feats_l, feats_r) = input_
    name = "news-ssk/mat_{}_{}.npy".format(str(i).rjust(2,"0"), str(j).rjust(2,"0"))

    #print( "Calculating {}".format( name ) )

    if not path.isfile(name):
        mat = string_kernel(feats_l, feats_r, 5, .8)

        np.save(name, mat)

    stdout.write(".")
    stdout.flush()

def create_ssk_rows( input_ ):
    i, total_partitions, n, partition_size = input_
    name_i = "news-ssk/mat_{}.npy".format(str(i).rjust(2,"0"))

    if path.isfile(name_i):
        return

    print( "Saving matrix row {}".format( name_i ) )
    if i < total_partitions - 1:
        mat_i = np.zeros( (partition_size, n) )
    else:
        mat_i = np.zeros( (n % partition_size, n) )

    for j in range(total_partitions):
        if j < i:
            name_ij = "news-ssk/mat_{}_{}.npy".format(str(j).rjust(2,"0"), str(i).rjust(2,"0"))
            mat_ij = np.load( name_ij ).T
        else:
            name_ij = "news-ssk/mat_{}_{}.npy".format(str(i).rjust(2,"0"), str(j).rjust(2,"0"))
            mat_ij = np.load( name_ij )
        mat_i[:, j*partition_size:(j+1)*partition_size] = mat_ij

    np.save( name_i, mat_i )

if __name__ == '__main__':
    test = open("./news.txt").read().split('\n')
    del test[-1] # last one is empty

    if not path.isdir("news-ssk"):
        os.mkdir("news-ssk")

    n = len(test)
    partition_size = 1000
    total_partitions = int(ceil(n/float(partition_size)))

    test_partitioned = [ test[i*partition_size:(i+1)*partition_size] for i in range(total_partitions) ]

    print( "Text partitions: {}".format(total_partitions) )
    print( "Total partitions: {}".format(total_partitions*total_partitions) )
    print( "Total (partition) computations: {}".format( int(total_partitions*(total_partitions+1)/2.) ) )

    tests = []
    for i, testl in enumerate(test_partitioned):
        feats_l = np.array(testl).reshape( (len(testl), 1) )
        for j, testr in enumerate(test_partitioned):
            if j < i:
                continue
            if i==j:
                feats_r = feats_l
            else:
                feats_r = np.array(testr).reshape( (len(testr), 1) )
            tests.append( (i, j, feats_l, feats_r) )

    pool = Pool(processes=10)
    pool.map( process_string_list, tests )

    print( "Finished computing 'dot product' (kernel application) between all pairs of news" )

    for input_ in [(i, total_partitions, n, partition_size) for i in range(total_partitions)]:
        create_ssk_rows(input_)
    #pool.map(create_ssk_rows, [(i, total_partitions, n, partition_size) for i in range(total_partitions)])
