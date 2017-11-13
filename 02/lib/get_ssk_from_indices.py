import numpy as np
from math import floor

n = 32603
total_partitions = 33
partition_size = 1000

def ssk_from_indices( indices_l, indices_r ):
    lenl, lenr = len(indices_l), len(indices_r)

    mat = np.zeros( (lenl, lenr) )

    indices_l_sorted = [[] for i in range(total_partitions)]
    indices_orig_l_sorted = [[] for i in range(total_partitions)]
    for jl, il in enumerate(indices_l):
        i = int(floor(il/partition_size))
        indices_l_sorted[i].append( jl )
        indices_orig_l_sorted[i].append( il )

    for i in range(total_partitions):
        if len( indices_l_sorted[i] ) == 0:
            continue
        name_i = "news-ssk/mat_{}.npy".format(str(i).rjust(2,"0"))
        mat_i = np.load( name_i )

        jls = indices_l_sorted[i]
        ils = np.array( [[il] for il in indices_orig_l_sorted[i]] ) - i*partition_size

        mat[jls, :] = mat_i[ils, indices_r]

    return mat
