'''
    File name: dtw.py
    Author: Xiao Yang
    Date created: 17/08/2018
    Date last modified: 17/08/2018
    Python Version: 3.6
'''

import doctest
import numpy as np
import collections

def euclian_distance(x, y):
    """
    Calculates the Eucliean distance

    >>> euclian_distance(0, 3)
    9
    >>> euclian_distance(-1, 3)
    16
    >>> euclian_distance(-1, 4)
    25
    >>>

    """
    return (x-y)**2

def dtw_helper(t1, t2, distance, time_warp_window):
    """
    Calculates Dynamic Time Warping of two sequences

    Based on the following paper:
    http://www.producao.usp.br/handle/BDPI/51065
    
    :param array t1: size r
    :param array t2: size c
    :param func dist: distance measure
    :param time_warp_window: to limit the search range

    >>> t1 = [1,3,4,9,8,2,1,5,7,3]
    >>> t2 = [1,6,2,3,0,9,4,3,6,3]
    >>> sum(euclian_distance(x, y) for x, y in zip(t1, t2))
    176
    >>> time_warp_window = max(len(t1), len(t2))
    >>> dtw_helper(t1, t2, euclian_distance, time_warp_window)
    array([[  0.,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf],
           [ inf,   0.,  25.,  26.,  30.,  31.,  95., 104., 108., 133., 137.],
           [ inf,   4.,   9.,  10.,  10.,  19.,  55.,  56.,  56.,  65.,  65.],
           [ inf,  13.,   8.,  12.,  11.,  26.,  44.,  44.,  45.,  49.,  50.],
           [ inf,  77.,  17.,  57.,  47.,  92.,  26.,  51.,  80.,  54.,  85.],
           [ inf, 126.,  21.,  53.,  72., 111.,  27.,  42.,  67.,  58.,  79.],
           [ inf, 127.,  37.,  21.,  22.,  26.,  75.,  31.,  32.,  48.,  49.],
           [ inf, 127.,  62.,  22.,  25.,  23.,  87.,  40.,  35.,  57.,  52.],
           [ inf, 143.,  63.,  31.,  26.,  48.,  39.,  40.,  39.,  36.,  40.],
           [ inf, 179.,  64.,  56.,  42.,  75.,  43.,  48.,  55.,  37.,  52.],
           [ inf,  inf,  73.,  57.,  42.,  51.,  79.,  44.,  44.,  46.,  37.]])
    >>> time_warp_window = max(len(t1), len(t2)) // 2
    >>> dtw_helper(t1, t2, euclian_distance, time_warp_window)
    array([[  0.,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf],
           [ inf,   0.,  25.,  26.,  30.,  31.,  95.,  inf,  inf,  inf,  inf],
           [ inf,   4.,   9.,  10.,  10.,  19.,  55.,  56.,  inf,  inf,  inf],
           [ inf,  13.,   8.,  12.,  11.,  26.,  44.,  44.,  45.,  inf,  inf],
           [ inf,  77.,  17.,  57.,  47.,  92.,  26.,  51.,  80.,  54.,  inf],
           [ inf, 126.,  21.,  53.,  72., 111.,  27.,  42.,  67.,  58.,  79.],
           [ inf, 127.,  37.,  21.,  22.,  26.,  75.,  31.,  32.,  48.,  49.],
           [ inf,  inf,  62.,  22.,  25.,  23.,  87.,  40.,  35.,  57.,  52.],
           [ inf,  inf,  inf,  31.,  26.,  48.,  39.,  40.,  39.,  36.,  40.],
           [ inf,  inf,  inf,  inf,  42.,  75.,  43.,  48.,  55.,  37.,  52.],
           [ inf,  inf,  inf,  inf,  inf,  51.,  79.,  44.,  44.,  46.,  37.]])
    >>>

    """

    upper_bound = sum(distance(x, y) for x, y in zip(t1, t2))

    r, c = len(t1), len(t2)

    D = np.full((r+1, c+1), np.inf)
    D[0, 0] = 0

    search_start = 1
    search_end = 1

    # iterate row-wise

    for i in range(1, r+1): # row
        
        below_upper_bound = False
        next_search_end = i
        
        # iterate column-wise

        j_begin = max(search_start, i - time_warp_window)
        j_end = min(i + time_warp_window, c)
        for j in range(j_begin, j_end+1):
            
            D[i, j] = distance(t1[i-1], t2[j-1]) + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
            
            # pruning, i.e. define the start and end for the next row, and skip current row instances
            if D[i, j] > upper_bound:
                if below_upper_bound == False:
                    search_start = j + 1
                if j >= search_end:
                    # the next value in the row is above the upper bound
                    break
            else:
                below_upper_bound = True
                next_search_end = j + 1

        search_end = next_search_end

    return D

def trace(D):
    """
    Finds the path with the minimum cost

    :param array D: size (r, c)

    >>> t1 = [1,3,4,9,8,2,1,5,7,3]
    >>> t2 = [1,6,2,3,0,9,4,3,6,3]
    >>> time_warp_window = max(len(t1), len(t2)) // 2
    >>> D = dtw_helper(t1, t2, euclian_distance, time_warp_window)
    >>> D
    array([[  0.,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf],
           [ inf,   0.,  25.,  26.,  30.,  31.,  95.,  inf,  inf,  inf,  inf],
           [ inf,   4.,   9.,  10.,  10.,  19.,  55.,  56.,  inf,  inf,  inf],
           [ inf,  13.,   8.,  12.,  11.,  26.,  44.,  44.,  45.,  inf,  inf],
           [ inf,  77.,  17.,  57.,  47.,  92.,  26.,  51.,  80.,  54.,  inf],
           [ inf, 126.,  21.,  53.,  72., 111.,  27.,  42.,  67.,  58.,  79.],
           [ inf, 127.,  37.,  21.,  22.,  26.,  75.,  31.,  32.,  48.,  49.],
           [ inf,  inf,  62.,  22.,  25.,  23.,  87.,  40.,  35.,  57.,  52.],
           [ inf,  inf,  inf,  31.,  26.,  48.,  39.,  40.,  39.,  36.,  40.],
           [ inf,  inf,  inf,  inf,  42.,  75.,  43.,  48.,  55.,  37.,  52.],
           [ inf,  inf,  inf,  inf,  inf,  51.,  79.,  44.,  44.,  46.,  37.]])
    >>> trace(D)
    deque([(0, 0), (1, 1), (2, 2), (2, 3), (2, 4), (3, 5), (4, 6), (5, 6), (6, 7), (7, 8), (8, 9), (9, 9), (10, 10)])
    >>>

    """
    i, j = D.shape[0] - 1, D.shape[1] - 1
    steps = collections.deque()
    steps.appendleft((i, j))
    while (i > 0) or (j > 0):
        index = np.argmin((D[i-1, j-1], D[i-1, j], D[i, j-1]))
        if index == 0:
            i -= 1
            j -= 1
        elif index == 1:
            i -= 1
        else:
            j -= 1
        steps.appendleft((i,j))

    return steps

def dtw(t1, t2, time_warp_window):
    D = dtw_helper(t1, t2, euclian_distance, time_warp_window)
    steps = trace(D)
    return D[len(t1), len(t2)], steps, D

if __name__ == "__main__":
    # doctest.testmod()
    
    t1 = [1,3,4,9,8,2,1,5,7,3]
    print("t1")
    print(t1)
    
    t2 = [1,6,2,3,0,9,4,3,6,3]
    print("t2")
    print(t2)

    time_warp_window = max(len(t1), len(t2)) // 2
    print("time_warp_window")
    print(time_warp_window)

    cost, steps, D = dtw(t1, t2, time_warp_window)
    print("cost")
    print(cost)
    print("steps")
    print(steps)
    print("D")
    print(D)

