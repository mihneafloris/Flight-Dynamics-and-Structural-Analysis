
""" 
 This is file is part of the developer code of the AE3212-II project. 
 The function defined in this code allows for a linear, least-squares regression
 of a given 2D data set.

 Author: Sam van Elsloo
 Date:   06/02/2021
"""

import numpy as np 

def polyfit(x, y, z, kx=3, ky=3):
    ### Polyfit-function, adapted from answers provided by 
    # https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python

    ### Input parameters:
    # x,y: array-like, 1d, containing coordinates in x- and y-direction.
    # z: np.ndarray, 2d, containing data values.
    # kx,ky: polynomial order in x and y, respectively.

    def transformx (x):
        return -1 + 2 * (x - xmin)/(xmax - xmin)
    def transformy (y):
        return -1 + 2 * (y - ymin)/(ymax - ymin)

    xmax = np.amax(x)
    xmin = np.amin(x)
    ymax = np.amax(y)
    ymin = np.amin(y)

    x = transformx(x)
    y = transformy(y)

    xx, yy = np.meshgrid(x,y,copy=False)

    coef = np.ones((kx+1,ky+1))

    A = np.zeros((coef.size,xx.size))

    for index, (i, j) in enumerate(np.ndindex(coef.shape)):
        arr = coef[i,j] * xx**i * yy**j
        A[index] = arr.ravel()

    coef, r, rank, s = np.linalg.lstsq(A.T,np.ravel(z),rcond=None)

    res = lambda x,y: np.polynomial.polynomial.polyval2d(transformx(x), 
                                                         transformy(y), 
                                                         coef.reshape((kx+1,
                                                                       ky+1)))

    return res