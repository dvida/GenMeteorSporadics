#!python
#cython: language_level=2, boundscheck=False, wraparound=False

from __future__ import division, print_function
import numpy as np

# Import cython libraries
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport pow, sqrt, sin, asin, cos, acos, atan2, M_PI
from libc.stdlib cimport malloc, free

# Define cython numpy types
INT_TYPE = np.int32
ctypedef np.int32_t INT_TYPE_t

FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t



cdef double radians(double angle):
    """ Convert the angle from degrees to radians. """

    return angle*(M_PI/180.0)



@cython.nonecheck(False)
@cython.cdivision(True) 
cdef double calcD_SH(double *point1, double *point2):
    """ Calculate the Southworth and Hawking meteoroid orbit dissimilarity criterion. Only parameters used are
        q, e, i, O, w, other are disregarded.

    Arguments:
        point1: [double pointer] container for:
            ra1: [double] right ascension, 1st orbit (deg)
            dec1: [double] declination, 1st orbit (deg)
            sol1: [double] solar longitude, 1st orbit (deg)
            vg1: [double] geocentric velocity, 1st orbit (km/s)
            q1: [double] perihelion distance of the first orbit
            e1: [double] num. eccentricity of the first orbit
            i1: [double] inclination of the first orbit (deg)
            O1: [double] longitude of ascending node of the first orbit (deg)
            w1: [double] argument of perihelion of the first orbit (deg)

        point2: [double pointer] container for:
            ra2: [double] right ascension, 2nd orbit (deg)
            dec2: [double] declination, 2nd orbit (deg)
            sol2: [double] solar longitude, 2nd orbit (deg)
            vg2: [double] geocentric velocity, 2nd orbit (km/s)
            q2: [double] perihelion distance of the second orbit
            e2: [double] num. eccentricity of the second orbit
            i2: [double] inclination of the second orbit (deg)
            O2: [double] longitude of ascending node of the second orbit (deg)
            w2: [double] argument of perihelion of the second orbit (deg)

    Return:
        [double] D_SH value

    """

    cdef double q1, e1, i1, O1, w1, q2, e2, i2, O2, w2

    # Unpack input values and convert to radians
    q1 = point1[0]
    e1 = point1[1]
    i1 = radians(point1[2])
    O1 = radians(point1[3])
    w1 = radians(point1[4])

    q2 = point2[0]
    e2 = point2[1]
    i2 = radians(point2[2])
    O2 = radians(point2[3])
    w2 = radians(point2[4])

    cdef double rho = 1

    if (abs(O2 - O1) > M_PI):
        rho = -1

    cdef double I21 = acos(cos(i1) * cos(i2) + sin(i1) * sin(i2) * cos(O2 - O1))
    cdef double pi21 = w2 - w1 + 2 * rho * asin(cos((i2 + i1)/2.0) * sin((O2 - O1)/2.0) * 1/cos(I21 / 2.0))
    cdef double DSH2 = pow((e2 - e1), 2) + pow((q2 - q1), 2) + pow((2 * sin(I21/2.0)), 2) + \
        pow((e2 + e1)/2.0, 2) * pow((2 * sin(pi21 / 2.0)), 2)


    return sqrt(DSH2)



def findNearestNeighborDistDSH(np.ndarray[FLOAT_TYPE_t, ndim=2] original_data):
    """ Find the mean nearest neighbor D_SH by pairing each input orbit with each orbit. """


    # Define the data indices
    cdef int q_index = 0
    cdef int e_index = 1
    cdef int i_index = 2
    cdef int node_index = 3
    cdef int peri_index = 4

    cdef int i = 0
    cdef int j = 0

    cdef int n_orbits = 0

    cdef float minimum_dsh = 999.0
    cdef float dsh_sum = 0.0

    # Determine the input data vector size
    cdef int vector_size = 5

    # Init the vector for the 2 input points
    cdef double *point1 = <double *>malloc(vector_size*sizeof(double))
    cdef double *point2 = <double *>malloc(vector_size*sizeof(double))
    cdef int ind

    print('Size of input data', original_data.shape[0])

    # Take every Nth point (to speed up the calculation)
    cdef int take_every_nth = 20

    for i in range(original_data.shape[0]):

        minimum_dsh = 999.0

        # Print every 1000th entry to see the progress
        if i % 1000 == 0:
            print(i)

        # Take every 10th point to speed up the process
        if not (i % take_every_nth == 0):
            continue

        # Count the number of checked orbits
        n_orbits += 1

        for j in range(original_data.shape[0]):

            # Skip if the two orbits are the same
            if i == j:
                continue

            # Fill the points with input data
            for ind in range(vector_size):
                point1[ind] = original_data[i, ind]
                point2[ind] = original_data[j, ind]

            # Calculate DSH
            current_dsh = calcD_SH(point1, point2)

            if current_dsh < minimum_dsh:
                minimum_dsh = current_dsh

        dsh_sum += minimum_dsh


    # Calculate the average DSH
    # return dsh_sum/(float(original_data.shape[0]) / take_every_nth)
    return dsh_sum/float(n_orbits)



def findMinimumDSH(np.ndarray[FLOAT_TYPE_t, ndim=2] original_data, np.ndarray[FLOAT_TYPE_t, ndim=2] synthetic_data):
    """ Find the mean mimimum D_SH by pairing each original orbit to the synthetic orbit. I.e. take the 
        minimum D_SH for each (original-synthetic) pair and average them. """


    # Define the data indices
    cdef int q_index = 0
    cdef int e_index = 1
    cdef int i_index = 2
    cdef int node_index = 3
    cdef int peri_index = 4

    cdef int i = 0
    cdef int j = 0

    cdef float minimum_dsh = 999.0
    cdef float dsh_sum = 0.0

    # Determine the input data vector size
    cdef int vector_size = 5

    # Init the vector for the 2 input points
    cdef double *point1 = <double *>malloc(vector_size*sizeof(double))
    cdef double *point2 = <double *>malloc(vector_size*sizeof(double))
    cdef int ind

    print('Size of input data', original_data.shape[0])

    for i in range(original_data.shape[0]):

        minimum_dsh = 999.0

        # Print every 100th entry to see the progress
        if i % 100 == 0:
            print(i)

        for j in range(synthetic_data.shape[0]):

            # Fill the points with input data
            for ind in range(vector_size):
                point1[ind] = original_data[i, ind]
                point2[ind] = synthetic_data[j, ind]

            # Calculate DSH
            current_dsh = calcD_SH(point1, point2)

            if current_dsh < minimum_dsh:
                minimum_dsh = current_dsh

        dsh_sum += minimum_dsh


    # Calculate the average DSH
    return dsh_sum/float(original_data.shape[0])







