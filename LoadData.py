""" Loading meteor orbit data from a UFO Orbit format CSV file or CAMS format. """

from __future__ import print_function

import numpy as np
from datetime import timedelta
from datetime import datetime


# Define Julian epoch
JULIAN_EPOCH = datetime(2000, 1, 1, 12) # noon (the epoch name is unrelated)
J2000_JD = timedelta(2451545) # julian epoch in julian dates



def date2JD(year, month, day, hour, minute, second, millisecond=0, UT_corr=0.0, mjd=False):
    """ Convert date and time to Julian Date with epoch J2000.0. 

    Arguments:
        year: [int] year
        month: [int] month
        day: [int] day of the date
        hour: [int] hours
        minute: [int] minutes
        second: [int] seconds
        millisecond: [int] milliseconds (optional)
        UT_corr: [float] UT correction in hours (difference from local time to UT)
    
    Keyword arguments:
        mjd: [bool] False by default, if true, the function return the modified Julian date

    Return:
        [float] julian date, epoch 2000.0 (or MJD if mjd=True)

    """

    # Create datetime object of current time
    dt = datetime(year, month, day, hour, minute, second, int(millisecond*1000.0))

    # Calculate Julian date
    julian = dt - JULIAN_EPOCH + J2000_JD - timedelta(hours=UT_corr)
    
    # Convert seconds to day fractions
    JD = julian.days + (julian.seconds + julian.microseconds/1000000.0)/86400.0

    # Return the modified Julian date if mjd=True
    if mjd:
        return JD - 2400000.5

    # Return JD if mjd=False
    else:
        return JD



def loadData(file_name, sol_min, sol_max, dV_thres=3):
    """ Load UFOOrbit format orbit data to a numpy array by a solar longitude range. 

    Arguments:
        file_name: [string] path to the UFOOrbit format data file
        sol_min: [int] minimum solar longitude
        sol_max: [int] maximum solar longitude

    Keyword arguments:
        dV_thres: [float] maximum meteor velocity difference in % used as an orbit quality criteria, 
            disable by setting dV_thresh = -1

    Return:
        [2D ndarray] numpy array containing meteor orbit data

    """

    orbits_array = []

    first_line = True
    for line in open(file_name):

        # Skip first line
        if first_line:
            first_line = False
            continue

        # Split line into list
        line = line.split()

        # Check solar longitude
        sol_lon = float(line[4])

        if (sol_lon > sol_max) and (sol_max > sol_min):
            break

        # Take orbits only from the given solar longitude range (take care of 0-360 wraparound)
        if (sol_lon > sol_min) or ((sol_min > sol_max) and (sol_lon < sol_max)):

            # Velocity difference in percents
            dV12 = float(line[74])

            # Reject orbit if the velocity difference is too large
            if (dV12 > dV_thres) and (dV_thres != -1):
                continue

            # ID of a meteor
            ID = str(line[1])

            # Julian date
            mjd = float(line[3])

            # Absolute magnitude
            mag_abs = float(line[7])

            # Right ascension (geocentric)
            ra_t = float(line[10])

            # Declination (geocentric)
            dec_t = float(line[11])

            # Velocity at infinity
            vi = float(line[15])

            # Geocentric velocity
            vg = float(line[16])

            # Semi-major axis
            a = float(line[18])

            # Perihelion distance
            q = float(line[19])

            # Num. eccentricity
            e = float(line[20])

            # Argument of perihelion
            peri = float(line[22])

            # longitude of ascending node
            node = float(line[23])

            # Inclination
            incl = float(line[24])

            # Beginning height
            H_beg = float(line[43])

            # End height
            H_end = float(line[51])

            # Cross angle of two observed planes (deg)
            Qc = float(line[71])

            # Total quality assessment
            Qa = float(line[78])

            orbits_array.append(np.array([ID, mjd, sol_lon, mag_abs, ra_t, dec_t, vi, vg, 
                a, q, e, incl, node, peri, H_beg, H_end, Qc, dV12, Qa]))

    # Remove first entry (zeros) and return
    return np.array(orbits_array)



def loadDataCAMS(file_name, sol_min, sol_max, Q_thresh=15.0, Vg_error_thresh=0.1, shower_num=None):
    """ Load UFOOrbit format orbit data to a numpy array by a solar longitude range. 
    
    Arguments:
        file_name: [string] path to the UFOOrbit format data file
        sol_min: [int] minimum solar longitude
        sol_max: [int] maximum solar longitude
        Q_thresh: [float] minimum convergence angle between stations
        Vg_error_thresh: [float] Vg error must be within this fraction of the Vg
        shower_num: [int or None] returns the members of the given shower (IAU numer), if None, all orbits are 
            returned

    Return
        """

    skip_rows = 6
    delimiter = ','

    orbits_array = []

    for line in open(file_name).readlines()[skip_rows:]:

        # Split line into list
        line = line.split(delimiter)

        # Skip if empty row
        if line[0] == '' or line[1] == '':
            continue

        # Check solar longitude
        sol_lon = float(line[57])

        if (sol_lon > sol_max) and (sol_max > sol_min):
            break

        # Take orbits only from the given solar longitude range (take care of 0-360 wraparound)
        if (sol_lon > sol_min) or ((sol_min > sol_max) and (sol_lon < sol_max)):

            # ID of a meteor
            ID = str(line[0])

            # Referent date and time
            date = line[2]

            # Skip if empty row
            if date == '':
                continue

            MM, dd, yy = map(int, date.split('-'))

            time = line[3]
            hh, mm, ss = map(int, time.split(':'))

            # Create datetime object with meteor time
            meteor_datetime = datetime(yy, MM, dd, hh, mm, ss)
            
            Tbeg_delta = float(line[4])
            Tend_delta = float(line[5])

            # Calculate real meteor beg/end times
            meteor_datetime_beg = meteor_datetime + timedelta(seconds=Tbeg_delta)
            meteor_datetime_end = meteor_datetime + timedelta(seconds=Tend_delta)

            # Get begining info
            yy = meteor_datetime_beg.year
            MM = meteor_datetime_beg.month
            dd = meteor_datetime_beg.day
            hh = meteor_datetime_beg.hour
            mm = meteor_datetime_beg.minute
            ss = meteor_datetime_beg.second
            ms = meteor_datetime_beg.microsecond/1000.0

            # Modified julian date
            mjd = date2JD(yy, MM, dd, hh, mm, ss, ms, mjd=True)

            # If looking only for specific shower, skip all others
            if shower_num is not None:
                if int(line[37]) != shower_num:
                    continue

            # Absolute magnitude
            mag_abs = float(line[32])

            # Right ascension (geocentric)
            ra_t = float(line[41])

            # Declination (geocentric)
            dec_t = float(line[43])

            # Velocity at infinity
            vi = float(line[10])

            # Geocentric velocity
            vg = float(line[45])
            vg_error = float(line[46])

            # Skip this orbit if Vg has a too large error
            if vg_error >= vg*Vg_error_thresh:
                continue

            # Semi-major axis
            a = float(line[63])

            # Perihelion distance
            q = float(line[59])
            q_std = float(line[60])

            # Num. eccentricity
            e = float(line[64])
            e_std = float(line[65])

            # Argument of perihelion
            peri = float(line[68])
            peri_std = float(line[69])

            # longitude of ascending node
            node = float(line[70])
            node_std = float(line[71])

            # Inclination
            incl = float(line[66])
            incl_std = float(line[67])

            # Beginning height
            H_beg = float(line[20])

            # End height
            H_end = float(line[26])

            # Cross angle of two observed planes (deg)
            Qc = float(line[28])

            # Skip this orbit if the convergence angle is too low
            if Qc < Q_thresh:
                continue

            # Geocentric eliptic latitude
            beta_g = float(line[49])

            ### Set missing params to 0
            # Velocity difference
            dV12 = 0

            ###

            # Append the orbit
            orbits_array.append(np.array([ID, mjd, sol_lon, mag_abs, ra_t, dec_t, vi, vg, 
                a, q, e, incl, node, peri, H_beg, H_end, Qc, dV12, beta_g, q_std, e_std, incl_std, node_std, 
                peri_std]))
            
            # print ID, mjd, sol_lon, ra_t, dec_t, vg, q, e, incl, node, peri

    return np.array(orbits_array)
        



if __name__ == "__main__":

    # Parse CAMS data for Perseids (IAU shower number 7)
    orbits_array = loadDataCAMS('CAMS-v2-2013.csv', 100, 150, shower_num=7)

    print(orbits_array)