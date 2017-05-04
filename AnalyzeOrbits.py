""" CAMS data plotting procedures. Data investigation only. """

import numpy as np


from LoadData import loadDataCAMS
from SporadicsGenerateSynthetic import *





if __name__ == "__main__":

    # Name of the input data file
    input_file_name = "CAMS-v2-2013.csv"

    ### Set data parameters

    # Range of solar longitudes to use
    sol_min = 0
    sol_max = 360

    # Convergence angle threshold
    Q_thresh = 15.0 # deg

    # Geocentric velocity threshold
    Vg_error_thresh = 0.1

    ###

    # Load CAMS data (sporadics only, thus shower_num=0)
    print('Loading data...')
    loaded_data = loadDataCAMS(input_file_name, sol_min, sol_max, Q_thresh, Vg_error_thresh, 
        shower_num=0).astype(np.float64)

    # Reject all hyperbolic and parabolic orbits
    loaded_data = loaded_data[loaded_data[:, ECC_INDX] < 1.0]
    print('Input data size:', len(loaded_data))



    ### DATA FILTERING    
    # loaded_data = loaded_data[loaded_data[:, BETA_G_INDX] < 0]
    # loaded_data = loaded_data[loaded_data[:, SOL_INDX] > 180]
    # loaded_data = loaded_data[loaded_data[:, INCL_INDX] < 90]
    # loaded_data = loaded_data[loaded_data[:, ECC_INDX] > 0.9]

    # loaded_data = loaded_data[(loaded_data[:, SOL_INDX] < 90) | (loaded_data[:, SOL_INDX] > 270)]

    #loaded_data = loaded_data[(loaded_data[:, SOL_INDX] - loaded_data[:, NODE_INDX]) > 90]

    # loaded_data = loaded_data[loaded_data[:, LAMBDA_G_INDX] > 90.0]

    ###


    # Earth's range of perihelion distances
    q_earth_min = 0.983
    q_earth_max = 1.016


    # Extract argument of perihelion data
    peri_data = loaded_data[:, PERI_INDX]

    # Extract q data
    q_data = loaded_data[:, Q_INDX]

    omegas = np.linspace(0, 360, 100)

    # # Plot the Earth's range of perihleion distances
    # plt.plot(omegas, np.zeros_like(omegas) + q_earth_min)
    # plt.plot(omegas, np.zeros_like(omegas) + q_earth_max)

    ### PLOT ARROWS

    # # Plot an arrow pointing to the low-q branch
    # opt_arrow_1 = dict(color='k', ec='w', lw=1,
    #         arrowstyle = 'simple, head_width=.75, head_length=.75',
    #         connectionstyle = 'arc3, rad=0')

    # arrow_low_q = plt.gca().annotate(r'$\lambda_{\odot} \geq 180$', xy=(170, q_earth_min), xycoords='data', 
    #     xytext=(85, q_earth_min - 0.1), textcoords = 'data', arrowprops=opt_arrow_1, size=15)


    # # Plot an arrow pointing to the low-q branch
    # opt_arrow_2 = dict(color='k', ec='w', lw=1,
    #         arrowstyle = 'simple, head_width=.75, head_length=.75',
    #         connectionstyle = 'arc3, rad=0')

    # arrow_high_q = plt.gca().annotate(r'$\lambda_{\odot} < 180$', xy=(190, q_earth_max - 0.005), xycoords='data', 
    #     xytext =(230, q_earth_max - 0.1), textcoords = 'data', arrowprops=opt_arrow_2, size=15)

    # ######

    # Positive and negative beta branche
    q_neg_beta = 1/2.0*(1 + np.cos(np.radians(omegas)))
    q_pos_beta = 1/2.0*(1 + np.cos(np.pi + np.radians(omegas)))

    # Plot beta lines
    plt.plot(omegas, q_neg_beta, color='r', linestyle='--', label=r"$\beta < 0$, ascending node")
    plt.plot(omegas, q_pos_beta, color='b', label=r"$\beta > 0$, descending node")

    plt.legend(loc='lower left')

    plot2DHistogram(peri_data, q_data, '', '$\omega$', '$q$', subplot=True)

    # plt.colorbar(label='Counts')

    savePlot(plt, 'peri_vs_q_explanation')
    plt.show()



    # Plot the solar longitude vs. frequency histogram
    plt.hist(loaded_data[:, SOL_INDX], bins=60)
        
    plt.xlim([0, 360])

    plt.xlabel('$\lambda_{\odot}$')
    plt.ylabel('Counts')

    plt.show()