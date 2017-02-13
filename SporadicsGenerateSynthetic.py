""" 
    Python module for generating synthetic meteoroid orbits using Kernel Density estimation.

    Reference: Vida, D., Brown, P.G., Campbell-Brown, M. (2017) "Generating realistic synthetic meteoroid 
        orbits"
    

    Style guide used: https://developer.lsst.io/coding/python_style_guide.html 
    Exceptions to the style guide:
        - variable names are "snake_case" instead of "camelCase"
    
"""

# Copyright (c) 2017, Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import division, print_function

import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use the classic matplotlib plotting style
matplotlib.style.use('classic')

import scipy.stats as stat

from LoadData import loadDataCAMS

# Check if cython is installed
try:
    
    import cython
    
    # Cython init
    import pyximport
    pyximport.install(setup_args={'include_dirs':[np.get_include()]})
    from SporadicsCompareOrbitsDSH import findMinimumDSH, findNearestNeighborDistDSH

except:
    print("'Cython' module not installed - you will not se able to calculate the nearest neighbour distances \
        in D_SH!")


# loadData indices - indices of parameters passed from the loadData function
MJD_INDX =      1
SOL_INDX =      2
AMAG_INDX =     3
RA_INDX =       4
DEC_INDX =      5
VI_INDX =       6
VG_INDX =       7

A_INDX =        8
Q_INDX =        9
ECC_INDX =      10
INCL_INDX =     11
NODE_INDX =     12
PERI_INDX =     13

Q_STD_INDX =    20
ECC_STD_INDX =  21
INCL_STD_INDX = 22
NODE_STD_INDX = 23
PERI_STD_INDX = 24
VG_STD_INDX =   25

HBEG_INDX =     14
HEND_INDX =     15
QC_INDX =       16
DV12_INDX =     17
LAMBDA_G_INDX = 18
BETA_G_INDX =   19


# Font sizes for histogram
HIST_TICK_FONTSIZE = 8
HIST_TITLE_FONTSIZE = 9
HIST_LABEL_FONTSIZE = 14



def drawFromHistogram(data_array, n_bins, n_samples):
    """ Draw random samples from a histogram using the Cumulative density method. 

    Arguments:
        data_array: [1D ndarray] numpy array containing the data
        n_bins: [int] number of bins of the histogram
        n_samples: [int] number of samples to draw from the histogram

    Return:
        random_from_cdf: [ndarray] data drawn from the histogram

    """

    # Initialize the histogram
    hist, bins = np.histogram(data_array, bins=n_bins)

    # Calculate the midpoints of histograms
    bin_midpoints = bins[:-1] + np.diff(bins)/2
    
    # Generate the cumulative sum of the histogram
    cdf = np.cumsum(hist)

    # Normalize the cumulative sum
    cdf = cdf / float(cdf[-1])

    # Take uniform samples from the histogram
    values = np.random.rand(n_samples)

    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]

    bin_width = (max(data_array) - min(data_array))/n_bins

    # Uniformly sample from within each bin (if were were not doing this, then we would have only one value
    # per bin, i.e. only n_bins unique values)
    random_from_cdf += np.random.random(n_samples)*bin_width - bin_width/2.0

    return random_from_cdf



def plotCompareHistograms(original_data, synthetic_data, n_bins, title, subplot=False, save_legend=False, 
    data_label=''):
    """ Plot 2 histograms, one overlaid on top of the other, for easy comparison of the two. 
    
    Arguments:
        original_data: [ndarray] first data array (observed data)
        synthetic_data: [ndarray] second data array (synthetic data)
        n_bins: [ndarray] number of bins of histograms
        title: [str] title of the histogram

    Keyword arguments:
        subplot: [bool] False by default, set to True if doing multiple subplots on a single graph
        save_legend: [bool] plot the histogram legend
        data_label: [str] add this prefix in front of each name of the file that is saved to disk

    Return:
        None
    """
    
    # Init the colormap
    cmap = matplotlib.cm.get_cmap('inferno')

    plt.hist(original_data, n_bins, alpha=0.5, color=cmap(0.1), label='Observed', hatch='//', zorder=3)
    plt.hist(synthetic_data, n_bins, alpha=0.5, color=cmap(0.9), label='Synthetic', hatch='\\\\', zorder=3)

    plt.xlabel(title)
    plt.ylabel("Counts")

    # Set label size
    plt.gca().xaxis.label.set_size(HIST_LABEL_FONTSIZE)

    ## Uncomment if you want to plot the grid in the background
    #plt.grid(zorder=0)

    # Show legend if not plotting a subplot, or if the legend is requested
    if (not subplot) or (save_legend):
        plt.legend(loc='upper left')

    # Set the X axis limits to the limits of the original data
    plt.xlim((min(original_data), max(original_data)))


    if not subplot:

        plt.tight_layout()

        # Save the plot to disk
        savePlot(plt, title, data_label)

        plt.show()
        plt.clf()



def plot2DHistogram(x_array, y_array, plot_title, x_axis_label, y_axis_tabel, subplot=False, x_width=None, 
    y_width=None, vmin=None, vmax=None):
    """ Plot the 2D histogram (density plot) of two given data arrays.

    Arguments:
        x_array: [1D ndarray] array of X axis values
        y_array: [1D ndarray] array of Y axis values
        plot_title: [str] plot title
        x_axis_label: [str] label of the X axis
        y_axis_label: [str] label of the Y axis

    Keyword arguments:
        subplot: [bool] False by default, set to True if doing multiple subplots on a single graph
        x_width: [float] None by default, if provided it is used for calculating the number of bins by 
            diving the range of the data into equally sized chunks: x_nbins = (x_max - x_min)/x_width
        y_width: [float] None by default, if provided it is used for calculating the number of bins by 
            diving the range of the data into equally sized chunks: y_nbins = (y_max - y_min)/y_width
        vmin: [float] None by default, if given it is the lower limit of the color representing the histogram 
            counts
        vmax: [float] None by default, if given it is the upper limit of the color representing the histogram 
            count

    Return:
        (vmin, vmax): [tuple] minimum and maximum histogram bin counts

    """

    # Construct the x*y matrix
    eq_matrix = np.column_stack((x_array, y_array))
    x, y = eq_matrix.T

    # Determine the data limits
    xmin, xmax = min(x_array), max(x_array)
    ymin, ymax = min(y_array), max(y_array)

    # Calculate the bin number for X data if the widths are provided
    if x_width is not None:
        x_nbins = (xmax - xmin)/x_width
    else:
        # If the widths were not defined, take the default value
        x_nbins = 100
        
    # Calculate the bin number for Y data if the widths are provided
    if y_width is not None:
        y_nbins = (ymax - ymin)/y_width

    else:
        # If the widths were not defined, take the default value
        y_nbins = 100


    # Make a histogram
    hist, x_bins, y_bins = np.histogram2d(x, y, bins=(x_nbins, y_nbins))
    plt.hist2d(x, y, bins=(x_nbins, y_nbins), cmap='inferno_r', vmin=vmin, vmax=vmax)

    # Set plot limits
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))

    # Set the title and labels
    plt.title(plot_title, fontsize=HIST_TITLE_FONTSIZE)
    plt.xlabel(x_axis_label, fontsize=HIST_LABEL_FONTSIZE)
    plt.ylabel(y_axis_tabel, fontsize=HIST_LABEL_FONTSIZE)

    # Change font size of ticks
    plt.tick_params(axis='both', which='major', labelsize=HIST_TICK_FONTSIZE)
    
    if not subplot:
        plt.show()
        plt.clf()


    return np.min(hist), np.max(hist)



def plot2DResidualHistogram(x1_array, y1_array, x2_array, y2_array, plot_title, x_axis_label, y_axis_tabel, 
    subplot=False, x_width=None, y_width=None, vmin=None, vmax=None):
    """ Plot the absolute difference of two 2D histograms (density plots), of two sets of data. The color 
        range of bin counts is determined by the range of the first data set.

    Arguments:
        x1_array: [1D ndarray] first array of X axis values
        y1_array: [1D ndarray] first array of Y axis values
        x2_array: [1D ndarray] second array of X axis values
        y2_array: [1D ndarray] second array of Y axis values
        plot_title: [str] plot title
        x_axis_label: [str] label of the X axis
        y_axis_label: [str] label of the Y axis

    Keyword arguments:
        subplot: [bool] False by default, set to True if doing multiple subplots on a single graph
        x_width: [float] None by default, if provided it is used for calculating the number of bins by 
            diving the range of the data into equally sized chunks: x_nbins = (x_max - x_min)/x_width
        y_width: [float] None by default, if provided it is used for calculating the number of bins by 
            diving the range of the data into equally sized chunks: y_nbins = (y_max - y_min)/y_width
        vmin: [float] None by default, if given it is the lower limit of the color representing the histogram 
            counts
        vmax: [float] None by default, if given it is the upper limit of the color representing the histogram 
            count

    Return:
        None

    """

    # Construct the x1*y1 matrix
    eq_matrix1 = np.column_stack((x1_array, y1_array))
    x1, y1 = eq_matrix1.T

    # Construct the x2*y2 matrix
    eq_matrix2 = np.column_stack((x2_array, y2_array))
    x2, y2 = eq_matrix2.T

    # Determine the data limits by using the first data set
    xmin, xmax = min(x1_array), max(x1_array)
    ymin, ymax = min(y1_array), max(y1_array)

    # Calculate the bin number if the widths are provided
    if x_width is not None:
        x_nbins = (xmax - xmin)/x_width
    else:
        # If the widths were not defined, take the default value
        x_nbins = 100
        
    # Calculate the bin number if the widths are provided
    if y_width is not None:
        y_nbins = (ymax - ymin)/y_width

    else:
        # If the widths were not defined, take the default value
        y_nbins = 100


    # Make a first histogram
    hist1, x1_bins, y1_bins = np.histogram2d(x1, y1, bins=(x_nbins, y_nbins))

    # Make a second histogram
    hist2, x2_bins, y2_bins = np.histogram2d(x2, y2, bins=(x_nbins, y_nbins), range=[[np.min(x1), np.max(x1)], 
        [np.min(y1), np.max(y1)]])


    # Calculate the absolute difference of the two histogram
    residuals = np.abs(hist1 - hist2)

    # # Get the color range
    ## Uncomment if you want to force the color range to the original histogram range
    # vmin = np.min(hist1)
    # vmax = np.max(hist1)

    ## Uncomment if you want to force the color range to the residuals histogram range
    # vmin = np.min(residuals)
    # vmax = np.max(residuals)

    # Plot the difference histogram
    plt.gca().pcolorfast(x1_bins, y1_bins, residuals.T, cmap='inferno_r', vmin=vmin, vmax=vmax)

    # Set the plot limits
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))

    # Set the title and axis labels
    plt.title(plot_title, fontsize=HIST_TITLE_FONTSIZE)
    plt.xlabel(x_axis_label, fontsize=HIST_LABEL_FONTSIZE)
    plt.ylabel(y_axis_tabel, fontsize=HIST_LABEL_FONTSIZE)

    # Change font size of ticks
    plt.tick_params(axis='both', which='major', labelsize=HIST_TICK_FONTSIZE)
    
    if not subplot:
        plt.show()
        plt.clf()



def plotAllResults(loaded_data, inv_a_results, q_results, e_results, peri_results, node_results, incl_results,
        q_std, e_std, peri_std, node_std, incl_std, data_label=''):
    """ Plot all results of synthetic orbit generation. 

    Arguments:
        loaded_data: [2D ndarray] numpy array containing the input data, column indices are defined at the 
            beginning of this file in uppercase letters
        inv_a_results: [ndarray] array containing the inverse semi-major axis data
        q_results: [ndarray] array containing perihelion distance data
        e_results: [ndarray] array containing eccentricity data
        peri_results: [ndarray] array containing argument of perihelion data
        node_results: [ndarray] array containing node data
        incl_results: [ndarray] array containing inclination data
        q_std: [ndarray] mean perihelion distance standard deviation
        e_std: [ndarray] mean eccentricity standard deviation
        peri_std: [ndarray] mean argument of perihelion standard deviation
        node_std: [ndarray] mean node standard deviation
        incl_std: [ndarray] mean inclination standard deviation

    Keyword arguments:
        data_label: [str] add this prefix in front of each name of the file that is saved to disk

    Return:
        None
    """

    ### Plot individual 1D histograms

    n_bins = 80

    # Compare q histograms
    plotCompareHistograms(loaded_data[:, Q_INDX], q_results, n_bins, '$q$', data_label=data_label)

    plt.subplot(411)
    # Compare e histogram
    plotCompareHistograms(loaded_data[:, ECC_INDX], e_results, n_bins, '$e$', subplot=True, save_legend=True)

    plt.subplot(412)
    # Compare peri histogram
    plotCompareHistograms(loaded_data[:, PERI_INDX], peri_results, n_bins, '$\omega$', subplot=True)

    plt.subplot(413)
    # Compare node histogram
    plotCompareHistograms(loaded_data[:, NODE_INDX], node_results, n_bins, '$\Omega$', subplot=True)

    plt.subplot(414)
    # Compare inclination histogram
    plotCompareHistograms(loaded_data[:, INCL_INDX], incl_results, n_bins, '$i$', subplot=True)

    plt.tight_layout()

    # Adjust the spacing of subplots
    plt.subplots_adjust(hspace=0.58, bottom=0.08)

    # Adjust figure size
    plt.gcf().set_size_inches(8, 8, forward=True)

    # Save the plot to disk
    savePlot(plt, 'all', data_label)

    plt.show()
    plt.clf()

    ###

    ### Peri - Incl plots
    
    # Plot the observed data peri-incl 2D histogram
    ax1 = plt.subplot(131)
    vmin, vmax = plot2DHistogram(loaded_data[:, PERI_INDX], loaded_data[:, INCL_INDX], 'Observed data', 
        '$\omega$', '$i$', subplot=True, x_width=peri_std, y_width=incl_std)

    # Set limits of the X axis
    ax1.set_xlim([0, 360])

    
    # Plot the synthetic data peri-incl 2D histogram
    ax2 = plt.subplot(132, sharey=ax1)
    plot2DHistogram(peri_results, incl_results, 'Synthetic data', '$\omega$', '$i$', subplot=True, 
        x_width=peri_std, y_width=incl_std, vmin=vmin, vmax=vmax)

    # Set limits of the X axis
    ax2.set_xlim([0, 360])

    # Update the Y limits to conform to the input data
    plt.ylim((min(loaded_data[:, INCL_INDX]), max(loaded_data[:, INCL_INDX])))

    # Make Y axis tick labels invisible
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_ylabel('')


    # Plot the absolute residuals 2D histogram
    ax3 = plt.subplot(133, sharey=ax1)
    plot2DResidualHistogram(loaded_data[:, PERI_INDX], loaded_data[:, INCL_INDX], peri_results, 
        incl_results, '|Observed-synthetic| residuals', '$\omega$', '$i$', subplot=True, x_width=peri_std, 
        y_width=incl_std, vmin=vmin, vmax=vmax)

    # Set limits of the X axis
    ax3.set_xlim([0, 360])

    # Update the Y limits to conform to the input data
    plt.ylim((min(loaded_data[:, INCL_INDX]), max(loaded_data[:, INCL_INDX])))

    # Make Y axis tick labels invisible
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.set_ylabel('')

    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, wspace=0.05)


    # Get plot positions
    [[x00,y00],[x01,y01]] = ax1.get_position().get_points()
    [[x10,y10],[x11,y11]] = ax3.get_position().get_points()
    pad = 0.01; width = 0.02

    # Add the colorbar axis
    fig = plt.gcf()
    cbar_ax = fig.add_axes([x11+pad, y10, width, y01-y10])
    axcb = plt.colorbar(cax=cbar_ax, label='Counts')
    axcb.ax.tick_params(labelsize=HIST_TICK_FONTSIZE)

    # Save the plot to disk
    savePlot(plt, 'peri_vs_i', data_label)

    plt.show()
    plt.clf()

    ###

    ### e - 1/a plots

    # Plot the observed e vs 1/a 2D histogram
    ax1 = plt.subplot(131)
    vmin, vmax = plot2DHistogram(loaded_data[:, ECC_INDX], 1.0/loaded_data[:, A_INDX], 'Observed data', 
        '$e$', '$1/a$', subplot=True, x_width=e_std)

    # Set limits of the X axis
    ax1.set_xlim([0, 1])

    # Plot the synthetic e vs 1/a 2D histogram
    ax2 = plt.subplot(132, sharey=ax1)
    plot2DHistogram(e_results, inv_a_results, 'Synthetic data', '$e$', '$1/a$', subplot=True, 
        x_width=e_std, vmin=vmin, vmax=vmax)

    # Set limits of the X axis
    ax2.set_xlim([0, 1])

    # Update the Y limits to conform to the input data
    plt.ylim((min(1.0/loaded_data[:, A_INDX]), max(1.0/loaded_data[:, A_INDX])))

    # Make Y axis tick labels invisible
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_ylabel('')

    # Hide '0.0' in xticks
    labels = ax2.get_xticks().tolist()
    labels[0] = ''
    ax2.set_xticklabels(labels)


    # Plot the absolute residuals 2D histogram
    ax3 = plt.subplot(133, sharey=ax1)
    plot2DResidualHistogram(loaded_data[:, ECC_INDX], 1.0/loaded_data[:, A_INDX], e_results, 
        inv_a_results, '|Observed-synthetic| residuals', '$e$', '$1/a$', subplot=True, x_width=e_std, 
        vmin=vmin, vmax=vmax)

    # Set limits of the X axis
    ax3.set_xlim([0, 1])

    # Update the Y limits to conform to the input data
    plt.ylim((min(1.0/loaded_data[:, A_INDX]), max(1.0/loaded_data[:, A_INDX])))

    # Make Y axis tick labels invisible
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.set_ylabel('')

    # Hide '0.0' in xticks
    labels = ax3.get_xticks().tolist()
    labels[0] = ''
    ax3.set_xticklabels(labels)

    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, wspace=0.05)

    # Get plot positions
    [[x00,y00],[x01,y01]] = ax1.get_position().get_points()
    [[x10,y10],[x11,y11]] = ax3.get_position().get_points()
    pad = 0.01; width = 0.02

    # Add the colorbar axis
    fig = plt.gcf()
    cbar_ax = fig.add_axes([x11+pad, y10, width, y01-y10])
    axcb = plt.colorbar(cax=cbar_ax, label='Counts')
    axcb.ax.tick_params(labelsize=HIST_TICK_FONTSIZE)

    # Save the plot to disk
    savePlot(plt, 'e_vs_inv_a', data_label)

    plt.show()
    plt.clf()

    ###

    ### Peri-q plots

    # Plot the observed peri vs q plot
    ax1 = plt.subplot(131)
    vmin, vmax = plot2DHistogram(loaded_data[:, PERI_INDX], loaded_data[:, Q_INDX], 'Observed data', 
        '$\omega$', '$q$', subplot=True, x_width=peri_std, y_width=q_std)

    # Set limits of the X axis
    ax1.set_xlim([0, 360])


    # Plot the synthetic data peri-incl 2D histogram
    ax2 = plt.subplot(132, sharey=ax1)
    plot2DHistogram(peri_results, q_results, 'Synthetic data', '$\omega$', '$q$', subplot=True, 
        x_width=peri_std, y_width=q_std, vmin=vmin, vmax=vmax)

    # Set limits of the X axis
    ax2.set_xlim([0, 360])

    # Update the Y limits to conform to the input data
    plt.ylim((min(loaded_data[:, Q_INDX]), max(loaded_data[:, Q_INDX])))

    # Make Y axis tick labels invisible
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_ylabel('')

    
    # Plot the absolute residuals 2D histogram
    ax3 = plt.subplot(133, sharey=ax1)
    plot2DResidualHistogram(loaded_data[:, PERI_INDX], loaded_data[:, Q_INDX], peri_results, q_results, 
        '|Observed-synthetic| residuals', '$\omega$', '$q$', subplot=True, x_width=peri_std, 
        y_width=q_std, vmin=vmin, vmax=vmax)

    # Set limits of the X axis
    ax3.set_xlim([0, 360])

    # Update the Y limits to conform to the input data
    plt.ylim((min(loaded_data[:, Q_INDX]), max(loaded_data[:, Q_INDX])))

    # Make Y axis tick labels invisible
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.set_ylabel('')

    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, wspace=0.05)


    # Get plot positions
    [[x00,y00],[x01,y01]] = ax1.get_position().get_points()
    [[x10,y10],[x11,y11]] = ax3.get_position().get_points()
    pad = 0.01; width = 0.02

    # Add the colorbar axis
    fig = plt.gcf()
    cbar_ax = fig.add_axes([x11+pad, y10, width, y01-y10])
    axcb = plt.colorbar(cax=cbar_ax, label='Counts')
    axcb.ax.tick_params(labelsize=HIST_TICK_FONTSIZE)

    # Save the plot to disk
    savePlot(plt, 'peri_vs_q', data_label)

    plt.show()
    plt.clf()

    ###



def generateJopekMethodE(loaded_data, n_samples, q_std, node_std, e_std, peri_std, incl_std, show_plots=True, 
    get_more_data=False):
    """ Generate synthetic orbits using the Jopek & Bronikowska (2016) method E approach.

    Arguments:
        loaded_data: [2D ndarray] numpy array containing the input data, column indices are defined at the 
            beginning of this file in uppercase letters
        n_samples: [int] number of orbits to draw
        q_std: [float] standard deviation of the perihelion distance, used for plotting 2D histograms
        node_std: [float] standard deviation of the node, used for plotting 2D histograms
        e_std: [float] standard deviation of the eccentricity, used for plotting 2D histograms
        peri_std: [float] standard deviation of the argument of perihelion, used for plotting 2D histograms
        incl_std: [float] standard deviation of the inclination, used for plotting 2D histograms

    Keyword arguments:
        show_plots: [bool] True by default, if False the plots are not generated, only data vectors are 
            returned
        get_more_data: [bool] False by default, used internally as a flag to indicate that the function is in
            recursion, NOTE: do not set this value manually!

    Return:
        (inv_a_generated, q_generated, e_generated, peri_generated, node_generated, incl_generated): [tuple of
        ndarrays] orbital elements of generated orbits

    """

    # Calculate the fraction of orbits with geocentric ecliptic latitudes beta > 0
    positive_beta_fraction = len(loaded_data[loaded_data[:, BETA_G_INDX] > 0])/len(loaded_data)

    print('Beta > 0 fraction:', positive_beta_fraction)

    ### Independently generate each orbital element from their histograms

    # Specify number of bins for each parameter
    e_bins = 80
    peri_bins = 80
    node_bins = 80
    incl_bins = 80

    # Generate e from histogram
    e_generated = drawFromHistogram(loaded_data[:, ECC_INDX], e_bins, n_samples)
    
    # Generate peri from histogram
    peri_generated = drawFromHistogram(loaded_data[:, PERI_INDX], peri_bins, n_samples)

    # Generate node from histogram
    node_generated = drawFromHistogram(loaded_data[:, NODE_INDX], node_bins, n_samples)

    # Generate inclination from histogram
    incl_generated = drawFromHistogram(loaded_data[:, INCL_INDX], incl_bins, n_samples)

    ###

    ### Calculate q from the results
    
    # 1 AU (set for constant, an improvement would be for it to vary with solar longitude)
    au_dist = 1.0

    # Find the sign of the e*cosw expression in respect to the beta fraction
    beta_signs = [-1 if entry < positive_beta_fraction else 1 for entry in np.random.random(n_samples)]
    beta_signs = np.array(beta_signs)

    # Calculate q
    q_generated = au_dist*(1.0 + beta_signs*e_generated*np.cos(np.radians(peri_generated)))/(1.0 + \
        e_generated)

    ###

    # Calculate 1/a
    inv_a_generated = (1 - e_generated)/q_generated

    # Filter out all entries where the generated 1/a is larger than the maximum input 1/a
    good_indices = np.where(inv_a_generated <= max(1.0/loaded_data[:, A_INDX]))[0]
    
    inv_a_generated = inv_a_generated[good_indices]
    q_generated = q_generated[good_indices]
    e_generated = e_generated[good_indices]
    peri_generated = peri_generated[good_indices]
    node_generated = node_generated[good_indices]
    incl_generated = incl_generated[good_indices]

    # Generate some more data to replace the rejected orbits:
    if (len(good_indices) < n_samples) and not get_more_data:

        # Calculate the number of missing orbits
        missing_orbits_n = n_samples - len(good_indices)

        # Generate new synthetic orbits
        inv_a_more, q_more, e_more, peri_more, node_more, incl_more = generateJopekMethodE(loaded_data, 
            n_samples, q_std, node_std, e_std, peri_std, incl_std, show_plots=False, get_more_data=True)

        # Fill the rejected orbits with new ones, randomly generated
        random_choice = np.random.randint(0, len(q_more)-1, missing_orbits_n)
        inv_a_generated = np.r_[inv_a_generated, inv_a_more[random_choice]]
        q_generated = np.r_[q_generated, q_more[random_choice]]
        e_generated = np.r_[e_generated, e_more[random_choice]]
        peri_generated = np.r_[peri_generated, peri_more[random_choice]]
        node_generated = np.r_[node_generated, node_more[random_choice]]
        incl_generated = np.r_[incl_generated, incl_more[random_choice]]


    if show_plots:

        # Plot all results
        plotAllResults(loaded_data, inv_a_generated, q_generated, e_generated, peri_generated, node_generated,
            incl_generated, q_std, e_std, peri_std, node_std, incl_std, data_label='jopek')


    return inv_a_generated, q_generated, e_generated, peri_generated, node_generated, incl_generated



def generateKDEOrbits(loaded_data, n_samples, bandwidth, q_std, node_std, e_std, peri_std, incl_std, 
    covariance=None, show_plots=True, get_more_data=False):
    """ Implementation of the KDE-based orbit generation. 
    
    Arguments:
        loaded_data: [2D ndarray] numpy array containing the input data, column indices are defined at the 
            beginning of this file in uppercase letters
        n_samples: [int] number of orbits to draw
        bandwidth: [float] a number by which the covariance matrix will be multipled with
        q_std: [float] standard deviation of the perihelion distance, used for plotting 2D histograms
        node_std: [float] standard deviation of the node, used for plotting 2D histograms
        e_std: [float] standard deviation of the eccentricity, used for plotting 2D histograms
        peri_std: [float] standard deviation of the argument of perihelion, used for plotting 2D histograms
        incl_std: [float] standard deviation of the inclination, used for plotting 2D histograms

    Keyword arguments:
        covariance: [2D ndarray] None by default, custom covarience matrix
        show_plots: [bool] True by default, if False the plots are not generated, only data vectors are 
            returned
        get_more_data: [bool] False by default, used internally as a flag to indicate that the function is in
            recursion, NOTE: do not set this value manually!

    Return:
        (inv_a_results, q_results, e_results, peri_results, node_results, incl_results): [tuple of ndarrays] 
            orbital elements of generated orbits

    """

    # Extract individual parameters from the loaded data and rescale them
    inv_a_array = 1.0/loaded_data[:, A_INDX]
    q_array = loaded_data[:, Q_INDX]
    node_array = np.radians(loaded_data[:, NODE_INDX])
    e_array = loaded_data[:, ECC_INDX]
    peri_array = np.radians(loaded_data[:, PERI_INDX])
    incl_array = np.radians(loaded_data[:, INCL_INDX])

    # Create a matrix of the input parameters
    kde_matrix = np.column_stack((q_array, node_array, e_array, peri_array, incl_array))

    # Create the Gaussian kernel density estimate
    kde_kernel = stat.gaussian_kde(kde_matrix.T)

    # Make a covariance matrix if none was given
    if covariance is None:

        # Set the covariance matrix
        kde_kernel.covariance = bandwidth*np.array(
            [[q_std,    0,       0,       0,        0],
            [   0,   np.radians(node_std),   0,       0,        0],
            [   0,      0,     np.radians(e_std),     0,        0],
            [   0,      0,       0,   np.radians(peri_std),     0],
            [   0,      0,       0,       0,    np.radians(incl_std)]])**2

    else:

        # Use the covariance matrix if it was given
        kde_kernel.covariance = bandwidth*covariance

    # Set the inverse of the covariance
    kde_kernel.inv_cov = np.linalg.inv(kde_kernel.covariance)

    # Draw sample orbits from the KDE
    kde_results = kde_kernel.resample(n_samples)

    # Extract individual parameter arrays
    q_results, node_results, e_results, peri_results, incl_results = np.hsplit(kde_results.T, 5)

    # Convert the input to degees
    node_array = np.degrees(node_array)
    peri_array = np.degrees(peri_array)
    incl_array = np.degrees(incl_array)

    # Convert results to degrees (and wrap to 360)
    node_results = np.degrees(node_results) % 360
    peri_results = np.degrees(peri_results) % 360

    # Convert the inclination to degrees, and mirror the edges
    incl_results = np.degrees(incl_results)
    incl_results[incl_results < 0] *= -1
    incl_results[incl_results > 180] = 360 - incl_results[incl_results > 180]

    # Calculate 1/a from the results
    inv_a_results = (1.0 - e_results)/q_results

    ### Cut the resulting range to the input range
    input_names = ['1/a', 'q', 'node', 'e', 'peri', 'incl']
    input_list = [inv_a_array, q_array, node_array, e_array, peri_array, incl_array]
    results_list = [inv_a_results, q_results, node_results, e_results, peri_results, incl_results]

    for i, result in enumerate(results_list):

        # Find the input min and max
        min_input, max_input = min(input_list[i]), max(input_list[i])

        # # Print the range of the input data
        # print('Parameter:', input_names[i], 'Min:', min_input, 'Max:', max_input)

        # Find the indices of the results that are inside the input range
        inside_range_indices = np.where((result >= min_input) & (result <= max_input))

        # Filter every dimension
        for j in range(len(results_list)):
            results_list[j] = results_list[j][inside_range_indices]

    # Unpack the filtered results
    inv_a_results, q_results, node_results, e_results, peri_results, incl_results = results_list

    ###


    # Generate some more data to replace the rejected orbits:
    if (len(q_results) < n_samples) and not get_more_data:

        # print('Rejected orbits percent:', (1.0 - len(q_results)/n_samples)*100.0)

        # Calculate the number of missing orbits
        missing_orbits_n = n_samples - len(q_results)

        # Generate new synthetic orbits
        inv_a_more, q_more, e_more, peri_more, node_more, incl_more = generateKDEOrbits(loaded_data, 
            n_samples, bandwidth, q_std, node_std, e_std, peri_std, incl_std, show_plots=False, 
            get_more_data=True)

        # Fill the rejected orbits with new ones, which are randomly chosen
        random_choice = np.random.randint(0, len(q_more) - 1, missing_orbits_n)
        inv_a_results = np.r_[inv_a_results, inv_a_more[random_choice]]
        q_results = np.r_[q_results, q_more[random_choice]]
        e_results = np.r_[e_results, e_more[random_choice]]
        peri_results = np.r_[peri_results, peri_more[random_choice]]
        node_results = np.r_[node_results, node_more[random_choice]]
        incl_results = np.r_[incl_results, incl_more[random_choice]]


    if show_plots:

        ### Discern what file name this data should have

        # If the bandwidth is a scalar, use that scalar as a label
        if np.isscalar(bandwidth):
            bandwidth_label = str(bandwidth).replace('.', '_')

        # If bandwidth is a matrix, mark it as nonscalar and use the sum of the diagonal as the label
        else:
            bandwidth_label = 'nonscalar_' + str(np.sum(bandwidth.diagonal())).replace('.', '_')

        ###

        # Plot all results
        plotAllResults(loaded_data, inv_a_results, q_results, e_results, peri_results, node_results,
            incl_results, q_std, e_std, peri_std, node_std, incl_std, 
            data_label='kde_bandwidth_'+str(bandwidth_label))


    return inv_a_results, q_results, e_results, peri_results, node_results, incl_results



def hist2dDiff(original_data, synthetic_data, x_std, y_std, x_index, y_index):
    """ Calculate the difference between histograms of the original and simulated data. 

    The bin widths are chosen to be similar to the standard deviation of the data.

    Arguments:
        original_data: [2D ndarray] 2D numpy array containing the observed data
        synthetic_data: [2D ndarray] 2D numpy array containing synthetic data
        x_std: [float] standard deviation of the axis X data
        y_std: [float] standard deviation of the axis Y data
        x_index: [int] column index of the X axis orbital element in the data
        y_index: [int] column index of the Y axis orbital element in the data
    
    Return:
        [float] mean top 1% absolute differences between the observed and synthetic data histograms

    """

    ### Calculate the number of bins

    # Get the data range
    x_min, x_max = min(original_data[:, x_index]), max(original_data[:, x_index])
    y_min, y_max = min(original_data[:, y_index]), max(original_data[:, y_index])

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Calculate the number of bins used
    x_nbins = int(x_range/x_std)
    y_nbins = int(y_range/y_std)

    ###

    # Create a histogram of the original data
    orig_H, _, _ = np.histogram2d(original_data[:, x_index], original_data[:, y_index], bins=[x_nbins, \
        y_nbins])

    # Create a histogram of the synthetic data
    synt_H, _, _  = np.histogram2d(synthetic_data[:, x_index], synthetic_data[:, y_index], bins=[x_nbins, \
        y_nbins])

    # How many percent of total points to take to estimate the average difference
    top_n_percent = 0.01

    top_n = int(x_nbins*y_nbins*(1.0 - top_n_percent))

    # Calculate the mean of top 1% absolute differences
    return np.mean(np.sort(np.abs(orig_H - synt_H).ravel())[top_n:])



def getHistDiff(loaded_data, n_samples, bandwidth, q_std, node_std, e_std, peri_std, incl_std):
    """ Generate KDE orbits using the given bandwidth and calculate the mean top 1% absolute histogram 
        difference. 
    
    Arguments:
        loaded_data: [2D ndarray] numpy array containing the input data, column indices are defined at the 
            beginning of this file in uppercase letters
        n_samples: [int] number of orbits to draw
        bandwidth: [float] a number by which the covariance matrix will be multipled with
        q_std: [float] standard deviation of the perihelion distance, used for plotting 2D histograms
        node_std: [float] standard deviation of the node, used for plotting 2D histograms
        e_std: [float] standard deviation of the eccentricity, used for plotting 2D histograms
        peri_std: [float] standard deviation of the argument of perihelion, used for plotting 2D histograms
        incl_std: [float] standard deviation of the inclination, used for plotting 2D histograms
    
    Return:
        [float] mean top 1% absolute differences between the observed and synthetic data histograms

    """

    # Extract individual parameters from the loaded data
    q_array = loaded_data[:, Q_INDX]
    node_array = loaded_data[:, NODE_INDX]
    e_array = loaded_data[:, ECC_INDX]
    peri_array = loaded_data[:, PERI_INDX]
    incl_array = loaded_data[:, INCL_INDX]

    # Make the 2D observed data array
    original_data_array = np.column_stack([q_array, e_array, peri_array, node_array, incl_array])

    # Draw KDE orbits
    inv_a_generated, q_generated, e_generated, peri_generated, node_generated, \
        incl_generated = generateKDEOrbits(loaded_data, n_samples, bandwidth, q_std, node_std, e_std, 
            peri_std, incl_std, show_plots=False)

    # Make the 2D synthetic data array
    synthetic_data_array = np.column_stack([q_generated, e_generated, peri_generated, node_generated, 
        incl_generated])

    # Set the histogram comparison axes as peri and q
    peri_index = 2 # peri index
    q_index = 0 # q index

    # Calculate the histogram difference
    return hist2dDiff(original_data_array, synthetic_data_array, peri_std, q_std, peri_index, q_index)



def estimateBandwidth(loaded_data, n_samples, q_std, node_std, e_std, peri_std, incl_std):
    """ Run the KDE for a range of bandwidths and calculate top histogram differences for each bandwidth. 
        
    Arguments:
        loaded_data: [2D ndarray] numpy array containing the input data, column indices are defined at the 
            beginning of this file in uppercase letters
        n_samples: [int] number of orbits to draw
        q_std: [float] standard deviation of the perihelion distance, used for plotting 2D histograms
        node_std: [float] standard deviation of the node, used for plotting 2D histograms
        e_std: [float] standard deviation of the eccentricity, used for plotting 2D histograms
        peri_std: [float] standard deviation of the argument of perihelion, used for plotting 2D histograms
        incl_std: [float] standard deviation of the inclination, used for plotting 2D histograms
    
    Return:
        None
    """

    # Create a log distribution of bandwidth points from 10.0 to 0.01
    bandwidth_range = np.logspace(1, -2, 20)

    # Extract individual parameters from the loaded data and rescale them
    q_array = loaded_data[:, Q_INDX]
    node_array = loaded_data[:, NODE_INDX]
    e_array = loaded_data[:, ECC_INDX]
    peri_array = loaded_data[:, PERI_INDX]
    incl_array = loaded_data[:, INCL_INDX]

    # Make the 2D data arrays
    original_data_array = np.column_stack([q_array, e_array, peri_array, node_array, incl_array])

    histogram_differences = []

    # Go thourgh the given bandwidths
    for bandwidth in bandwidth_range:

        # Repeat 10 times and take the average
        temp_hist_diff = []
        
        for i in range(10):

            # Sample orbits from KDE
            inv_a_generated, q_generated, e_generated, peri_generated, node_generated, \
                incl_generated = generateKDEOrbits(loaded_data, n_samples, bandwidth, q_std, node_std, e_std, 
                    peri_std, incl_std, show_plots=False)


            synthetic_data_array = np.column_stack([q_generated, e_generated, peri_generated, node_generated, 
                incl_generated])

            # Set the histogram comparison axes as peri and q
            peri_index = 2
            q_index = 0 # q index

            # Calculate the histogram difference
            temp_hist_diff.append(hist2dDiff(original_data_array, synthetic_data_array, peri_std, q_std, 
                peri_index, q_index))


        histogram_differences.append(np.mean(temp_hist_diff))
    
    print('Bandwidths:', bandwidth_range)
    print('Histogram diffs:', histogram_differences)

    # Save the numpy arrays to disk
    np.save('bandwidth_range', np.array(bandwidth_range))
    np.save('histogram_differences', np.array(histogram_differences))


    plt.plot(bandwidth_range, histogram_differences)

    # Flip the X axis
    plt.gca().invert_xaxis()

    # Set X axis as log
    plt.gca().set_xscale('log')

    # Set axes labels
    plt.xlabel('Bandwidth')
    plt.ylabel('Mean of top 1% absolute histogram differences')

    plt.grid()

    plt.show()



def calculateNN(loaded_data, n_samples, q_std, node_std, e_std, peri_std, incl_std):
    """ Calculate the average nearest neightbour distance of the synthetic dataset for a range of bandwidths.
    
    Arguments:
        loaded_data: [2D ndarray] numpy array containing the input data, column indices are defined at the 
            beginning of this file in uppercase letters
        n_samples: [int] number of orbits to draw
        q_std: [float] standard deviation of the perihelion distance, used for plotting 2D histograms
        node_std: [float] standard deviation of the node, used for plotting 2D histograms
        e_std: [float] standard deviation of the eccentricity, used for plotting 2D histograms
        peri_std: [float] standard deviation of the argument of perihelion, used for plotting 2D histograms
        incl_std: [float] standard deviation of the inclination, used for plotting 2D histograms

    Return:
        None
    """

    # Check the nn for a range of bandwidths (from 10.0 to 0.01)
    bandwidth_range = np.logspace(1, -2, 20)

    nn_results = []
    for bandwidth in bandwidth_range:

        # Draw orbits from the KDE
        inv_a_generated, q_generated, e_generated, peri_generated, node_generated, \
            incl_generated = generateKDEOrbits(loaded_data, n_samples, bandwidth, q_std, node_std, e_std, 
                peri_std, incl_std, show_plots=False)

        # Make the 2D data arrays
        synthetic_data_array = np.column_stack([q_generated, e_generated, incl_generated, node_generated, 
            peri_generated])

        
        # Find the average NN distance
        synthetic_nn = findNearestNeighborDistDSH(synthetic_data_array)
        print('Synthetic NN', synthetic_nn)

        nn_results.append(synthetic_nn)

    print('---------------')
    print('Bandwidths:', bandwidth_range)
    print('NN results:', nn_results)

    # Save the numpy arrays to disk
    np.save('bandwidth_range', np.array(bandwidth_range))
    np.save('nn_results', np.array(nn_results))

    # Flip the X axis
    plt.gca().invert_xaxis()

    # Set X axis as log
    plt.gca().set_xscale('log')

    plt.plot(bandwidth_range, nn_results)
    plt.xlabel('Bandwidth')
    plt.ylabel('Average nearest neighbor, $D_{SH}$')

    plt.show()
    plt.clf()
    plt.close()



def estimateBandwidthMatrixWithNN(loaded_data, n_samples, q_std, node_std, e_std, peri_std, incl_std):
    """ Test a range of q and peri bandwidths and find the mean NN distance and histogram difference for each. 
    
    Arguments:
        loaded_data: [2D ndarray] numpy array containing the input data, column indices are defined at the 
            beginning of this file in uppercase letters
        n_samples: [int] number of orbits to draw
        q_std: [float] standard deviation of the perihelion distance, used for plotting 2D histograms
        node_std: [float] standard deviation of the node, used for plotting 2D histograms
        e_std: [float] standard deviation of the eccentricity, used for plotting 2D histograms
        peri_std: [float] standard deviation of the argument of perihelion, used for plotting 2D histograms
        incl_std: [float] standard deviation of the inclination, used for plotting 2D histograms

    Return:
        None
        """

    # Extract individual parameters from the loaded data and rescale them
    q_array = loaded_data[:, Q_INDX]
    node_array = loaded_data[:, NODE_INDX]
    e_array = loaded_data[:, ECC_INDX]
    peri_array = loaded_data[:, PERI_INDX]
    incl_array = loaded_data[:, INCL_INDX]

    # Make the 2D data arrays
    original_data_array = np.column_stack([q_array, e_array, peri_array, node_array, incl_array])


    # Check the NN for a range of q bandwidths
    q_bandwidth_range = np.logspace(1, -3, 40)

    # Check the NN for a range of peri bandwidths
    peri_bandwidth_range = np.logspace(1, -1, 40)

    # Form the results matrix
    nn_results_matrix = np.zeros((len(q_bandwidth_range), len(peri_bandwidth_range)))

    # Histogram differences matrix
    hist_diff_matrix = np.zeros((len(q_bandwidth_range), len(peri_bandwidth_range)))

    # Bandwidth for other parameters
    bandwidth = 10.0

    for i, q_bandwidth in enumerate(q_bandwidth_range):

        for j, peri_bandwidth in enumerate(peri_bandwidth_range):

            # Init the bandwidth matrix
            bandwidth_matrix = np.array(
                [[q_bandwidth,    0,       0,       0,        0], # q
                [   0,   bandwidth,   0,       0,        0], # node
                [   0,      0,     bandwidth,     0,        0], # e
                [   0,      0,       0,   peri_bandwidth,     0], # peri
                [   0,      0,       0,       0,    bandwidth]]) # i

            # Run the new method
            inv_a_generated, q_generated, e_generated, peri_generated, node_generated, \
                incl_generated = generateKDEOrbits(loaded_data, n_samples, bandwidth_matrix, q_std, node_std, 
                    e_std, peri_std, incl_std, show_plots=False)

            # Create a synthetic data srray
            synthetic_data_array = np.column_stack([q_generated, e_generated, peri_generated, node_generated, 
                incl_generated])

            # Check the nearest neighbor distance
            synthetic_nn = findNearestNeighborDistDSH(synthetic_data_array)

            # Save the results to a final matrix
            nn_results_matrix[i,j] = synthetic_nn

            print('---------')
            print('q bandwidth:', q_bandwidth, 'peri bandwidth:', peri_bandwidth)
            print('Mean NN:', synthetic_nn)

            # Set the histogram comparison axes as peri and q
            peri_index = 2
            q_index = 0 # q index

            # Calculate the histogram difference
            hist_diff = hist2dDiff(original_data_array, synthetic_data_array, peri_std, q_std, 
                peri_index, q_index)

            # Save the result to the histogram matrix
            hist_diff_matrix[i,j] = hist_diff

            print('Hist diff:', hist_diff)



    print('RESULTS:')
    print('Y axis: q bandwidth range:', q_bandwidth_range)
    print('X axis: Peri bandwidth range:', peri_bandwidth_range)
    print('NN matrix:')
    print(nn_results_matrix)
    print('Histogram difference:')
    print(hist_diff_matrix)

    # Save the results to files
    np.save('q_bandwidths', q_bandwidth_range)
    np.save('peri_bandwidths', peri_bandwidth_range)
    np.save('nn_results_matrix', nn_results_matrix)
    np.save('hist_diff_matrix', hist_diff_matrix)



def savePlot(plt, file_name, data_label=''):
    """ Saves the plot to disk.

    Arguments:
        plt: [plot handle]
        file_name: [str] name of the file where the plot will be saved

    Keyword arguments:
        data_label: [str] add this prefix in front of each name of the file that is saved to disk

    Return:
        None

    """

    if data_label != '':
        data_label += '_'

    # Save the plot to disk
    plt.savefig(data_label + file_name.replace('$', '')+'.png', dpi=300)



# Main part of the program
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


    # Define the number of orbits to draw from the KDE (the same as the number of observed input orbits)
    n_samples = len(loaded_data)


    # Calculate means of standard deviations
    q_std = np.mean(loaded_data[:, Q_STD_INDX])
    e_std = np.mean(loaded_data[:, ECC_STD_INDX])
    incl_std = np.mean(loaded_data[:, INCL_STD_INDX])
    peri_std = np.mean(loaded_data[:, PERI_STD_INDX])
    
    node_std = np.mean(loaded_data[:, NODE_STD_INDX])

    # # Uncomment this part if you want to see how node stddev changes with inclination
    # # Inclination vs. node stddev plot
    # plt.scatter(loaded_data[:, INCL_INDX], loaded_data[:, NODE_STD_INDX])
    # plt.xlabel('Inclination')
    # plt.ylabel('Node SD')
    # plt.xlim([-1, 181])
    # plt.ylim([0, np.max(loaded_data[:, NODE_STD_INDX])])
    # plt.show()
    # plt.clf()

    # # Filter nodes std's by inclination (low inclinations have huge deviations in node, so take only those
    # # in the 20 deg to 160 deg range for standard deviation calculation)
    # node_std_filtered = loaded_data[:, NODE_STD_INDX][np.where((loaded_data[:, INCL_INDX] > 20) & \
    #    (loaded_data[:, INCL_INDX] < 160))]
    # node_std = np.mean(node_std_filtered)


    print('Standard deviations, mean:')
    print('q', q_std)
    print('e', e_std)
    print('incl', incl_std)
    print('node', node_std)
    print('peri', peri_std)
    print('-----')


    print('Jopek method results')

    # Generate synthetic sporadic orbits using Jopek & Bronikowska (2016) method E
    inv_a_jopek, q_jopek, e_jopek, peri_jopek, node_jopek, \
        incl_jopek = generateJopekMethodE(loaded_data, len(loaded_data), q_std, node_std, e_std, peri_std, 
            incl_std)


    # # Uncomment if you wish to calculate the nearest neighbour distance of orbits generated using Method E
    # jopek_data_array = np.column_stack([q_jopek, e_jopek, incl_jopek, node_jopek, peri_jopek])
    # jopek_nn = findNearestNeighborDistDSH(jopek_data_array)
    # print('Jopek NN:', jopek_nn)


    # Extract individual parameters from the loaded data
    q_array = loaded_data[:, Q_INDX]
    node_array = loaded_data[:, NODE_INDX]
    e_array = loaded_data[:, ECC_INDX]
    peri_array = loaded_data[:, PERI_INDX]
    incl_array = loaded_data[:, INCL_INDX]



    ### UNCOMMENT IF YOU WISH TO GENERATE THE PLOT WHICH SHOWS THE INFLUENCE OF DIFFERENT BANDWIDTHS ON
    ### HISTOGRAM DIFFERENCES OR NEAREST NEIGHBOUR VALUE - ONLY FOR SCALAR MATRIX BANDWIDTH

    # # Run bandwidth estimation
    # estimateBandwidth(loaded_data, n_samples, q_std, node_std, e_std, peri_std, incl_std)

    # # Calculate Nearest Neighbor distances for a variety of bandwidths
    # calculateNN(loaded_data, n_samples, q_std, node_std, e_std, peri_std, incl_std)

    # sys.exit()

    ###


    ### UNCOMMENT IF YOU WANT TO GENERATE THE 2D HISTOGRAM PLOT OF q AND peri BANDWIDTHS INFLUENCE ON
    ### HISTOGRAM DIFFERENCES OR NEAREST NEIGHBOUR VALUE

    ### WARNING: This calculation can take >24 hours, depending on the system
    # # Check a range of q and peri bandwidths to see how the mean NN and histogram difference behaves
    # estimateBandwidthMatrixWithNN(loaded_data, n_samples, q_std, node_std, e_std, peri_std, incl_std)

    # sys.exit()

    ###



    # Define the nonscalar bandwidth matrix
    bandwidth = 10

    bandwidth_matrix = np.array(
        [[0.01,    0,       0,       0,        0], # q
        [   0,   bandwidth,   0,       0,        0], # node
        [   0,      0,     bandwidth,     0,        0], # e
        [   0,      0,       0,   6,     0], # peri
        [   0,      0,       0,       0,    bandwidth]]) # i


    # Make a list of bandwidths to go through
    bandwidth_list = [10, 0.1, bandwidth_matrix]

    for bandwidth in bandwidth_list:

        print('KDE, bandwidth:', bandwidth)

        # Draw KDE orbits
        inv_a_generated, q_generated, e_generated, peri_generated, node_generated, \
            incl_generated = generateKDEOrbits(loaded_data, n_samples, bandwidth, q_std, node_std, e_std, 
                peri_std, incl_std)

        
        original_data_array = np.column_stack([q_array, e_array, incl_array, node_array, peri_array])
        synthetic_data_array = np.column_stack([q_generated, e_generated, incl_generated, node_generated, 
            peri_generated])

        # # UNCOMMENT if you want to find nearest neighbors distance (WARNING: it takes a while)
        # original_nn = findNearestNeighborDistDSH(original_data_array)
        # print('Observed NN:', original_nn)

        # # Check the nearest neighbor distance
        
        # synthetic_nn = findNearestNeighborDistDSH(synthetic_data_array)
        # print('Synthetic NN', synthetic_nn)

        # # Find the average minimum DSH between the original and synthetic data
        # avg_min_dsh = findMinimumDSH(original_data_array, synthetic_data_array)

        # print('Average minimum DSH::', avg_min_dsh)

        # Calculate the histogram difference
        hist_diff = getHistDiff(loaded_data, n_samples, bandwidth, q_std, node_std, e_std, peri_std, incl_std)
        print('Histogram diff:', hist_diff)

        # # Print out the results
        # print('   1/a,    Q,    Node,   e,    peri,   incl')
        # for line in np.hstack([inv_a_results, q_results, node_results, e_results, peri_results, incl_results]):
        #     inv_a, q, node, e, peri, incl = line

        #     pattern = "%.3f"
        #     print([float(pattern % i) for i in (inv_a, q, node, e, peri, incl)])



