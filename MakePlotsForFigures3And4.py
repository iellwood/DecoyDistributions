"""
Generates Figure panels from Figures 3 & 4 as well as the FID score plot from Figure 6C from
"Validating density estimation models with decoy distributions".

Note that the figure legends, titles and axis lables were reformatted for the paper as matplotlib does a poor job
of rendering them.

N.B. When computing the slope between log_p_decoy and log_p_model, outliers from log_p_model were discarded by
removing points more than 3 sigma from the mean of log_p_model. Without doing this, a few of the linear fits are
distorted by outliers. Without removing outliers, a few of the models appear to be uncorrelated with their respective
decoy distribution even though they are secretly well correlated with log_p_decoy for most samples.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import Util.prettyplot as prettyplot
import scipy.stats as stats
import seaborn as sb

MNIST_decoy_vs_FlowGAN_filename = 'DataForFigurePlots/Data_FLOWGAN/ThermalRegressionPlotData_MNIST.pickle'
MNIST_decoy_vs_FlowGAN_matched_width_filename = 'DataForFigurePlots/Data_FLOWGAN/ThermalRegressionPlotData_MNIST_MatchedSeed.pickle'
CIFAR10_decoy_vs_FlowGAN_filename = 'DataForFigurePlots/Data_FLOWGAN/ThermalRegressionPlotData_CIFAR10.pickle'

MNIST_decoy_vs_GLOW_filename = 'DataForFigurePlots/Data_GLOW/ThermalRegressionPlotData_GLOW_MNIST.pickle'
CIFAR10_decoy_vs_GLOW_filename = 'DataForFigurePlots/Data_GLOW/ThermalRegressionPlotData_GLOW_CIFAR10.pickle'


def get_data_from_file(file_name, channels=1):
    file_handle = open(file_name, 'rb')
    A = pickle.load(file_handle)
    file_handle.close()

    image_widths = A['image_widths']
    log_likelihood_widths = A['log_likelihood_widths']
    data = A['data']

    pvalues = []
    slopes = []
    intercepts = []
    intercepts_divided_by_d = []
    r_squared_values = []
    stderrors = []
    KL_divergences = []
    log_likelihood_width_divided_by_d = []


    for image_width_index, image_width in enumerate(image_widths):
        slopes.append([])
        intercepts.append([])
        intercepts_divided_by_d.append([])

        d = image_width**2*channels

        r_squared_values.append([])
        stderrors.append([])
        KL_divergences.append([])
        log_likelihood_width_divided_by_d.append([])
        pvalues.append([])


        for log_likelihood_width_index, log_likelihood_width in enumerate(log_likelihood_widths[image_width_index]):
            log_p_decoy = data[image_width_index][log_likelihood_width_index][0]
            log_p_model = data[image_width_index][log_likelihood_width_index][1]

            # NOTE: Outliers are removed via this test
            I = np.abs(log_p_model - np.mean(log_p_model))/np.std(log_p_model) < 3

            # Remove the [I] from both arguments to see the plots without removing outliers.
            res = stats.linregress(log_p_decoy[I], log_p_model[I])


            pvalues[image_width_index].append(res.pvalue)

            slopes[image_width_index].append(res.slope)
            intercepts[image_width_index].append(res.intercept)
            intercepts_divided_by_d[image_width_index].append(res.intercept/d)

            r_squared_values[image_width_index].append(res.rvalue**2)
            stderrors[image_width_index].append(res.stderr)

            KL_divergences[image_width_index].append(np.mean(log_p_decoy - log_p_model)/d)

            log_likelihood_width_divided_by_d[image_width_index].append(log_likelihood_width/d)

    return_dict = {
        'data': data,
        'image_width': image_widths,
        'log_likelihood_width': log_likelihood_widths,
        'slope': slopes,
        'intercept': intercepts,
        'pvalue': pvalues,
        'intercept_divided_by_d': intercepts_divided_by_d,
        'r_squared': r_squared_values,
        'stderror': stderrors,
        'KL_divergence': KL_divergences,
        'log_likelihood_width_divided_by_d': log_likelihood_width_divided_by_d
    }

    return return_dict


def make_plot(data_dict, x_axis, y_axis, widths_to_plot=None, label_appended_str='', threshold_variable=None, threshold_value=0, scatter=False,color_offset=0, x_log=True, y_log=False, horizontal_line=None, xlabel=None, ylabel=None, title=None):
    i = 0
    for image_width_index, image_width in enumerate(data_dict['image_width']):
        plot_this_width=True
        if widths_to_plot is not None:
            if image_width not in widths_to_plot:
                plot_this_width = False
        if plot_this_width:
            xs = np.array(data_dict[x_axis][image_width_index])
            ys = np.array(data_dict[y_axis][image_width_index])
            if threshold_variable is not None:
                threshold_values = np.array(data_dict[threshold_variable][image_width_index])
                I = threshold_values > threshold_value
                #print(threshold_values, threshold_value)
                xs = xs[I]
                ys = ys[I]
            w = str(image_width)
            if scatter:
                plt.scatter(xs, ys, marker='o',
                         label=(w + ' x ' + w + label_appended_str),
                         color=prettyplot.color_list[i + color_offset])
            else:
                plt.plot(xs, ys, marker='o', label=(w + ' x ' + w + label_appended_str),
                         color=prettyplot.color_list[i + color_offset])
            i = i + 1


    if x_log: plt.gca().set_xscale('log')
    if y_log: plt.gca().set_yscale('log')
    if horizontal_line is not None:
        plt.axhline(horizontal_line, color=prettyplot.colors['black'], linestyle='dashed')
    plt.legend(frameon=False)
    prettyplot.no_box()
    if title is not None: prettyplot.title(title)
    if xlabel is not None: prettyplot.xlabel(xlabel)
    if ylabel is not None: prettyplot.ylabel(ylabel)

# Make the plots
decoy_vs_FlowGAN_MNIST_data_dict = get_data_from_file(MNIST_decoy_vs_FlowGAN_filename)
decoy_vs_FlowGAN_matched_width_MNIST_data_dict = get_data_from_file(MNIST_decoy_vs_FlowGAN_matched_width_filename)
decoy_vs_FlowGAN_CIFAR10_data_dict = get_data_from_file(CIFAR10_decoy_vs_FlowGAN_filename, channels=3)

decoy_vs_GLOW_MNIST_data_dict = get_data_from_file(MNIST_decoy_vs_GLOW_filename)
decoy_vs_GLOW_CIFAR10_data_dict = get_data_from_file(CIFAR10_decoy_vs_GLOW_filename, channels=3)

if True:
    # Density of log-likelihoods plot MNIST
    fig, axes = plt.subplots(1,3, figsize=[13, 4])

    d = 32*32
    m = 0
    log_likelihood_width = decoy_vs_FlowGAN_MNIST_data_dict['log_likelihood_width'][3][m]
    axes[0].hist(decoy_vs_FlowGAN_MNIST_data_dict['data'][3][m][0]/d, 50, density=True, histtype='step', color=prettyplot.colors['black'], label='decoy', linewidth=2)
    axes[0].hist(decoy_vs_FlowGAN_MNIST_data_dict['data'][3][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['blue'], label='FlowGAN', linewidth=2)
    axes[0].hist(decoy_vs_FlowGAN_matched_width_MNIST_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['orange'], label='FlowGAN (matched seed)', linewidth=2)
    axes[0].hist(decoy_vs_GLOW_MNIST_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['red'], label='GLOW', linewidth=2)
    prettyplot.x_axis_only(axes[0])
    prettyplot.title('distribution of log-likelihoods, W = ' + str(log_likelihood_width), axes[0])
    axes[0].set_xlim([2, 8])
    axes[0].set_ylim([0, 3])

    d = 32*32
    m = 2
    log_likelihood_width = decoy_vs_FlowGAN_MNIST_data_dict['log_likelihood_width'][3][m]
    axes[1].hist(decoy_vs_FlowGAN_MNIST_data_dict['data'][3][m][0]/d, 50, density=True, histtype='step', color=prettyplot.colors['black'], label='decoy', linewidth=2)
    axes[1].hist(decoy_vs_FlowGAN_MNIST_data_dict['data'][3][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['blue'], label='FlowGAN', linewidth=2)
    axes[1].hist(decoy_vs_FlowGAN_matched_width_MNIST_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['orange'], label='FlowGAN (matched seed)', linewidth=2)
    axes[1].hist(decoy_vs_GLOW_MNIST_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['red'], label='GLOW', linewidth=2)
    prettyplot.x_axis_only(axes[1])
    prettyplot.title('distribution of log-likelihoods, W = ' + str(log_likelihood_width), axes[1])
    axes[1].set_xlim([2, 8])
    axes[1].set_ylim([0, 3])

    d = 32*32
    m = 4
    log_likelihood_width = decoy_vs_FlowGAN_MNIST_data_dict['log_likelihood_width'][3][m]
    axes[2].hist(decoy_vs_FlowGAN_MNIST_data_dict['data'][3][m][0]/d, 50, density=True, histtype='step', color=prettyplot.colors['black'], label='decoy', linewidth=2)
    axes[2].hist(decoy_vs_FlowGAN_MNIST_data_dict['data'][3][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['blue'], label='FlowGAN', linewidth=2)
    axes[2].hist(decoy_vs_FlowGAN_matched_width_MNIST_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['orange'], label='FlowGAN (matched seed)', linewidth=2)
    axes[2].hist(decoy_vs_GLOW_MNIST_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['red'], label='GLOW', linewidth=2)
    prettyplot.x_axis_only(axes[2])
    prettyplot.title('distribution of log-likelihoods, W = ' + str(log_likelihood_width), axes[2])
    axes[2].legend(frameon=False)
    axes[2].set_xlim([2, 8])
    axes[2].set_ylim([0, 3])
    prettyplot.xlabel('log_likelihood / d', axes[2])

    plt.show()

    # plt.savefig('PlotSaves/DensityOfLogLikelihoods_MNIST.pdf', format='pdf', transparent=True)


    # ***********************************************************************************************************

    # Density of log-likelihoods plot CIFAR10
    fig, axes = plt.subplots(1,3, figsize=[13, 4])

    d = 32*32*3
    m = 0
    log_likelihood_width = decoy_vs_FlowGAN_CIFAR10_data_dict['log_likelihood_width'][0][m]
    axes[0].hist(decoy_vs_FlowGAN_CIFAR10_data_dict['data'][0][m][0]/d, 50, density=True, histtype='step', color=prettyplot.colors['black'], label='decoy', linewidth=2)
    axes[0].hist(decoy_vs_FlowGAN_CIFAR10_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['blue'], label='FlowGAN', linewidth=2)
    axes[0].hist(decoy_vs_GLOW_CIFAR10_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['red'], label='GLOW', linewidth=2)
    prettyplot.x_axis_only(axes[0])
    prettyplot.title('distribution of log-likelihoods, W = ' + str(log_likelihood_width), axes[0])
    axes[0].set_xlim([1, 7])
    axes[0].set_ylim([0, 3])

    d = 32*32*3
    m = 2
    log_likelihood_width = decoy_vs_FlowGAN_CIFAR10_data_dict['log_likelihood_width'][0][m]
    axes[1].hist(decoy_vs_FlowGAN_CIFAR10_data_dict['data'][0][m][0]/d, 50, density=True, histtype='step', color=prettyplot.colors['black'], label='decoy', linewidth=2)
    axes[1].hist(decoy_vs_FlowGAN_CIFAR10_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['blue'], label='FlowGAN', linewidth=2)
    axes[1].hist(decoy_vs_GLOW_CIFAR10_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['red'], label='GLOW', linewidth=2)
    prettyplot.x_axis_only(axes[1])
    prettyplot.title('distribution of log-likelihoods, W = ' + str(log_likelihood_width), axes[1])
    axes[1].set_xlim([1, 7])
    axes[1].set_ylim([0, 3])

    d = 32*32*3
    m = 4
    log_likelihood_width = decoy_vs_FlowGAN_CIFAR10_data_dict['log_likelihood_width'][0][m]
    axes[2].hist(decoy_vs_FlowGAN_CIFAR10_data_dict['data'][0][m][0]/d, 50, density=True, histtype='step', color=prettyplot.colors['black'], label='decoy', linewidth=2)
    axes[2].hist(np.maximum(decoy_vs_FlowGAN_CIFAR10_data_dict['data'][0][m][1]/d, 0), 50, density=True, histtype='step', color=prettyplot.colors['blue'], label='FlowGAN', linewidth=2)
    axes[2].hist(decoy_vs_GLOW_CIFAR10_data_dict['data'][0][m][1]/d, 50, density=True, histtype='step', color=prettyplot.colors['red'], label='GLOW', linewidth=2)
    prettyplot.x_axis_only(axes[2])
    prettyplot.title('distribution of log-likelihoods, W = ' + str(log_likelihood_width), axes[2])
    axes[2].legend(frameon=False)
    axes[2].set_xlim([1, 7])
    axes[2].set_ylim([0, 3])
    prettyplot.xlabel('log_likelihood / d', axes[2])

    #plt.savefig('PlotSaves/DensityOfLogLikelihoods_CIFAR10.pdf', format='pdf', transparent=True)
    plt.show()

# KL Divergence Multi resolution plot
if True:
    fig = plt.figure(figsize=[3, 2.3])
    make_plot(decoy_vs_FlowGAN_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='KL_divergence',
              x_log=True,
              y_log=False,
              horizontal_line=None,
              xlabel='log likelihood width / d',
              ylabel='D_KL(decoy, model) / d',
              title='D_KL(decoy, model)'
              )

    plt.ylim([0, 6])
    plt.xlim([0.01, 10])
    plt.savefig('PlotSaves/KL_divergence_Multiresolution.pdf', format='pdf', transparent=True)

    plt.show()


# KL Divergence plot
if True:
    fig = plt.figure(figsize=[3, 2.3])
    make_plot(decoy_vs_FlowGAN_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='KL_divergence',
              x_log=True,
              y_log=False,
              widths_to_plot=[32],
              horizontal_line=None,
              label_appended_str=' (FlowGAN MNIST)',
              xlabel='log likelihood width / d',
              ylabel='D_KL(decoy, model) / d',
              title='D_KL(decoy, model)'
              )

    make_plot(decoy_vs_FlowGAN_matched_width_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='KL_divergence',
              x_log=True,
              y_log=False,
              widths_to_plot=[32],
              horizontal_line=None,
              color_offset=1,

              label_appended_str=' (FlowGAN MNIST matched seed)',
              xlabel='log likelihood width / d',
              ylabel='D_KL(decoy, model) / d',
              title='D_KL(decoy, model)'
              )

    make_plot(decoy_vs_GLOW_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='KL_divergence',
              x_log=True,
              y_log=False,
              widths_to_plot=[32],
              horizontal_line=None,
              label_appended_str=' (GLOW MNIST)',
              color_offset=2,
              xlabel=None,
              ylabel=None,
              title=None
              )


    make_plot(decoy_vs_FlowGAN_CIFAR10_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='KL_divergence',
              x_log=True,
              y_log=False,
              widths_to_plot=[32],
              horizontal_line=None,
              label_appended_str=' (FlowGAN, CIFAR10)',
              color_offset=3,
              xlabel=None,
              ylabel=None,
              title=None
              )
    make_plot(decoy_vs_GLOW_CIFAR10_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='KL_divergence',
              x_log=True,
              y_log=False,
              widths_to_plot=[32],
              horizontal_line=None,
              label_appended_str=' (GLOW, CIFAR10)',
              color_offset=4,
              xlabel=None,
              ylabel=None,
              title=None
              )

    plt.ylim([0, 2])
    #plt.savefig('PlotSaves/KL_divergence.pdf', format='pdf', transparent=True)

    plt.show()

# R SQUARED PLOT
if True:
    fig = plt.figure(figsize=[3, 2.3])

    # *********** FIGURE A ************
    make_plot(decoy_vs_FlowGAN_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='r_squared',
              x_log=True,
              y_log=False,
              horizontal_line=1,
              widths_to_plot=[32],
              label_appended_str=' (FlowGAN MIST)',
              color_offset=0,
              xlabel='log likelihood width',
              ylabel='R value',
              title='R value'
              )

    make_plot(decoy_vs_FlowGAN_matched_width_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='r_squared',
              x_log=True,
              y_log=False,
              horizontal_line=1,
              widths_to_plot=[32],
              label_appended_str=' (FlowGAN MNIST matched seed)',
              color_offset=1,
              xlabel='log likelihood width',
              ylabel='R value',
              title='R value of fit between log p decoy and log p FlowGAN'
              )

    make_plot(decoy_vs_GLOW_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='r_squared',
              x_log=True,
              y_log=False,
              horizontal_line=1,
              label_appended_str=' (GLOW MNIST)',
              color_offset=2,
              xlabel='log likelihood width',
              ylabel='R value',
              title='R value of fit between log p decoy and log p FlowGAN'
              )

    make_plot(decoy_vs_FlowGAN_CIFAR10_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='r_squared',
              x_log=True,
              y_log=False,
              horizontal_line=1,
              label_appended_str=' (FlowGAN, CIFAR10)',
              color_offset=3,
              xlabel='log likelihood width',
              ylabel='R value',
              title='R value of fit between log p decoy and log p FlowGAN'
              )

    make_plot(decoy_vs_GLOW_CIFAR10_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='r_squared',
              x_log=True,
              y_log=False,
              horizontal_line=1,
              label_appended_str=' (GLOW, CIFAR10)',
              color_offset=4,
              xlabel='log likelihood width',
              ylabel='R squared',
              title='R squared'
              )
    #plt.savefig('PlotSaves/RSquared.pdf', format='pdf', transparent=True)

    plt.show()

# SLOPE PLOT
if True:
    fig = plt.figure(figsize=[3, 2.3])
    scatter = False
    # *********** FIGURE B ************
    make_plot(decoy_vs_FlowGAN_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='slope',
              threshold_variable='r_squared',
              threshold_value=0.1,
              x_log=True,
              y_log=False,
              horizontal_line=1,
              widths_to_plot=[32],
              label_appended_str=' (FlowGAN, MNIST)',
              color_offset=0,
              scatter=scatter,
              xlabel='log likelihood width',
              ylabel='slope',
              title='slope'
              )

    make_plot(decoy_vs_FlowGAN_matched_width_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='slope',
              x_log=True,
              y_log=False,
              threshold_variable='r_squared',
              threshold_value=0.1,
              horizontal_line=1,
              widths_to_plot=[32],
              label_appended_str=' (FlowGAN, MNIST, matched)',
              color_offset=1,
              scatter=scatter,

              xlabel='log likelihood width',
              ylabel='slope',
              title='slope'
              )

    make_plot(decoy_vs_GLOW_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='slope',
              x_log=True,
              y_log=False,
              horizontal_line=1,
              widths_to_plot=[32],
              threshold_variable='r_squared',
              threshold_value=0.1,
              label_appended_str=' (GLOW, MNIST)',
              color_offset=2,
              scatter=scatter,

              xlabel='log likelihood width',
              ylabel='slope',
              title='slope'
              )

    make_plot(decoy_vs_FlowGAN_CIFAR10_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='slope',
              x_log=True,
              y_log=False,
              horizontal_line=1,
              widths_to_plot=[32],
              threshold_variable='r_squared',
              threshold_value=0.1,
              label_appended_str=' (FlowGAN, CIFAR10)',
              color_offset=3,
              xlabel='log likelihood width',
              ylabel='slope',
              title='slope',
              scatter = scatter,

    )

    make_plot(decoy_vs_GLOW_CIFAR10_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='slope',
              x_log=True,
              y_log=False,
              horizontal_line=1,
              widths_to_plot=[32],
              scatter=scatter,
              threshold_variable='r_squared',
              threshold_value=0.1,
              label_appended_str=' (GLOW, CIFAR10)',
              color_offset=4,
              xlabel='log likelihood width',
              ylabel='slope',
              title='slope'
              )
    plt.xlim([1/np.sqrt(10), 10])
    plt.ylim([0, 1.02])
    #plt.savefig('PlotSaves/Slope.pdf', format='pdf', transparent=True)

    plt.show()

# SLOPES FOR DIFFERENT SIZED IMAGES
if True:
    fig = plt.figure(figsize=[3, 2.3])
    scatter = False
    # *********** FIGURE B ************
    make_plot(decoy_vs_FlowGAN_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='slope',
              threshold_variable='r_squared',
              threshold_value=0.1,
              x_log=True,
              y_log=False,
              horizontal_line=1,
              color_offset=0,
              scatter=scatter,
              xlabel='log likelihood width',
              ylabel='slope',
              title='slope'
              )

    plt.xlim([0.1, 10])
    plt.ylim([0, 1.5])
    #plt.savefig('PlotSaves/SlopeWithDifferentImageSizes.pdf', format='pdf', transparent=True)

    plt.show()


# R SQUARED FOR DIFFERENT SIZED IMAGES
if True:
    fig = plt.figure(figsize=[3, 2.3])
    scatter = False
    # *********** FIGURE B ************
    make_plot(decoy_vs_FlowGAN_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='r_squared',
              x_log=True,
              y_log=False,
              horizontal_line=1,
              color_offset=0,
              scatter=scatter,
              xlabel='log likelihood width',
              ylabel='r_squared',
              title='r_squared'
              )

    plt.xlim([0.01, 10])
    plt.ylim([0, 1.02])
    #plt.savefig('PlotSaves/RSquaredWithDifferentImageSizes.pdf', format='pdf', transparent=True)

    plt.show()


if False:
    # *********** FIGURE D ************
    make_plot(decoy_vs_FlowGAN_MNIST_data_dict,
              x_axis='log_likelihood_width_divided_by_d',
              y_axis='intercept_divided_by_d',
              x_log=True,
              y_log=False,
              horizontal_line=0,
              xlabel='log likelihood width / d',
              ylabel='intercept / d',
              title='intercept / d of fit between log p decoy and log p FlowGAN'
              )
    plt.show()

# PLOTS OF LINEAR FIT
if True:
    fig = plt.figure(figsize=[3, 3])
    d = 32*32
    m = 2
    k = 3
    log_likelihood_width = decoy_vs_FlowGAN_MNIST_data_dict['log_likelihood_width'][k][m]
    r_squared = decoy_vs_FlowGAN_MNIST_data_dict['r_squared'][k][m]
    slope = decoy_vs_FlowGAN_MNIST_data_dict['slope'][k][m]

    #print('log likelihood width =', log_likelihood_width, 'r squared =', r_squared, 'slope =', slope)
    xs = decoy_vs_FlowGAN_MNIST_data_dict['data'][k][m][1]/d
    ys = decoy_vs_FlowGAN_MNIST_data_dict['data'][k][m][0]/d
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)

    plt.plot([-2, 2], [-2, 2], color='k', linestyle='dashed')
    plt.xlim([-.3, .3])
    plt.ylim([-.3, .3])
    sb.kdeplot(x=ys, y=xs, cmap='Blues', shade=True, levels=100, joint_kws={"linewidths":1, "linecolors":'k'})

    plt.plot(np.array([-2, 2]), slope*np.array([-2,2]), color=prettyplot.colors['red'])

    prettyplot.xlabel('log p model')
    prettyplot.ylabel('log p decoy')

    prettyplot.no_box()
    plt.gca().set_aspect(1)
    #plt.savefig('PlotSaves/FILLS_FitPlot_' + str(d) + '_' + str(log_likelihood_width) +'.pdf', format='pdf', transparent=True)

    plt.show()



if True:
    fig = plt.figure(figsize=[3, 3])
    d = 32*32
    m = 4
    k = 3
    log_likelihood_width = decoy_vs_FlowGAN_MNIST_data_dict['log_likelihood_width'][k][m]
    r_squared = decoy_vs_FlowGAN_MNIST_data_dict['r_squared'][k][m]
    slope = decoy_vs_FlowGAN_MNIST_data_dict['slope'][k][m]

    #print('log likelihood width =', log_likelihood_width, 'r squared =', r_squared, 'slope =', slope)
    xs = decoy_vs_FlowGAN_MNIST_data_dict['data'][k][m][1]/d
    ys = decoy_vs_FlowGAN_MNIST_data_dict['data'][k][m][0]/d
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)

    plt.plot([-3, 3], [-3, 3], color='k', linestyle='dashed')
    plt.xlim([-2.2, 2.2])
    plt.ylim([-2.2, 2.2])
    sb.kdeplot(x=ys, y=xs, cmap='Blues', shade=True, levels=100)

    plt.plot(np.array([-3, 3]), decoy_vs_FlowGAN_MNIST_data_dict['slope'][k][m]*np.array([-3,3]), color=prettyplot.colors['red'])

    prettyplot.xlabel('log p decoy')
    prettyplot.ylabel('log p model')

    prettyplot.no_box()
    plt.gca().set_aspect(1)
    #plt.savefig('PlotSaves/FILLS_FitPlot_' + str(d) + '_' + str(log_likelihood_width) +'.pdf', format='pdf', transparent=True)

    plt.show()


if True:
    fig = plt.figure(figsize=[3, 3])
    d = 4*4
    m = 0
    k = 0
    log_likelihood_width = decoy_vs_FlowGAN_MNIST_data_dict['log_likelihood_width'][k][m]
    r_squared = decoy_vs_FlowGAN_MNIST_data_dict['r_squared'][k][m]
    slope = decoy_vs_FlowGAN_MNIST_data_dict['slope'][k][m]
    #print('log likelihood width =', log_likelihood_width, 'r squared =', r_squared, 'slope =', slope)
    xs = decoy_vs_FlowGAN_MNIST_data_dict['data'][k][m][1]/d
    ys = decoy_vs_FlowGAN_MNIST_data_dict['data'][k][m][0]/d
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)

    plt.plot([-3, 3], [-3, 3], color='k', linestyle='dashed')
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    sb.kdeplot(x=ys, y=xs, cmap='Blues', shade=True, levels=100)

    plt.plot(np.array([-3, 3]), decoy_vs_FlowGAN_MNIST_data_dict['slope'][k][m]*np.array([-3, 3]), color=prettyplot.colors['red'])

    prettyplot.xlabel('log p decoy')
    prettyplot.ylabel('log p model')

    prettyplot.no_box()
    plt.gca().set_aspect(1)
    #plt.savefig('PlotSaves/FitPlot_' + str(d) + '_' + str(log_likelihood_width) +'.pdf', format='pdf', transparent=True)

    plt.show()

# Plot the FID scores for ML, GAN and AdvEnt decoy distributions trained on CIFAR10
file_handle = open('DataForFigurePlots/ML_GAN_Advent_ComparisonData/fid_dict.obj', 'rb')
FID_scores_dict = pickle.load(file_handle)
scores = [[], [], []]
for index, key in enumerate(FID_scores_dict.keys()):
    if index > 0:
        i = index  - 1
        scores[i//5].append(FID_scores_dict[key])

file_handle.close()

FWs_per_dimension = np.array([32, 128, 512, 2048, 4096])/1024
fig = plt.figure(figsize=(3, 2.3))
plt.plot(FWs_per_dimension, scores[0], label='Maximum Likelihood', c=prettyplot.colors['red'], marker='o')
plt.plot(FWs_per_dimension, scores[1], label='GAN', c=prettyplot.colors['blue'], marker='o')
plt.plot(FWs_per_dimension, scores[2], label='AdvEnt', c=prettyplot.colors['black'], marker='o')
prettyplot.no_box()
plt.gca().set_xscale('log')
prettyplot.ylabel('FID score')
prettyplot.xlabel('width per dimension')
plt.ylim([0, np.max(scores)*1.1])
plt.legend(frameon=False)
plt.savefig('PlotSaves/FID_scores.pdf', format='pdf', transparent=True)
plt.show()
