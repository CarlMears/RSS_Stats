def calc_binned_means_from_2d_hist(h, xbins=40, ybins=40, xr=[0.0, 1.0], yr=[-0.5, 0.5],diff_hist=True):
    import numpy as np
    import matplotlib.pyplot as plt

    binned_x_means = np.zeros(xbins)
    binned_y_means = np.zeros(xbins)
    binned_y_stddev = np.zeros(xbins)
    binned_num = np.zeros(xbins)
    binned_hist = h
    x_edges = np.zeros(xbins + 1)

    # loop over xbins
    ybin_size = (yr[1] - yr[0]) / ybins
    xbin_size = (xr[1] - xr[0]) / xbins

    sumy_tot = 0.0
    sum2_tot = 0.0
    sum3_tot = 0.0
    for ix in range(0, xbins):
        hist = h[ix, :]
        sumy = 0.0
        sumy2 = 0.0
        num = np.sum(hist)
        for iy in range(0, ybins):
            if diff_hist:
                ybin_center = yr[0] + (0.5 + iy) * ybin_size
            else:
                ybin_center = yr[0] + (0.5 + iy) * ybin_size - (xr[0]+(0.5 + ix) * xbin_size)


            sumy = sumy + hist[iy] * ybin_center
            sumy_tot = sumy_tot + hist[iy] * ybin_center
        binned_x_means[ix] = xr[0] + (0.5 + ix) * xbin_size
        binned_y_means[ix] = sumy / num
        binned_num[ix] = num

        sum2 = 0.0
        for iy in range(0, ybins):
            if diff_hist:
                ybin_center = yr[0] + (0.5 + iy) * ybin_size
            else:
                ybin_center = yr[0] + (0.5 + iy) * ybin_size - (xr[0] + (0.5 + ix) * xbin_size)


            sum2 = sum2 + np.square(ybin_center - binned_y_means[ix]) * hist[iy]
        binned_y_stddev[ix] = np.sqrt(sum2 / num)

    overall_bias = sumy_tot / np.sum(h)

    sum2_tot = 0.0
    for ix in range(0, xbins):
        for iy in range(0, ybins):
            ybin_center = yr[0] + (0.5 + iy) * ybin_size
            sum2_tot = sum2_tot + np.square(ybin_center - overall_bias) * h[ix, iy]
            sum3_tot = sum3_tot + np.square(ybin_center) * h[ix, iy]

    overall_std = np.sqrt(sum2_tot / np.sum(h))
    overall_rms = np.sqrt(sum3_tot / np.sum(h))
    edges = yr[0] + ybin_size * np.arange(0, ybins + 1)

    binned_stats = dict(binned_x_means=binned_x_means,
                        binned_y_means=binned_y_means,
                        binned_y_stddev=binned_y_stddev,
                        binned_num=binned_num,
                        overall_bias=overall_bias,
                        overall_std=overall_std,
                        overall_rms=overall_rms,
                        binned_hist=binned_hist,
                        edges=edges)
    return binned_stats


def calc_binned_means(x, y, mask, bins=40, xrng=[0.0, 1.0], num_hist_bins=100, hist_range=[-0.5, 0.5], verbose=False,
                      return_as_xarray=False):
    import numpy as np
    import matplotlib.pyplot as plt

    binned_x_means = np.zeros(bins)
    binned_y_means = np.zeros(bins)
    binned_y_stddev = np.zeros(bins)
    binned_num = np.zeros(bins)
    binned_hist = np.zeros((bins, num_hist_bins))
    x_edges = np.zeros(bins + 1)

    bin_size = (xrng[1] - xrng[0]) / bins
    x_edges = xrng[0] + np.arange(0, bins + 1) * bin_size

    for j in range(0, bins):
        bin_start = xrng[0] + j * bin_size
        bin_end = xrng[0] + (j + 1) * bin_size
        with np.errstate(invalid='ignore'):
            z = np.all([(mask < 0.5), (x >= bin_start), (x <= bin_end), np.isfinite(y), np.isfinite(x)], axis=(0))
        x_in_bin = x[z]
        y_in_bin = y[z]
        n_in_bin = np.sum(z)
        binned_x_means[j] = np.mean(x_in_bin)
        binned_y_means[j] = np.mean(y_in_bin)
        binned_y_stddev[j] = np.std(y_in_bin)
        binned_num[j] = n_in_bin
        hist, y_edges = np.histogram(y_in_bin, bins=num_hist_bins, range=hist_range)
        binned_hist[j, :] = hist
        if verbose:
            print(bin_start, bin_end, n_in_bin, binned_x_means[j], binned_y_means[j], binned_y_stddev[j])

    # also find overall statsitic for a large 0.1 to 0.9 bin
    bin_start = 0.1
    bin_end = 0.9
    with np.errstate(invalid='ignore'):
        z = np.all([(mask < 0.5), (x >= bin_start), (x <= bin_end), np.isfinite(y), np.isfinite(x)], axis=(0))
    x_in_bin = x[z]
    y_in_bin = y[z]

    overall_bias = np.mean(y_in_bin)
    overall_std = np.std(y_in_bin)
    overall_rms = np.sqrt(np.mean(np.square(y_in_bin)))

    if return_as_xarray:
        import xarray as xr
        xbin_nums = np.arange(0, bins)
        binned_stats = xr.Dataset(
            data_vars={'binned_x_means': (('xbin_num'), binned_x_means),
                       'binned_y_means': (('xbin_num'), binned_y_means),
                       'binned_y_stddev': (('xbin_num'), binned_y_stddev),
                       'binned_num': (('xbin_num'), binned_num),
                       'x_edges': (('xbin_num_ext'), x_edges),
                       'y_edges': (('ybin_num_ext'), y_edges),
                       'binned_hist': (('xbin_num', 'ybin_num'), binned_hist),
                       'overall_bias': overall_bias,
                       'overall_std': overall_std,
                       'overall_rms': overall_rms
                       },
            coords={'xbin_num': np.arange(0, bins),
                    'xbin_num_ext': np.arange(0, bins + 1),
                    'ybin_num': np.arange(0, num_hist_bins),
                    'ybin_num_ext': np.arange(0, num_hist_bins + 1)
                    }

        )

    else:  # return as dict
        binned_stats = dict(binned_x_means=binned_x_means,
                            binned_y_means=binned_y_means,
                            binned_y_stddev=binned_y_stddev,
                            binned_num=binned_num,
                            x_edges=x_edges,
                            overall_bias=overall_bias,
                            overall_std=overall_std,
                            overall_rms=overall_rms,
                            binned_hist=binned_hist,
                            y_edges=y_edges)
    return binned_stats

def extract_binned_stats(binned_stats):

    import xarray as xr
    #extract the  various components of binned_stats from dict or xarray representation of binned_stats

    if type(binned_stats) is dict:
        xmeans = binned_stats['binned_x_means']
        ymeans = binned_stats['binned_y_means']
        ystd   = binned_stats['binned_y_stddev']
        num    = binned_stats['binned_num']
        binned_hist = binned_stats['binned_hist']
        overall_bias = binned_stats['overall_bias']
        overall_std = binned_stats['overall_std']
        try:
            x_edges = binned_stats['x_edges']
            y_edges = binned_stats['y_edges']
        except:
            x_edges = 0.0
            y_edges = 0.0
    elif type(binned_stats) is xr.core.dataset.Dataset:
        xmeans = binned_stats.binned_x_means.data
        ymeans = binned_stats.binned_y_means.data
        ystd   = binned_stats.binned_y_stddev.data
        num    = binned_stats.binned_num.data
        binned_hist = binned_stats.binned_hist.data
        overall_bias = binned_stats.overall_bias.data
        overall_std = binned_stats.overall_std.data
        x_edges = binned_stats.x_edges.data
        y_edges = binned_stats.y_edges.data
    else:
        raise

    return xmeans,ymeans,ystd,num,binned_hist,overall_bias,overall_std,x_edges,y_edges

def combine_binned_stats(binned_stats1, binned_stats2):
    import numpy as np
    import math
    import xarray as xr

    xmeans1,ymeans1,ystd1,num1,binned_hist1,overall_bias1,overall_std1,x_edges1,y_edges1 = extract_binned_stats(binned_stats1)
    xmeans2,ymeans2,ystd2,num2,binned_hist2,overall_bias2,overall_std2,x_edges2,y_edges2 = extract_binned_stats(binned_stats2)

    shape1 = xmeans1.shape
    shape2 = xmeans2.shape
    if (shape1 != shape2):
        raise ValueError('Number of bins must be the same')
    if (xmeans1.ndim != 1):
        raise ValueError('should be 1 d arrays')
    if (binned_hist1.ndim != 2):
        raise ValueError('hist should be 2 d arrays')
    hist_shape1 = binned_hist1.shape
    hist_shape2 = binned_hist2.shape

    if (hist_shape1 != hist_shape2):
        raise ValueError('Number of bins in histograms must be the same')
    if (np.any(x_edges1 != x_edges2)):
        raise ValueError('Histogram X edgs values mush be the same')
    if (np.any(y_edges1 != y_edges2)):
        raise ValueError('Histogram Y edgs values mush be the same')

    n1 = (xmeans1.shape)[0]

    xbin3 = np.zeros((n1))
    ybin3 = np.zeros((n1))
    dybin3 = np.zeros((n1))
    numbin3 = np.zeros((n1))
    binned_hist3 = np.zeros(hist_shape1)

    # fix up bad bins to make sure no NANs
    bad = np.any([(num1 == 0), ~np.isfinite(ymeans1)], axis=(0))
    ymeans1[bad] = 0.0
    xmeans1[bad] = 0.0
    ystd1[bad] = 0.0
    num1[bad] = 0.0
    bad = np.any([(num2 == 0), ~np.isfinite(ymeans2)], axis=(0))
    ymeans2[bad] = 0.0
    xmeans2[bad] = 0.0
    ystd2[bad] = 0.0
    num2[bad] = 0.0

    bad2 = ~np.isfinite(ystd1)
    ystd1[bad2] = 0.0
    bad2 = ~np.isfinite(ystd2)
    ystd2[bad2] = 0.0


    with np.errstate(divide='ignore',invalid='ignore'):
        xbin3 = (num1*xmeans1 + num2*xmeans2) /(num1+num2)
        ybin3 = (num1*ymeans1 + num2*ymeans2) /(num1+num2)
        numbin3 = num1 + num2

        dybin3 = np.sqrt(
            (num1*ystd1*ystd1 + num2*ystd2*ystd2  +
             num1*(ymeans1-ybin3)*(ymeans1-ybin3) +
             num2*(ymeans2-ybin3)*(ymeans2-ybin3))/
            (num1 + num2))

    dybin3[numbin3 < 2] = np.nan
    xbin3[numbin3 < 1] = np.nan
    ybin3[numbin3 < 1] = np.nan






    tot_n1 = np.sum(num1)
    tot_n2 = np.sum(num2)

    if np.isnan(overall_bias1):
        overall_bias1 = 0.0
        overall_std1 = 0.0
        tot_n1 = 0.0

    if np.isnan(overall_bias2):
        overall_bias2 = 0.0
        overall_std2 = 0.0
        tot_n2 = 0.0

    tot_n3 = tot_n1 + tot_n2
    overall_bias3 = ((tot_n1 * overall_bias1) + (tot_n2 * overall_bias2)) / tot_n3

    overall_std3 = np.sqrt(tot_n1 * overall_std1 * overall_std1 +
                           tot_n2 * overall_std2 * overall_std2 +
                           tot_n1 * np.square(overall_bias1 - overall_bias3) +
                           tot_n2 * np.square(overall_bias2 - overall_bias3))/np.sqrt(tot_n3)

    overall_rms3 = math.sqrt(overall_bias3 * overall_bias3 + overall_std3 * overall_std3)

    binned_hist3 = binned_hist1 + binned_hist2

    if type(binned_stats1) is xr.core.dataset.Dataset:
        bins = xmeans1.shape[0]
        num_hist_bins = binned_hist1.shape[1]
        binned_stats = xr.Dataset(
            data_vars={'binned_x_means': (('xbin_num'), xbin3),
                       'binned_y_means': (('xbin_num'), ybin3),
                       'binned_y_stddev': (('xbin_num'), dybin3),
                       'binned_num': (('xbin_num'), numbin3),
                       'x_edges': (('xbin_num_ext'), x_edges1),
                       'y_edges': (('ybin_num_ext'), y_edges1),
                       'binned_hist': (('xbin_num', 'ybin_num'), binned_hist3),
                       'overall_bias': overall_bias3,
                       'overall_std': overall_std3,
                       'overall_rms': overall_rms3
                       },
            coords={'xbin_num': np.arange(0, bins),
                    'xbin_num_ext': np.arange(0, bins + 1),
                    'ybin_num': np.arange(0, num_hist_bins),
                    'ybin_num_ext': np.arange(0, num_hist_bins + 1)
                    })
    else:
        binned_stats = dict(binned_x_means=xbin3,
                            binned_y_means=ybin3,
                            binned_y_stddev=dybin3,
                            binned_num=numbin3,
                            overall_bias=overall_bias3,
                            overall_std=overall_std3,
                            overall_rms=overall_rms3,
                            binned_hist=binned_hist3,
                            x_edges=x_edges1,
                            y_edges = y_edges1)

    return binned_stats


def plot_binned_means(binned_stats, yrng=[-0.5, 0.5], xlab='Mean', ylab='Binned Difference', title=' ', requirement=0.0,
                      plot_num_in_bins=False, num_thres=0,plot_bins_under_thres = True):

    import numpy as np
    import matplotlib.pyplot as plt

    xbin,ybin,ystd,num,binned_hist1,overall_bias1,overall_std1,x_edges1,y_edges1 = extract_binned_stats(binned_stats)
    percent_in_bin = 100.0 * num/np.sum(num)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, title=title, xlabel=xlab, ylabel=ylab)
    plt.tight_layout()
    ax.errorbar(xbin[percent_in_bin > num_thres], ybin[percent_in_bin > num_thres], yerr=ystd[percent_in_bin > num_thres], fmt='s', color='blue')
    if plot_bins_under_thres:
        ax.errorbar(xbin[percent_in_bin <= num_thres], ybin[percent_in_bin <= num_thres], yerr=ystd[percent_in_bin <= num_thres], fmt='s', color='lightblue')
    ax.set_ylim(yrng)
    ax.tick_params(direction='in')
    ax.plot([0.0, 1.0], [0.0, 0.0], color='red')
    if requirement > 0.0001:
        ax.plot([0.0, 1.0], [-requirement, -requirement], color='gray')
        ax.plot([0.0, 1.0], [requirement, requirement], color='gray')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(21)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)

    if plot_num_in_bins:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(xbin[1:-1], percent_in_bin[1:-1], color='darkgreen')
        ax2.set_ylabel('Percent of Total Obs in Bin')
        ax2.set_ylim([0,3])

        for item in ([ax2.yaxis.label]):
            item.set_fontsize(16)

    return fig