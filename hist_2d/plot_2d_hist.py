import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import copy

def averages_from_histograms(hist, edges):
    num_hist = hist.size
    num_edges = edges.size
    mean_value = np.nan
    median_value = np.nan
    if ((num_hist + 1) == num_edges):
        tot_num = np.sum(hist)
        if tot_num > 2:
            median_num = tot_num / 2.0
            cumulative_sum = np.cumsum(hist)
            past_median = np.where(cumulative_sum > median_num)
            try:
                j = np.min(past_median)
            except:
                print
            median_value = edges[j] + ((median_num - cumulative_sum[j - 1]) / hist[j]) * (edges[j + 1] - edges[j])
            centers = 0.5 * (edges[0:num_edges - 1] + edges[1:num_edges])
            mean_value = np.sum(centers * hist) / tot_num
    return mean_value, median_value

def calc_stats_from_hist(hist,edges):
    num_bins = len(hist)
    bin_size = edges[1] - edges[0]
    bin_vals = np.arange(bin_size/2.0,edges[-1],bin_size)
    mean = np.sum(bin_vals*hist)/np.sum(hist)
    var = np.sum(hist*np.square(bin_vals-mean))/np.sum(hist)
    stddev = np.sqrt(var)
    num = np.sum(hist)
    return mean,stddev,num

def plot_2d_hist(hist, xedges, yedges, nbins=120,
                 title='', xtitle='', ytitle='',  
                 z1_range=(0.0, 1.2), z2_range=(0.0, 1.2), 
                 aspect='equal', plot_diagonal=False, plot_horiz_means=False,num_scale = 10000.0,reduce_max = 1.0,
                 plot_horiz_medians=True,plot_vert_medians = True,
                 fig_in = None,ax_in = None,
                 norm='Linear',cmap = 'ocean_r',plt_colorbar=True,fontsize=16):

    X, Y = np.meshgrid(xedges, yedges)
    hist_max = np.max(hist)*reduce_max
    
    if fig_in is None:
        fig = plt.figure(figsize=(8, 6))
    else:
        fig = fig_in

    if ax_in is None:
        ax = fig.add_subplot(111,aspect=aspect)
    else:
        ax = ax_in

    ax.set_title(title)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    if norm == 'Log':
        im = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=colors.LogNorm(vmin=hist_max / num_scale, vmax=hist_max))
    elif norm == 'Pow05':
        im = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=colors.PowerNorm(gamma=0.5))
    elif norm == 'Pow03':
        im = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=colors.PowerNorm(gamma=0.3))
    else:
        im = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=colors.Normalize())
   
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(fontsize)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
    
    if  z1_range is not None:
        ax.set_xlim(z1_range)
    if  z2_range is not None:
        ax.set_ylim(z2_range)
        ax.plot(z1_range,z2_range,color='r')

    if plt_colorbar:
        cbar = fig.colorbar(im,ax=ax)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_ylabel('Number of Observations', fontsize=fontsize)
    if plot_diagonal:
        ax.plot([np.nanmin(xedges),np.nanmax(xedges)],[np.nanmin(yedges),np.nanmax(yedges)], color='red')
    if plot_horiz_medians:
        num_xbins = (hist.shape)[0]
        num_ybins = (hist.shape)[1]
        median_arr = np.zeros((num_ybins))
        num_arr = np.zeros((num_ybins))
        for iy in range(0, num_ybins):
            mean_no_used,median_arr[iy] = averages_from_histograms(hist[iy, :], xedges)
            num_arr[iy] = np.sum(hist[iy, :])
        median_arr[num_arr < np.nanmax(num_arr)/5000.0] = np.nan

        y_vals  = (yedges[0:-1] + yedges[1:])/2.
        ax.plot(median_arr, y_vals,color='orange')
    if plot_vert_medians:
        num_xbins = (hist.shape)[0]
        num_ybins = (hist.shape)[1]
        median_arr = np.zeros((num_xbins))
        num_arr = np.zeros((num_xbins))
        for ix in range(0, num_xbins):
            mean_no_used,median_arr[ix] = averages_from_histograms(hist[:,ix], yedges)
            num_arr[ix] = np.sum(hist[:,ix])
        median_arr[num_arr < np.nanmax(num_arr)/5000.0] = np.nan

        x_vals  = (xedges[0:-1] + xedges[1:])/2.
        ax.plot(x_vals,median_arr,color='orange')

    return fig, ax

def plot_2D_hist_slices(hist, xedges, yedges, title='', xtitle='', ytitle='', nbins=120, z1_range=(0.0, 1.2),
                 z2_range=(0.0, 1.2),plot_diagonal=False,slice_indices = [3,4,5,6,7,8,9,10,11,12,18]):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, title=title, xlabel=xtitle, ylabel=ytitle)

    for i,islice in enumerate(slice_indices):

        slice = hist[:,2*islice] #+ hist[:,1 + 2*islice]
        xvals = (np.arange(0,60)/2.0) - (islice)
        ax.plot(xvals,slice,label=str(islice+0.25))

    ax.set_xlim(-6.0,6.0)
    ax.legend()

        #for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        #    item.set_fontsize(20)
        #for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        #   item.set_fontsize(16)
        # ax.plot(z1_range,z2_range,color='r')

    return fig,ax