import numpy as np
import xarray as xr

class BinnedStat():

    def __init__(self, num_bins = 40,x_rng=[0.0,1.0]):

        self.num_bins = num_bins
        self.x_rng = x_rng

        self.binned_x_tot           = np.zeros(self.num_bins)
        self.binned_y_tot           = np.zeros(self.num_bins)
        self.binned_y_totsqr        = np.zeros(self.num_bins)
        self.binned_num             = np.zeros(self.num_bins)

        self.overall_tot            = 0.0
        self.overall_totsqr         = 0.0
        self.overall_num            = 0.0

        self.xbin_size = (self.x_rng[1]-self.x_rng[0])/self.num_bins
        self.xbin_centers = np.arange(self.x_rng[0] + self.xbin_size/2.0,self.x_rng[1],self.xbin_size) # don't subtract from end point to make sure we get last point
        self.xbin_edges = np.arange(self.x_rng[0],self.x_rng[1]+0.1*self.xbin_size,self.xbin_size)

    def from_dataset(ds = None):
        #creates BinnedStat object from a binned stat xarray dataset
        xbin_centers = ds['xbin_centers'] 
        try:
            xbin_edges = ds['xbin_edges']
        except KeyError:
            delta_x = xbin_centers.values[1]-xbin_centers.values[0]
            xbin_edges = np.arange(xbin_centers.values[0]-delta_x/2.0,xbin_centers.values[-1]+delta_x/1.9,delta_x)
        
        x_rng = [xbin_edges.values[0],xbin_edges.values[-1]]
        num_bins = xbin_centers.shape[0]

        self = BinnedStat(num_bins = num_bins,x_rng=x_rng)

        self.binned_x_tot= ds['binned_x_tot']
        self.binned_y_tot = ds['binned_y_tot']
        self.binned_y_totsqr = ds['binned_y_totsqr']
        self.binned_num = ds['binned_num']

        self.overall_num = ds.attrs['overall_num']
        self.overall_tot = ds.attrs['overall_tot']
        self.overall_totsqr = ds.attrs['overall_totsqr']

        return self

    def from_DataArray(da = None):
        #creates BinnedStat object from a binned stat xarray DataArray
        try:
            xbin_centers = da['xbin_centers'] 
        except:
            try:
                xbin_centers = da['wind_bin']
            except:
                try:
                    xbin_centers = da['binned_stat_xbin_centers']
                except:
                    raise ValueError('can not find xbin_centers')
        
        bin_size = xbin_centers[1]-xbin_centers[0]
        x_rng = [xbin_centers.values[0]-bin_size/2.0,xbin_centers.values[-1]+bin_size/2.0]

        num_bins = xbin_centers.shape[0]

        self = BinnedStat(num_bins = num_bins,x_rng=x_rng)
        try:
            stat_list = da.bin_stat.values
        except:
            try:
                stat_list = da.binned_stat_type.values
            except:
                raise ValueError('can not find stat names')

        # figure out locations of various data types in array
        try:
            num_index = np.where(stat_list == 'number')[0][0]
        except:
            num_index = np.where(stat_list == 'num')[0][0]

        ytot_index = np.where(stat_list == 'y_tot')[0][0]
        ytotsqr_index = np.where(stat_list == 'y_totsqr')[0][0]
        xtot_index = np.where(stat_list == 'x_tot')[0][0]
        
        # figure out is transpose needed.
        try:
            shape = da.shape
        except:
            shape = da.values.shape
        transpose_needed = False
        if shape[0] == num_bins:
            transpose_needed = True

        if transpose_needed:
            self.binned_num      = da[:,num_index].values
            self.binned_x_tot    = da[:,xtot_index].values
            self.binned_y_tot    = da[:,ytot_index].values
            self.binned_y_totsqr = da[:,ytotsqr_index].values
        else:
            self.binned_num      = da[num_index,:].values
            self.binned_x_tot    = da[xtot_index,:].values
            self.binned_y_tot    = da[ytot_index,:].values
            self.binned_y_totsqr = da[ytotsqr_index,:].values

        try:
            self.overall_num = da.attrs['overall_num']
        except:
            self.overall_num = np.nansum(self.binned_num)
        try:
            self.overall_tot = da.attrs['overall_tot']
        except:
            self.overall_tot = np.nansum(self.binned_y_tot)
        try:
            self.overall_totsqr = da.attrs['overall_totsqr']
        except:
            self.overall_totsqr = np.nansum(self.binned_y_totsqr)

        return self

    def from_netcdf(nc_file = None):

        try:
            ds = xr.open_dataset(nc_file)
        except:
            return np.nan
        try:
            self = BinnedStat.from_dataset(ds)
        except KeyError:
            self = BinnedStat.from_DataArray(ds)
        return self

    def combine(self,self2):

        if isinstance(self2,BinnedStat):
            #print('Class types match')
            try:
                assert(np.all(self.xbin_edges == self2.xbin_edges))
            except:
                raise ValueError('bin definitions to not match, can not combine')

            self.overall_num =     self.overall_num +     self2.overall_num
            self.overall_tot =     self.overall_tot +     self2.overall_tot
            self.overall_totsqr =  self.overall_totsqr +  self2.overall_totsqr

            self.binned_num =      self.binned_num +      self2.binned_num
            self.binned_x_tot =    self.binned_x_tot +    self2.binned_x_tot
            self.binned_y_tot =    self.binned_y_tot +    self2.binned_y_tot
            self.binned_y_totsqr = self.binned_y_totsqr + self2.binned_y_totsqr
        else:
            raise ValueError('objects do not match, can not combine')

        return self


    def add_data(self,x,y,mask=None,verbose=False):
        
        if mask is None:
            mask = np.zeros_like(x)

        for j in range(0, self.num_bins):

            bin_start = self.xbin_edges[j]
            bin_end =   self.xbin_edges[j+1]
            
            with np.errstate(invalid='ignore'):
                z = np.all([(mask < 0.5), (x >= bin_start), (x <= bin_end),(np.isfinite(y))], axis=(0))
            x_in_bin = x[z]
            y_in_bin = y[z]
            n_in_bin = np.sum(z)
            if verbose:
                print(bin_start, bin_end, n_in_bin)

            if n_in_bin > 0:
                self.binned_x_tot[j] = self.binned_x_tot[j] + np.sum(x[z])
                self.binned_y_tot[j] = self.binned_y_tot[j] + np.sum(y[z])
                self.binned_y_totsqr[j] = self.binned_y_totsqr[j] + np.sum(np.square(y[z]))
                self.binned_num[j] = self.binned_num[j] + n_in_bin

                self.overall_tot = self.overall_tot  + np.sum(y[z])    
                self.overall_totsqr =self.overall_totsqr + np.sum(np.square(y[z]))
                self.overall_num  =  self.overall_num + n_in_bin
        
        #print(self.overall_tot/self.overall_num)
        
    def as_DataArray(self):
        # return as an xarray DataArray
        # for this representation, binned hist is ignored

        temp = np.zeros((self.num_bins,4))
        temp[:,0] = self.binned_num
        temp[:,1] = self.binned_x_tot
        temp[:,2] = self.binned_y_tot
        temp[:,3] = self.binned_y_totsqr



        binned_stats_xr = xr.DataArray(
            temp,
            dims = ('binned_stat_xbin_centers','binned_stat_type'),

            coords={'binned_stat_xbin_centers'      : self.xbin_centers,
                    'binned_stat_type'              : ['num','x_tot','y_tot','y_totsqr']
                    },
            attrs ={'overall_tot'       : self.overall_tot,
                    'overall_totsqr'    : self.overall_totsqr,
                    'overall_num'       : self.overall_num,
                    'binned_stat_xbin_edges'        : self.xbin_edges}
                    )
                    
        return binned_stats_xr

    def calc_stats(self,return_as_xr = False):  
        '''returns binned means, stddevs, xbinned means based on current state of object.
           unless requested, returns a dictionary of mostly numpy arrays.  if requested, returns an
           xarray dataset'''
        binned_x_means = np.full((self.num_bins),np.nan)
        binned_y_means =np.full((self.num_bins),np.nan)
        binned_y_stddev = np.full((self.num_bins),np.nan)
        
        ok = self.binned_num > 0
        if np.sum(ok) > 0:
            binned_x_means[ok] = self.binned_x_tot[ok]/self.binned_num[ok]
            binned_y_means[ok] = self.binned_y_tot[ok]/self.binned_num[ok]

        sum_of_sqr_dev = np.full((self.num_bins),np.nan)
        ok = self.binned_num > 2
        if np.sum(ok) > 0:
            sum_of_sqr_dev[ok] = self.binned_y_totsqr[ok] - (np.square(self.binned_y_tot[ok]))/self.binned_num[ok]
            binned_y_stddev[ok] = np.sqrt(sum_of_sqr_dev[ok]/(self.binned_num[ok] - 1))

        overall_mean = np.nan
        overall_stddev = np.nan
        overall_rms = np.nan
        overall_num = self.overall_num

        if self.overall_num > 0:
            overall_mean = self.overall_tot/self.overall_num
        if self.overall_num > 2:
            overall_stddev = np.sqrt((self.overall_num*self.overall_totsqr - np.square(self.overall_tot))/
                                     (self.overall_num*(self.overall_num-1)))
            overall_rms    = np.sqrt(self.overall_totsqr/self.overall_num)

        if return_as_xr:
            binned_stats = xr.Dataset(
                        data_vars={'binned_x_means':    (('xbin_centers'),binned_x_means),
                                   'binned_y_means':    (('xbin_centers'),binned_y_means),
                                   'binned_y_stddev':   (('xbin_centers'),binned_y_stddev),
                                   'binned_num'     :        (('xbin_centers'),self.binned_num)},
                        coords={'xbin_centers'      : self.xbin_centers,
                                'xbin_edges'        : self.xbin_edges},
                        attrs ={'overall_mean'      : overall_mean,
                                'overall_stddev'    : overall_stddev,
                                'overall_rms'       : overall_rms,
                                'overall_num'       : overall_num}
                               )
        else:
            binned_stats = dict(binned_x_means=binned_x_means,
                            binned_y_means=binned_y_means,
                            binned_y_stddev=binned_y_stddev,
                            binned_num=self.binned_num,
                            overall_mean=overall_mean,
                            overall_stddev=overall_stddev,
                            overall_rms=overall_rms,
                            overall_num=self.overall_num,
                            xbin_centers = self.xbin_centers,
                            xbin_edges = self.xbin_edges,
                            xrng = [self.xbin_edges[0],self.xbin_edges[-1]])
        
        return binned_stats

    
    def plt(self, yrng=None, xlab='Wind', ylab='Binned Difference', title=' ', requirement=None,
                      plot_num_in_bins=False, num_thres=0):
        import numpy as np
        import xarray as xr
        import matplotlib.pyplot as plt

        binned_stats = self.calc_stats()

        xbin = binned_stats['binned_x_means']
        ybin = binned_stats['binned_y_means']
        ystd = binned_stats['binned_y_stddev']
        num  = binned_stats['binned_num']

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111, title=title, xlabel=xlab, ylabel=ylab)
        ax.errorbar(xbin[num > num_thres], ybin[num > num_thres], yerr=ystd[num > num_thres], fmt='s', color='blue')
        ax.errorbar(xbin[num <= num_thres], ybin[num <= num_thres], yerr=ystd[num <= num_thres], fmt='s', color='lightblue')

        if yrng is None:
            max = np.floor(np.nanmax([np.nanmax(ybin) + np.nanmax(ystd),-np.nanmin(ybin) + np.nanmax(ystd)]))
            yrng = [-max,max]

        ax.set_ylim(yrng)
        if isinstance(binned_stats,xr.Dataset):
            ax.plot(binned_stats.attrs['xrng'], [0.0, 0.0], color='red')
        else:
            ax.plot(binned_stats['xrng'], [0.0, 0.0], color='red')

        if requirement is not None:
            ax.plot([0.0, 1.0], [-requirement, -requirement], color='gray')
            ax.plot([0.0, 1.0], [requirement, requirement], color='gray')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(16)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)

        if plot_num_in_bins:
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.semilogy(xbin[1:-1], num[1:-1], color='darkgreen')
            ax2.set_ylabel('Number of Observations')
            for item in ([ax2.yaxis.label]):
                item.set_fontsize(16)

        return fig,ax



       

    def stddev(self):
        stddev_map = np.sqrt(self.variance())  



               
            
'''the  routines below are deprecated.  Use the class defined above.'''


def calc_binned_stats(x, y, mask=None, bins=40, xrng=[0.0, 1.0],
                      num_hist_bins=40, verbose=False,
                      hist_range=[-1.0, 1.0], init=False,return_as_xr = False):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

    binned_x_means = np.zeros(bins)
    binned_y_means = np.zeros(bins)
    binned_y_stddev = np.zeros(bins)
    binned_num = np.zeros(bins)

    bin_size = (xrng[1] - xrng[0]) / bins
    overall_bias = np.nan
    overall_std = np.nan
    overall_rms = np.nan
    overall_num = 0.0
    edges = np.arange(hist_range[0], hist_range[1] + (hist_range[1] - hist_range[0]) / num_hist_bins,
                      (hist_range[1] - hist_range[0]) / num_hist_bins)
    xbin_centers = np.zeros(bins)
    xbin_edges = np.zeros(bins+1)

    ybin_centers = np.zeros(num_hist_bins)
    ybin_edges = np.zeros(num_hist_bins+1)

    if init is False:

        if mask is None:
            mask = np.zeros((x.shape))

        for j in range(0, bins):
            bin_start = xrng[0] + j * bin_size
            bin_end = xrng[0] + (j + 1) * bin_size
            xbin_centers[j] = 0.5*(bin_start+bin_end)
            xbin_edges[j] = bin_start
            xbin_edges[j+1] = bin_end
            with np.errstate(invalid='ignore'):
                z = np.all([(mask < 0.5), (x >= bin_start), (x <= bin_end)], axis=(0))
            x_in_bin = x[z]
            y_in_bin = y[z]
            n_in_bin = np.sum(z)
            if verbose:
                print(bin_start, bin_end, n_in_bin)

            if n_in_bin > 0:
                binned_x_means[j] = np.mean(x_in_bin)
                binned_y_means[j] = np.mean(y_in_bin)
                if n_in_bin > 2:
                    binned_y_stddev[j] = np.std(y_in_bin)
                else:
                    binned_y_stddev[j] = np.nan
            else:
                binned_x_means[j] = 0.5*(bin_start+bin_end)
                binned_y_means[j] = np.nan
                binned_y_stddev[j] = np.nan
            binned_num[j] = n_in_bin
            if n_in_bin > 0:
                hist, ybin_edges = np.histogram(y_in_bin, bins=num_hist_bins, range=hist_range)
                binned_hist[j, :] = hist
                ybin_centers = 0.5*(ybin_edges[0:-1] + ybin_edges[1:])

        # also find overall statistic for a large 0.1 to 0.9 bin
        
        ok = np.isfinite(y)
        overall_bias = np.mean(y[ok])
        overall_std = np.std(y[ok])
        overall_rms = np.sqrt(np.mean(np.square(y[ok])))
        overall_num = np.sum(ok)
    if return_as_xr:
        bin_centers = (edges[0:-1] + edges[1:])*0.5
        binned_stats = xr.DataArray(
                        data_vars={'binned_x_means':    (('xbin_centers'),binned_x_means),
                                   'binned_y_means':    (('xbin_centers'),binned_y_means),
                                   'binned_y_stddev':   (('xbin_centers'),binned_y_stddev),
                                   'binned_num':        (('xbin_centers'),binned_num)},
                        coords={'xbin_centers'  : xbin_centers,
                                'ybin_centers'  : ybin_centers,
                                'xbin_edges'    : xbin_edges,
                                'ybin_edges'    : ybin_edges},
                        attrs  ={'over_bias'    : overall_bias,
                                'overall_std'  : overall_std,
                                'overall_rms'  : overall_rms,
                                'overall_num'  : overall_num,
                                'xrng'         : xrng}
                               )
    else:
        for j in range(0, bins):
            bin_start = xrng[0] + j * bin_size
            bin_end = xrng[0] + (j + 1) * bin_size
            xbin_centers[j] = 0.5*(bin_start+bin_end)
            xbin_edges[j] = bin_start
            xbin_edges[j+1] = bin_end
        ybin_edges = np.arange(hist_range[0], hist_range[1] + (hist_range[1] - hist_range[0]) / num_hist_bins,
                      (hist_range[1] - hist_range[0]) / num_hist_bins)
        ybin_centers = 0.5*(ybin_edges[0:-1] + ybin_edges[1:])
        
        binned_stats = dict(binned_x_means=binned_x_means,
                            binned_y_means=binned_y_means,
                            binned_y_stddev=binned_y_stddev,
                            binned_num=binned_num,
                            xbin_centers = xbin_centers,
                            ybin_centers = ybin_centers,
                            xbin_edges = xbin_edges,
                            ybin_edges = ybin_edges,
                            overall_bias=overall_bias,
                            overall_std=overall_std,
                            overall_rms=overall_rms,
                            overall_num=overall_num,
                            binned_hist=binned_hist,
                            xrng=xrng)
    return binned_stats

def convert_to_xr(binned_stats_dict):

        
        binned_x_means = binned_stats_dict['binned_x_means']
        binned_y_means = binned_stats_dict['binned_y_means']
        binned_y_stddev = binned_stats_dict['binned_y_stddev']
        binned_num = binned_stats_dict['binned_num']
        overall_bias=binned_stats_dict['overall_bias']
        overall_std=binned_stats_dict['overall_std']
        overall_rms=binned_stats_dict['overall_rms']
        overall_num=binned_stats_dict['overall_num']
        binned_hist=binned_stats_dict['binned_hist']
        edges=binned_stats_dict['edges']
        xrng=binned_stats_dict['xrng']

        bins =  binned_x_means.shape[0]
        bin_size = (xrng[1]-xrng[0])/bins
        xbin_centers = np.zeros((bins))

        for j in range(0, bins):
            bin_start = xrng[0] + j * bin_size
            bin_end = xrng[0] + (j + 1) * bin_size
            xbin_centers[j] = 0.5*(bin_start+bin_end)

        ybin_centers = (edges[0:-1] + edges[1:])*0.5
        binned_x_means_xr = xr.DataArray(binned_x_means,
                        dims = ('xbin_centers'),
                        coords={'xbin_centers':xbin_centers})
        binned_y_means_xr = xr.DataArray(binned_y_means,
                        dims = ('xbin_centers'),
                        coords={'xbin_centers':xbin_centers})
        binned_y_stddev_xr = xr.DataArray(binned_y_stddev,
                        dims = ('xbin_centers'),
                        coords={'xbin_centers':xbin_centers})
        binned_num_xr = xr.DataArray(binned_num,
                        dims = ('xbin_centers'),
                        coords={'xbin_centers':xbin_centers})
                        
        return {'binned_x_means_xr' : binned_x_means_xr,
                'binned_y_means_xr' : binned_y_means_xr,
                'binned_y_stddev_xr' : binned_y_stddev_xr,
                'binned_num_xr'      : binned_num_xr}
        
        

def combine_binned_stats(binned_stats1, binned_stats2):
    import numpy as np
    import math

    old_settings = np.seterr()
    np.seterr(divide='ignore', invalid='ignore')

    shape1 = binned_stats1['binned_x_means'].shape
    shape2 = binned_stats2['binned_x_means'].shape
    if (shape1 != shape2):
        raise ValueError('Number of bins must be the same')
    if (binned_stats1['binned_x_means'].ndim != 1):
        raise ValueError('should be 1 d arrays')
    if (binned_stats1['binned_hist'].ndim != 2):
        raise ValueError('hist should be 2 d arrays')
    hist_shape1 = binned_stats1['binned_hist'].shape
    hist_shape2 = binned_stats2['binned_hist'].shape
    if (hist_shape1 != hist_shape2):
        raise ValueError('Number of bins in histograms must be the same')
    if (np.any((binned_stats1['ybin_edges'] - binned_stats2['ybin_edges']) > 0.000001)):
        raise ValueError('Histogram edges values must be the same')
    if (np.any((binned_stats1['xbin_edges'] - binned_stats2['xbin_edges']) > 0.000001)):
        raise ValueError('Bin edges values must be the same')

    n1 = (binned_stats1['binned_x_means'].shape)[0]

    xbin3 = np.zeros((n1))
    ybin3 = np.zeros((n1))
    dybin3 = np.zeros((n1))
    numbin3 = np.zeros((n1))
    binned_hist3 = np.zeros(hist_shape1)

    # fix up bad bins to make sure no NANs
    for binned_stats in [binned_stats1, binned_stats2]:
        # z = np.all([(land_mask == 0.0),(x >= bin_start),(x <= bin_end)],axis=(0))
        bad = np.all([(binned_stats['binned_num'] == 0), ~np.isfinite(binned_stats['binned_y_means'])], axis=(0))

        (binned_stats['binned_y_means'])[bad] = 0.0
        (binned_stats['binned_x_means'])[bad] = 0.0
        (binned_stats['binned_y_stddev'])[bad] = 0.0
        (binned_stats['binned_num'])[bad] = 0.0

        if ~np.isfinite(binned_stats['overall_bias']):
            binned_stats['overall_bias'] = 0.0
        if ~np.isfinite(binned_stats['overall_std']):
            binned_stats['overall_std'] = 0.0
        if ~np.isfinite(binned_stats['overall_rms']):
            binned_stats['overall_rms'] = 0.0

        bad2 = ~np.isfinite(binned_stats['binned_y_stddev'])
        (binned_stats['binned_y_stddev'])[bad2] = 0.0

    np.seterr(divide='ignore', invalid='ignore')
    xbin3 = (binned_stats1['binned_num'] * binned_stats1['binned_x_means'] +
             binned_stats2['binned_num'] * binned_stats2['binned_x_means']) / (
                    binned_stats1['binned_num'] + binned_stats2['binned_num'])

    ybin3 = (binned_stats1['binned_num'] * binned_stats1['binned_y_means'] +
             binned_stats2['binned_num'] * binned_stats2['binned_y_means']) / (
                    binned_stats1['binned_num'] + binned_stats2['binned_num'])

    numbin3 = (binned_stats1['binned_num'] + binned_stats2['binned_num'])

    dybin3 = np.sqrt(
        (binned_stats1['binned_num'] * binned_stats1['binned_y_stddev'] * binned_stats1['binned_y_stddev'] +
         binned_stats2['binned_num'] * binned_stats2['binned_y_stddev'] * binned_stats2['binned_y_stddev'] +
         binned_stats1['binned_num'] * (binned_stats1['binned_y_means'] - ybin3) * (
                 binned_stats1['binned_y_means'] - ybin3) +
         binned_stats2['binned_num'] * (binned_stats2['binned_y_means'] - ybin3) * (
                 binned_stats2['binned_y_means'] - ybin3)) /
        (binned_stats1['binned_num'] + binned_stats2['binned_num']))

    dybin3[numbin3 < 2] = float('NaN')
    xbin3[numbin3 < 1] = float('NaN')
    ybin3[numbin3 < 1] = float('NaN')

    tot_n1 = np.sum(binned_stats1['binned_num'])
    tot_n2 = np.sum(binned_stats2['binned_num'])

    tot_n3 = tot_n1 + tot_n2
    overall_bias3 = ((tot_n1 * binned_stats1['overall_bias']) + (tot_n2 * binned_stats2['overall_bias'])) / tot_n3
    overall_std3 = math.sqrt(tot_n1 * binned_stats1['overall_std'] * binned_stats1['overall_std'] +
                             tot_n2 * binned_stats2['overall_std'] * binned_stats2['overall_std'] +
                             tot_n1 * math.pow(binned_stats1['overall_bias'] - overall_bias3, 2) +
                             tot_n2 * math.pow(binned_stats2['overall_bias'] - overall_bias3, 2)) / math.sqrt(tot_n3)

    overall_rms3 = math.sqrt(overall_bias3 * overall_bias3 + overall_std3 * overall_std3)
    overall_num3 = tot_n3

    binned_hist3 = binned_stats1['binned_hist'] + binned_stats2['binned_hist']

    
    binned_stats = dict(binned_x_means=xbin3,
                            binned_y_means=ybin3,
                            binned_y_stddev=dybin3,
                            binned_num=numbin3,
                            overall_bias=overall_bias3,
                            overall_std=overall_std3,
                            overall_rms=overall_rms3,
                            overall_num = overall_num3,
                            binned_hist=binned_hist3,
                            xbin_centers = binned_stats1['xbin_centers'],
                            ybin_centers = binned_stats1['ybin_centers'],
                            xbin_edges   = binned_stats1['xbin_edges'],
                            ybin_edges   = binned_stats1['ybin_edges'],
                            xrng=binned_stats1['xrng'])

    np.seterr(**old_settings)
    return binned_stats


def plot_binned_means(binned_stats, yrng=[-0.5, 0.5], xlab='Mean', ylab='Binned Difference', title=' ', requirement=0.0,
                      plot_num_in_bins=False, num_thres=0):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

    xbin = binned_stats['binned_x_means']
    ybin = binned_stats['binned_y_means']
    ystd = binned_stats['binned_y_stddev']
    num  = binned_stats['binned_num']

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, title=title, xlabel=xlab, ylabel=ylab)
    ax.errorbar(xbin[num > num_thres], ybin[num > num_thres], yerr=ystd[num > num_thres], fmt='s', color='blue')
    ax.errorbar(xbin[num <= num_thres], ybin[num <= num_thres], yerr=ystd[num <= num_thres], fmt='s', color='lightblue')
    ax.set_ylim(yrng)
    if isinstance(binned_stats,xr.Dataset):
        ax.plot(binned_stats.attrs['xrng'], [0.0, 0.0], color='red')
    else:
        ax.plot(binned_stats['xrng'], [0.0, 0.0], color='red')

    if requirement > 0.0001:
        ax.plot([0.0, 1.0], [-requirement, -requirement], color='gray')
        ax.plot([0.0, 1.0], [requirement, requirement], color='gray')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(16)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)

    if plot_num_in_bins:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.semilogy(xbin[1:-1], num[1:-1], color='darkgreen')
        ax2.set_ylabel('Number of Observations')
        for item in ([ax2.yaxis.label]):
            item.set_fontsize(16)

    return fig
'''
if __name__ == '__main__':

    from matplotlib import pyplot as plt

    nc_file = 'C:/job_CCMP/compare_ccmp_vs_ASCAT/nc_files/era5_current_corrected_scl_01/w_ccmp_minus_ascat_b_binned_stats_01_2015_no_sat.1hr.nc'

    w_ccmp_minus_ascat_b = BinnedStat.from_netcdf(nc_file=nc_file)
    w_ccmp_minus_ascat_b.plt(yrng=[-5.0,5.0],ylab='CCMP - ASCAT B',title='era5_current_corrected_scl_01, January')
    nc_file = 'C:/job_CCMP/compare_ccmp_vs_ASCAT/nc_files/era5_current_corrected_scl_01/w_ccmp_minus_ascat_b_binned_stats_02_2015_no_sat.1hr.nc'

    w_ccmp_minus_ascat_b_new = BinnedStat.from_netcdf(nc_file=nc_file)
    w_ccmp_minus_ascat_b_new.plt(yrng=[-5.0,5.0],ylab='CCMP - ASCAT B',title='era5_current_corrected_scl_01, February')
    w_ccmp_minus_ascat_b.combine( w_ccmp_minus_ascat_b_new)

    w_ccmp_minus_ascat_b.plt(yrng=[-5.0,5.0],ylab='CCMP - ASCAT B',title='era5_current_corrected_scl_01, Jan + Feb')
    plt.show()
    print'''