import numpy as np
import xarray as xr

class BinnedStat():
    '''
    Class to accumulate binned statistics

    Attributes
    ----------
    num_bins : int
        number of bins
    x_rng : [float,float]
        lower and upper limit to be put into bins
    binned_x_tot: numpy float64 array[num_bins]
        total of the x values acculated into each bin
    binned_y_tot: numpy float64 array[num_bins]
        total of the y values acculated into each bin
    binned_y_totsqr: numpy float64 array[num_bins]
        total of the square of the y values acculated into each bin
    binned_num: numpy float64 array[num_bins]
        total number of data points in each bin
    overall_tot : float64
        total of all y values
    overall_totsqr : float64
        total of the sqaure of all y values
    overall_num : float 64
        total number of datapoints accumulated
    xbin_size : float64
        width of each bin, the same for each bin
    xbin_centers : np float64 array[num_bins]
        center of each bin (different from average x values!)
    xbin_edges : np float64 array[num_bins+1]
        edges of each bin

    Methods
    --------
    __init__(num_bin=40,x_rng=[0.0,1.0])
        initializes a class instance
    add_data(x,y,mask=None,verbose=False)
        adds data at points x with values y to the binned stat object
    plot()

    '''

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

        return self


    def add_data(self,x,y,mask=None,verbose=False):
        
        if mask is None:
            mask = np.zeros_like(x)

        for j in range(0, self.num_bins):

            bin_start = self.xbin_edges[j]
            bin_end =   self.xbin_edges[j+1]
            
            with np.errstate(invalid='ignore'):
                z = np.all([(mask < 0.5), (x >= bin_start), (x <= bin_end),(np.isfinite(y))], axis=(0))
            
            n_in_bin = np.sum(z)
            if verbose:
                print(bin_start, bin_end, n_in_bin)

            if n_in_bin > 0:
                self.binned_x_tot[j] = self.binned_x_tot[j] + np.sum(x[z])
                self.binned_y_tot[j] = self.binned_y_tot[j] + np.sum(y[z])
                self.binned_y_totsqr[j] = self.binned_y_totsqr[j] + np.sum(np.square(y[z]))
                self.binned_num[j] = self.binned_num[j] + n_in_bin

        with np.errstate(invalid='ignore'):
            z = np.all([(mask < 0.5), (x >= self.xbin_edges[0]), (x <= self.xbin_edges[-1]),(np.isfinite(y))], axis=(0))
            self.overall_tot = np.sum(y[z])    
            self.overall_totsqr = np.sum(np.square(y[z]))
            self.overall_num = np.sum(z)
        
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
                    'binned_stat_xbin_edges'        : self.xbin_edges
                    }
                    )
                    
        return binned_stats_xr

    def to_netcdf(self,ncfilename=None):

        if ncfilename is not None:
            self.as_DataArray().to_netcdf(ncfilename)
            return 0
        else:
            return -1


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

    def calc_stats_over_range(self,xrange=[0.0,20.0]): 
        print 
        '''returns mean,rms and stddev over the specified xrange.
           returns a dictionary
           '''
        

        #first, figure out what bins to include
        bin_width = self.xbin_edges[1] - self.xbin_edges[0] 
        bin_ok = np.all([(self.xbin_edges >= xrange[0]),(self.xbin_edges <= xrange[1]-bin_width)],axis=0)
        bin_ok = bin_ok[0:-1]
        y_tot = np.sum(self.binned_y_tot[bin_ok])
        y_totsqr = np.sum(self.binned_y_totsqr[bin_ok])
        num = np.sum(self.binned_num[bin_ok])

        mean = y_tot/num
        rms    = np.sqrt(y_totsqr/num)
        stddev = np.sqrt((num*y_totsqr - np.square(y_tot))/(num*(num-1)))

        stats = dict(num=num,
                     mean=mean,
                     rms=rms,
                     stddev=stddev)

        return stats
    
        
    def plot(self, yrng=None, xrng=None,xlab='Wind', ylab='Binned Difference', title=' ', requirement=None,
                      plot_num_in_bins=False, num_thres=0,fig_in = None,ax_in = None,fontsize=16,as_percent=False):
        import numpy as np
        import xarray as xr
        import matplotlib.pyplot as plt

        binned_stats = self.calc_stats()

        xbin = binned_stats['binned_x_means']
        ybin = binned_stats['binned_y_means']
        ystd = binned_stats['binned_y_stddev']
        num  = binned_stats['binned_num']

        if as_percent:
            ybin = 100.0*ybin
            ystd = 100.0*ystd
            requirement = 100.0*requirement
            yrng=100.0*np.array(yrng)
            xrng=100.0*np.array(xrng)
            xbin = 100.0*xbin

        if fig_in == None:
            fig = plt.figure(figsize=(9, 5))
        else:
            fig = fig_in
        
        if ax_in == None:
            ax = fig.add_subplot(111)
        else:
            ax = ax_in
        
        ax.errorbar(xbin[num > num_thres], ybin[num > num_thres], yerr=ystd[num > num_thres], fmt='s', color='blue')
        ax.errorbar(xbin[num <= num_thres], ybin[num <= num_thres], yerr=ystd[num <= num_thres], fmt='s', color='lightblue')

        if yrng is None:
            max = np.floor(np.nanmax([np.nanmax(ybin) + np.nanmax(ystd),-np.nanmin(ybin) + np.nanmax(ystd)]))
            yrng = [-max,max]

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(yrng)
        if xrng is not None:
            ax.set_xlim(xrng)
        if isinstance(binned_stats,xr.Dataset):
            ax.plot(binned_stats.attrs['xrng'], [0.0, 0.0], color='red')
        else:
            ax.plot(binned_stats['xrng'], [0.0, 0.0], color='red')

        if requirement is not None:
            ax.plot(xrng, [-requirement, -requirement], color='gray')
            ax.plot(xrng, [requirement, requirement], color='gray')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(fontsize)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize-4)

        if plot_num_in_bins:
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.semilogy(xbin[1:-1], num[1:-1], color='darkgreen')
            ax2.set_ylabel('Number of Observations')
            for item in ([ax2.yaxis.label]):
                item.set_fontsize(fontsize)

        return fig,ax




