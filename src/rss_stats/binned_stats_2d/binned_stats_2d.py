import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
from rss_plotting.plot_2d_array import plot_2d_array

def plot_array_triple(*,num,
                        mean,
                        sdev,
                        xbin_edges,
                        ybin_edges,
                        mean_rng=[-0.8,0.8],
                        sdev_rng=[0.2,1.4],
                        var_name='',
                        figsize = [7.0,10.0],
                        ylab = 'Wind Speed (m/s)',
                        xlab = 'SST (C)'):
    fig,axs = plt.subplots(ncols=1,nrows=3,figsize = figsize)

    plot_2d_array(num, [1.0,np.nanmax(num)], xbin_edges, ybin_edges,
                title=var_name, xtitle=xlab, ytitle=ylab,   
                #aspect='equal',
                fig_in = fig,ax_in = axs[0],
                norm='Log',cmap = 'PuRd',plt_colorbar=True,fontsize=14,colorbar_title='Number of Collocs')

    plot_2d_array(mean, mean_rng, xbin_edges, ybin_edges,
                title='', xtitle=xlab, ytitle=ylab,   
                #aspect='equal',
                fig_in = fig,ax_in = axs[1],
                norm='Linear',cmap = 'PuOr_r',plt_colorbar=True,fontsize=14,colorbar_title='Mean SST Difference')

    plot_2d_array(sdev, sdev_rng, xbin_edges, ybin_edges,
                title='', xtitle=xlab, ytitle=ylab,   
                #aspect='equal',
                fig_in = fig,ax_in = axs[2],
                norm='Linear',cmap = 'magma',plt_colorbar=True,fontsize=14,colorbar_title='Std. Dev. SST Difference')

    fig.subplots_adjust(bottom=0.1,top=0.9,left = 0.1,right=0.9,hspace=0.35)

    return fig,axs


class BinnedStat2D():
    '''
    Class to accumulate binned statistics

    Attributes
    ----------
    num_xbins : int
        number of bins in x direction
    num_ybins : int
        number of bins in y direction
    x_rng : [float,float]
        lower and upper limit to be put into x bins
    y_rng : [float,float]
        lower and upper limit to be put into y bins

    binned_tot: numpy float64 array[num_ybins,num_xbins]
        total of the values acculated into each bin
    binned_totsqr: numpy float64 array[num_ybins,num_xbins]
        total of the square of the y values acculated into each bin
    binned_num: numpy float64 array[num_ybins,num_xbins]
        total number of data points in each bin

    overall_tot : float64
        total of all values
    overall_totsqr : float64
        total of the sqaure of all values
    overall_num : float 64
        total number of datapoints accumulated

    xbin_size,ybin_size : float64
        width of each bin, the same for each bin
    xbin_centers, ybin_centers : np float64 array[num_bins]
        center of each bin (different from average x values!)
    xbin_edges,ybin_edges : np float64 array[num_bins+1]
        edges of each bin

    Methods
    --------
    __init__(num_ybin=40,num_xbin=30,x_rng=[0.0,1.0],y_rng=[0.0,1.0])
        initializes a class instance
    add_data(x,y,z, mask=None,verbose=False)
        adds data at points x,y with values y to the binned stat object

    plot()

    '''

    def __init__(self, num_xbins = 40,x_rng=[0.0,1.0],num_ybins = 40,y_rng=[0.0,1.0],var_name=' '):

        self.num_ybins = num_ybins
        self.y_rng = y_rng
        self.num_xbins = num_xbins
        self.x_rng = x_rng
        self.var_name = var_name

       
        self.binned_tot             = np.zeros((self.num_ybins,self.num_xbins))
        self.binned_totsqr          = np.zeros((self.num_ybins,self.num_xbins))
        self.binned_num             = np.zeros((self.num_ybins,self.num_xbins))

        self.overall_tot            = 0.0
        self.overall_totsqr         = 0.0
        self.overall_num            = 0.0

        self.xbin_size = (self.x_rng[1]-self.x_rng[0])/self.num_xbins
        self.xbin_centers = np.arange(self.x_rng[0] + self.xbin_size/2.0,self.x_rng[1],self.xbin_size) # don't subtract from end point to make sure we get last point
        self.xbin_edges = np.arange(self.x_rng[0],self.x_rng[1]+0.1*self.xbin_size,self.xbin_size)

        self.ybin_size = (self.y_rng[1]-self.y_rng[0])/self.num_ybins
        self.ybin_centers = np.arange(self.y_rng[0] + self.ybin_size/2.0,self.y_rng[1],self.ybin_size) # don't subtract from end point to make sure we get last point
        self.ybin_edges = np.arange(self.y_rng[0],self.y_rng[1]+0.1*self.ybin_size,self.ybin_size)

    '''def from_dataset(ds = None):
        #creates BinnedStat object from a binned stat xarray dataset
        try:
            xbin_centers = ds['binned_stat_xbin_centers'] 
        except:
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
    '''
    '''
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
    '''
    def combine(self,self2):

        try:
            assert(np.all(self.xbin_edges == self2.xbin_edges))
            assert(np.all(self.ybin_edges == self2.ybin_edges))
        except:
            raise ValueError('bin definitions to not match, can not combine')

        self.overall_num =     self.overall_num +     self2.overall_num
        self.overall_tot =     self.overall_tot +     self2.overall_tot
        self.overall_totsqr =  self.overall_totsqr +  self2.overall_totsqr

        self.binned_num =      self.binned_num +      self2.binned_num
        self.binned_tot =    self.binned_tot +    self2.binned_tot
        self.binned_totsqr = self.binned_totsqr + self2.binned_totsqr

        return self


    def add_data(self,x,y,z,mask=None,verbose=False):
        
        if mask is None:
            mask = np.zeros_like(x)

        for j in range(0, self.num_ybins):

            ybin_start = self.ybin_edges[j]
            ybin_end =   self.ybin_edges[j+1]

            for i in range(0,self.num_xbins):
                xbin_start = self.xbin_edges[i]
                xbin_end =   self.xbin_edges[i+1]

            
            
                with np.errstate(invalid='ignore'):
                     z_ok = np.all([(mask < 0.5), 
                                    (x >= xbin_start), (x <= xbin_end),
                                    (y >= ybin_start), (y <= ybin_end),
                                        (np.isfinite(z))], axis=(0))
            
                n_in_bin = np.sum(z_ok)
                if verbose:
                    print(ybin_start, ybin_end, xbin_start, xbin_end, n_in_bin)

                if n_in_bin > 0:
                    self.binned_tot[j,i] = self.binned_tot[j,i] + np.sum(z[z_ok])
                    self.binned_totsqr[j,i] = self.binned_totsqr[j,i] + np.sum(np.square(z[z_ok]))
                    self.binned_num[j,i] = self.binned_num[j,i] + n_in_bin

        with np.errstate(invalid='ignore'):
            z_ok = np.all([(mask < 0.5), 
                           (x >= self.xbin_edges[0]), (x <= self.xbin_edges[-1]),
                           (y >= self.ybin_edges[0]), (y <= self.ybin_edges[-1]),
                           (np.isfinite(z))], axis=(0))
            self.overall_tot += np.sum(z[z_ok])    
            self.overall_totsqr += np.sum(np.square(z[z_ok]))
            self.overall_num += np.sum(z_ok)
        
    def as_DataArray(self):
        # return as an xarray DataArray
        # for this representation, binned hist is ignored

        temp = np.zeros((self.num_ybins,self.num_xbins,3))
        temp[:,:,0] = self.binned_num
        temp[:,:,1] = self.binned_tot
        temp[:,:,2] = self.binned_totsqr

        binned_stats_xr = xr.DataArray(
            temp,
            dims = ('binned_stat_ybin_centers','binned_stat_xbin_centers','binned_stat_type'),

            coords={'binned_stat_ybin_centers'      : self.ybin_centers,
                    'binned_stat_xbin_centers'      : self.xbin_centers,
                    'binned_stat_type'              : ['num','tot','totsqr']
                    },
            attrs ={'overall_tot'       : self.overall_tot,
                    'overall_totsqr'    : self.overall_totsqr,
                    'overall_num'       : self.overall_num,
                    'binned_stat_ybin_edges'        : self.ybin_edges,
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

    def from_netcdf(ncfilename,var_name = None):

        da = xr.open_dataarray(ncfilename)

        num_xbins = len(da.binned_stat_xbin_centers)
        x_rng = [da.binned_stat_xbin_edges[0],da.binned_stat_xbin_edges[-1]]

        num_ybins = len(da.binned_stat_ybin_centers)
        y_rng = [da.binned_stat_ybin_edges[0],da.binned_stat_ybin_edges[-1]]


        #initialize binned_stat2D object
        if var_name is None:
            var_name = 'temp' 
        z = BinnedStat2D(num_xbins = num_xbins,x_rng=x_rng,num_ybins=num_ybins,y_rng=y_rng,var_name=var_name)

        z.binned_num    = da.values[:,:,0]
        z.binned_tot    = da.values[:,:,1]
        z.binned_totsqr = da.values[:,:,2]

        z.overall_tot            = da.overall_tot
        z.overall_totsqr         = da.overall_totsqr
        z.overall_num            = da.overall_num

        z.binned_stat_xbin_centers = da.binned_stat_xbin_centers
        z.binned_stat_ybin_centers = da.binned_stat_ybin_centers

        z.binned_stat_xbin_edges = da.binned_stat_xbin_edges
        z.binned_stat_ybin_edges = da.binned_stat_ybin_edges

        z.attrs = da.attrs

        return z



    def calc_stats(self,return_as_xr = False):  
        '''returns binned means, stddevs, xbinned means based on current state of object.
           unless requested, returns a dictionary of mostly numpy arrays.  if requested, returns an
           xarray dataset'''
        binned_means = np.full((self.num_ybins,self.num_xbins),np.nan)
        binned_stddev = np.full((self.num_ybins,self.num_xbins),np.nan)
        
        ok = self.binned_num > 0
        if np.sum(ok) > 0:
            binned_means[ok] = self.binned_tot[ok]/self.binned_num[ok]

        sum_of_sqr_dev = np.full((self.num_ybins,self.num_xbins),np.nan)
        ok = self.binned_num > 2
        if np.sum(ok) > 0:
            sum_of_sqr_dev[ok] = self.binned_totsqr[ok] - (np.square(self.binned_tot[ok]))/self.binned_num[ok]
            binned_stddev[ok] = np.sqrt(sum_of_sqr_dev[ok]/(self.binned_num[ok] - 1))

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
                        data_vars={'binned_x_means':    (('ybin_centers','xbin_centers'),binned_means),
                                   'binned_y_stddev':   (('ybin_centers','xbin_centers'),binned_stddev),
                                   'binned_num'     :   (('ybin_centers','xbin_centers'),self.binned_num)},
                        coords={'ybin_centers'      : self.ybin_centers,
                                'ybin_edges'        : self.ybin_edges,
                                'xbin_centers'      : self.xbin_centers,
                                'xbin_edges'        : self.xbin_edges},
                        attrs ={'overall_mean'      : overall_mean,
                                'overall_stddev'    : overall_stddev,
                                'overall_rms'       : overall_rms,
                                'overall_num'       : overall_num}
                               )
        else:
            binned_stats = dict(binned_means=binned_means,
                            binned_stddev=binned_stddev,
                            binned_num=self.binned_num,
                            overall_mean=overall_mean,
                            overall_stddev=overall_stddev,
                            overall_rms=overall_rms,
                            overall_num=self.overall_num,
                            ybin_centers = self.ybin_centers,
                            ybin_edges = self.ybin_edges,
                            xbin_centers = self.xbin_centers,
                            xbin_edges = self.xbin_edges,
                            yrng = [self.ybin_edges[0],self.ybin_edges[-1]],
                            xrng = [self.xbin_edges[0],self.xbin_edges[-1]])
        
        return binned_stats
       
    def plot(self, yrng=None, xrng=None,xlab='', ylab='', title='', panel_label=None,panel_label_loc=[0.05,0.9],
                num_thres=0, fig_in = None, ax_in = None, fontsize=16, as_percent=False,mean_rng=[-0.8,0.8],stddev_rng=[0.2,1.4]):

        binned_stats = self.calc_stats()

        mean = binned_stats['binned_means']
        sdev = binned_stats['binned_stddev']
        num  = binned_stats['binned_num']

        fig,axs = plot_array_triple(num=num,mean=mean,sdev=sdev,
                                    xbin_edges = self.xbin_edges,
                                    ybin_edges = self.ybin_edges,
                                    mean_rng=[-0.8,0.8],
                                    sdev_rng=[0.2,1.4],
                                    var_name=self.var_name)

        return fig,axs    
        

if __name__ == '__main__':

    z_2d = BinnedStat2D(num_xbins = 20,x_rng=[0,5],num_ybins=30,y_rng=[0,3],var_name='Test Name')

    x = np.random.rand(10000)*5
    y = np.random.rand(10000)*3
    z = 1.0+np.random.rand(10000)

    z_2d.add_data(x,y,z,verbose=False)
    z_2d.plot(xlab='SST',ylab='Wind Speed')

    nc_file = 'junk.nc'
    z_2d.to_netcdf(nc_file)

    z_2d_in = BinnedStat2D.from_netcdf(nc_file)
    z_2d_in.plot(xlab='SST',ylab='Wind Speed')
    print()

