'''class for defining, accumulating, and displaying 2D histograms

    depends on numpy, xarray, matplotlib
    depends on rss:plot_2d_hist'''

import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt

class Hist2D():
    

    def __init__(self,num_xbins = 200,min_xval = -10.0,max_xval=10.0,
                      num_ybins = None,min_yval = None,max_yval=None,
                      xname='',xunits='',yname='',yunits=None):
        self.num_xbins = num_xbins
        self.min_xval = min_xval
        self.max_xval = max_xval
        self.xname = xname
        self.xunits = xunits
        self.yname = yname
        if yunits is None:  # assume units are same if ynunits not provided
            yunits=xunits
        self.yunits = yunits
        self.size_xbin = (self.max_xval - self.min_xval)/self.num_xbins

        if num_ybins is None:
            self.num_ybins = self.num_xbins
        else:
            self.num_ybins = num_ybins

        if min_yval is None:
            self.min_yval = self.min_xval
        else:
            self.min_yval = min_yval

        if max_yval is None:
            self.max_yval = self.max_xval
        else:
            self.max_yval = max_yval 

        self.size_ybin = (self.max_yval - self.min_yval)/self.num_ybins

        # calculate bin edges and bin centers

        xedges = self.min_xval + np.arange(0,self.num_xbins+1)*self.size_xbin
        yedges = self.min_yval + np.arange(0,self.num_ybins+1)*self.size_ybin

        xcenters = 0.5*(xedges[0:num_xbins]+xedges[1:self.num_xbins+1])
        ycenters = 0.5*(yedges[0:num_xbins]+yedges[1:self.num_xbins+1])

        self.xedges = xedges
        self.yedges = yedges
        self.xcenters = xcenters
        self.ycenters = ycenters

        self.data = xr.Dataset(
            data_vars = {'n' : (('ycenters','xcenters'),np.zeros((self.num_ybins,self.num_xbins),dtype = np.float32))},
            coords = {'xcenters' : xcenters,
                      'ycenters' : ycenters,
                      'xedges'   : xedges,
                      'yedges'   : yedges},
            attrs = { 'hist_2d_xunits':self.xunits,
                      'hist_2d_yunits': self.yunits,
                      'hist_2d_xname':self.xname,
                      'hist_2d_yname':self.yname,
                      'hist_2d_xedges':self.xedges,
                      'hist_2d_yedges':self.yedges}) 

    def add(self,hist_to_add):
        '''add histogram data to existing histogram'''


        

        #for now, we assume hist_to_add is either an xarray, or a numpy array
        try:
            h = hist_to_add.values
        except:
            h = hist_to_add

        if type(h) is not np.ndarray:
            print('array to add must be xarray or numpy array')
            raise(ValueError)

        self.data['n']=self.data['n'] + h

    def coords_compatible(self,z):

        '''checks to make x and y cooordinates are the same
            returns True if so
            returns False if not compatible, or if z is not an
            instance of the Hist2D class '''

        try:
            assert(isinstance(z,Hist2D))
            assert(self.num_xbins == z.num_xbins)
            assert(self.num_ybins == z.num_ybins)
            assert(self.min_xval  == z.min_xval)
            assert(self.max_xval  == z.max_xval)
            assert(self.min_yval  == z.min_yval)
            assert(self.max_yval  == z.max_yval)
        except:
            return False

        return True

    def combine(self,hist_to_add):
        '''combines two histograms by adding the number in  each  bin
           checks to make sure histograms are compatible -- if not raises 
           ValueError'''
        try:
            assert(self.coords_compatible(hist_to_add))
        except:
            raise ValueError('Var to combine not compatible')

        h = hist_to_add.data['n']
        self.add(h)
        
        return self


    def add_data(self,x,y):

        z = np.all([np.isfinite(x),np.isfinite(y)],axis=(0))
        hist_to_add,xedges,yedges = np.histogram2d(x[z],y[z],
                                                    bins=[self.num_xbins,self.num_ybins],
                                                    range=[[self.min_xval, self.max_xval],
                                                           [self.min_yval, self.max_yval]])

        # because num_bins and ranges are from self, the resulting hist_to_add is automatically compatible.
        self.add(hist_to_add)

    def to_netcdf(self,filename = None):
        ''' writes histogram to a netcdf file'''

        #self.data is an xarray instance, so we can use xarray.to_netcdf

        self.data.to_netcdf(path  = filename)

    def from_netcdf(nc_file = None,var = 'w',compare_sat=None,xname='',xunits='',yname='',yunits=''):

        try:
            ds = xr.open_dataset(nc_file)
        except:
            return np.nan

        if var in ['w']:
            xedges   = ds['xedges_spd']
            xcenters = ds['xcenters_spd'] 
            yedges   = ds['yedges_spd']
            ycenters = ds['ycenters_spd'] 
        elif var in['u','v']:
            xedges   = ds['xedges_vec']
            xcenters = ds['xcenters_vec'] 
            yedges   = ds['yedges_vec']
            ycenters = ds['ycenters_vec'] 

        min_xval = np.nanmin(xedges)
        max_xval = np.nanmax(xedges)
        min_yval = np.nanmin(yedges)
        max_yval = np.nanmax(yedges)
        num_xbins = (xcenters.shape)[0]
        num_ybins = (ycenters.shape)[0]

        self = Hist2D(num_xbins = num_xbins,min_xval = min_xval,max_xval=max_xval,
                      num_ybins = num_ybins,min_yval = min_yval,max_yval= max_yval,
                      xname=xname,xunits=xunits,yname=xname,yunits=yunits)

        varname = f"n_{var}_ccmp_{compare_sat}"
        z = ds[varname]

        self.add(z)

        return self

    def as_DataArray(self):
        # returns data as DataArray, which can be combined into a larger xarray dataset easily
        da = xr.DataArray(self.data['n'].values,
                          dims = ('hist2d_ycenters','hist_2d_xcenters'),
                          coords = {'hist_2d_ycenters':self.ycenters,
                                    'hist_2d_xcenters':self.xcenters},
                          attrs = { 'hist_2d_xunits':self.xunits,
                                    'hist_2d_yunits': self.yunits,
                                    'hist_2d_xname':self.xname,
                                    'hist_2d_yname':self.yname,
                                    'hist_2d_xedges':self.xedges,
                                    'hist_2d_yedges':self.yedges}) 
        return da

    def as_dataset(self):
        return self.data

    def plot(self, title='', xtitle=None, ytitle=None, 
             aspect='equal', plot_diagonal=True, 
             plot_vert_medians=False,
             plot_horiz_medians=False,
             rangex = None,rangey= None):

        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        from plot_2d_hist import plot_2d_hist,averages_from_histograms


        if xtitle is None:
            xtitle = self.xname
        if ytitle is None:
            ytitle = self.yname

        if rangex is None:
            rangex=(self.min_xval,self.max_xval)

        if rangey is None:
            rangey = rangex

        fig, ax =plot_2d_hist(self.data['n'].values, self.data.attrs['hist_2d_xedges'] , self.data.attrs['hist_2d_yedges'] , 
                                title=title, xtitle=xtitle, ytitle=ytitle, 
                                nbins=self.num_xbins, 
                                z1_range=rangex,
                                z2_range=rangey, 
                                aspect='equal', plot_diagonal=plot_diagonal,
                                plot_horiz_medians=plot_horiz_medians,plot_vert_medians = plot_vert_medians)
        return fig, ax


