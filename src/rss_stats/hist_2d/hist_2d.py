'''class for defining, accumulating, and displaying 2D histograms

    depends on numpy, xarray, matplotlib
    depends on rss:plot_2d_hist'''

import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from rss_plotting.plot_2d_hist import plot_2d_hist

class Hist2D():
    

    def __init__(self,num_xbins = 200,min_xval = -10.0,max_xval=10.0,
                      num_ybins = None,min_yval = None,max_yval=None,
                      xname='',xunits='',yname='',yunits=None,no_var=False):
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

        if no_var:
            self.data = xr.Dataset(
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
        else:
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

    def add(self,hist_to_add,name='n'):
        '''add histogram data to existing histogram'''

        #for now, we assume hist_to_add is either an xarray, or a numpy array
        try:
            h = hist_to_add.values
        except:
            h = hist_to_add

        if type(h) is not np.ndarray:
            print('array to add must be xarray or numpy array')
            raise(ValueError)

        self.data[name]+= h

    def compatible(self,z):

        '''checks to make x and y cooordinates are the same
            returns True if so
            returns False if not compatible, or if z is not an
            instance of the Hist2D class '''
        hist_compatible = True
        
        if isinstance(z,Hist2D):
            attr_to_check = ['num_xbins','num_ybins','min_xval','max_xval','min_yval','max_yval']
            for attr in attr_to_check:
                try:
                    if getattr(self,attr) != getattr(z,attr):
                        hist_compatible = False
                except KeyError:
                    hist_compatible = False  #missing critical key
        else:
            hist_compatible = False

        return hist_compatible

    def combine(self,hist_to_add,name='n'):
        '''combines two histograms by adding the number in  each  bin
           checks to make sure histograms are compatible -- if not raises 
           ValueError'''
        if  not self.compatible(hist_to_add):
            raise ValueError('histogram to combine not compatible')

        if name == 'ALL':
            name_list = list(self.data.data_vars.keys())
        else:
            name_list = [name]
        
        for name_to_do in name_list:
            h = hist_to_add.data[name_to_do]
            self.add(h,name=name_to_do)
        
        return self

    def add_data_var(self,name=None):
        if name is not None:
            self.data[name] = (('ycenters','xcenters'),np.zeros((self.num_ybins,self.num_xbins),dtype = np.float32))

    def add_data(self,x,y,name='n'):

        z = np.all([np.isfinite(x),np.isfinite(y)],axis=(0))
        hist_to_add,xedges,yedges = np.histogram2d(x[z],y[z],
                                                    bins=[self.num_xbins,self.num_ybins],
                                                    range=[[self.min_xval, self.max_xval],
                                                           [self.min_yval, self.max_yval]])

        # because num_bins and ranges are from self, the resulting hist_to_add is automatically compatible.
        self.add(hist_to_add,name)

    def to_1D(self,name='n',axis_to_sum='x'):
        # converts 2D array to 1D array by summing along one of the axes
        from rss_stats.hist_1d import Hist1D

        if axis_to_sum.lower() == 'x':
            hist1d = np.sum(self.data[name].values,axis=0)
            num_bins = self.num_ybins
            min_val = self.min_yval
            max_val = self.max_yval
            units = self.yunits
        
           
        elif axis_to_sum.lower() == 'y':
            hist1d = np.sum(self.data[name].values,axis=1)
            num_bins = self.num_xbins
            min_val = self.min_xval
            max_val = self.max_xval
            units = self.xunits

        else:
            raise ValueError('axis to sum must be x or y')
 
        h = Hist1D(num_xbins=num_bins,
                          min_xval = min_val,
                          max_xval = max_val,
                          units=units,
                          name = name)

        h.add(hist1d,name=name)

        return h


    def to_netcdf_old(self,filename = None):
        ''' writes histogram to a netcdf file'''

        #self.data is an xarray instance, so we can use xarray.to_netcdf

        self.data.to_netcdf(path  = filename)

    def to_netcdf(self,*,ncfilename,name='ALL'):
        ''' writes histogram to a netcdf file'''

        from netCDF4 import Dataset as netcdf_dataset

        root_grp = netcdf_dataset(ncfilename,'w',format = 'NETCDF4')

        root_grp.createDimension('xbin_centers',self.num_xbins)
        root_grp.createDimension('xbin_edges',self.num_xbins+1)
        root_grp.createDimension('ybin_centers',self.num_ybins)
        root_grp.createDimension('ybin_edges',self.num_ybins+1)
       
        xbin_centers  = root_grp.createVariable('xbin_centers','f4',('xbin_centers',))
        xbin_centers.units = self.data.attrs['hist_2d_xunits']
        xbin_centers.longname = self.data.attrs['hist_2d_xname']
        xbin_edges    = root_grp.createVariable('xbin_edges','f4',('xbin_edges',)) 
        ybin_centers  = root_grp.createVariable('ybin_centers','f4',('ybin_centers',))
        ybin_centers.units = self.data.attrs['hist_2d_yunits']
        ybin_centers.longname = self.data.attrs['hist_2d_yname']

        ybin_edges    = root_grp.createVariable('ybin_edges','f4',('ybin_edges',)) 

        if name == 'ALL':
            name_list = list(self.data.data_vars.keys())
        else:
            name_list = [name]

        var_arr = []
        for name_to_do in name_list:
            var_arr.append(root_grp.createVariable(name_to_do,'f8',('ybin_centers','xbin_centers',)))

        xbin_centers[:] = self.xcenters
        xbin_edges[:] = self.xedges
        ybin_centers[:] = self.ycenters
        ybin_edges[:] = self.yedges

        for ivar,name_to_do in enumerate(name_list):
            var_arr[ivar][:,:] = self.data[name_to_do].values

        root_grp.close()


    def from_netcdf(nc_file = None,varname = None,var = 'w',compare_sat=None,xname='',xunits='',yname='',yunits=''):

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

        if varname is None:
            varname = f'n_{var}_model_sat'

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

    def plot(self, name='n', title='', xtitle=None, ytitle=None, 
             aspect='equal', plot_diagonal=True,as_percent = False, 
             plot_vert_medians=False,
             plot_horiz_medians=False,
             rangex = None,rangey= None,num_scale=10000.0,reduce_max = 1.0,
             fig_in = None,ax_in = None,norm='Log',cmap = 'ocean_r',plt_colorbar=True,
             fontsize=16,return_im=False,
             panel_label=None,panel_label_loc=[0.07,0.9],
             transpose=False):

        if xtitle is None:
            xtitle = self.xname
        if ytitle is None:
            ytitle = self.yname

        if rangex is None:
            rangex=(self.min_xval,self.max_xval)

        if rangey is None:
            rangey = rangex
        
        edge_factor = 1.0
        if as_percent:
            rangex = 100.0*np.array(rangex)
            rangey = 100.0*np.array(rangey)
            edge_factor=100.0

        if transpose:
            return plot_2d_hist(np.transpose(self.data[name].values), self.data.attrs['hist_2d_xedges']*edge_factor , self.data.attrs['hist_2d_yedges']*edge_factor , 
                                    title=title, xtitle=xtitle, ytitle=ytitle, 
                                    x_range=rangex,
                                    y_range=rangey, 
                                    num_scale=num_scale,
                                    reduce_max=reduce_max,
                                    aspect=aspect, 
                                    plot_diagonal=plot_diagonal,
                                    plot_horiz_medians=plot_horiz_medians,
                                    plot_vert_medians = plot_vert_medians,
                                    fig_in = fig_in,
                                    ax_in = ax_in,
                                    norm=norm,
                                    cmap = cmap,
                                    plt_colorbar=plt_colorbar,
                                    fontsize=fontsize,
                                    return_im = return_im,
                                    panel_label=panel_label,
                                    panel_label_loc=panel_label_loc)
        else:
            return plot_2d_hist(self.data[name].values, self.data.attrs['hist_2d_xedges']*edge_factor , self.data.attrs['hist_2d_yedges']*edge_factor , 
                                title=title, xtitle=xtitle, ytitle=ytitle, 
                                x_range=rangex,
                                y_range=rangey, 
                                num_scale=num_scale,
                                reduce_max=reduce_max,
                                aspect=aspect, 
                                plot_diagonal=plot_diagonal,
                                plot_horiz_medians=plot_horiz_medians,
                                plot_vert_medians = plot_vert_medians,
                                fig_in = fig_in,
                                ax_in = ax_in,
                                norm=norm,
                                cmap = cmap,
                                plt_colorbar=plt_colorbar,
                                fontsize=fontsize,
                                return_im = return_im,
                                panel_label=panel_label,
                                panel_label_loc=panel_label_loc)
        
       


