'''class for defining, accumulating, and displaying 1D histograms'''
import numpy as np
from numpy.random.mtrand import noncentral_chisquare
import xarray as xr 
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class Hist1D():

    def __init__(self,num_xbins = 200,min_xval = -10.0,max_xval=10.0,
                 name=None,units=None,no_var = False):
        self.num_xbins = num_xbins
        self.min_xval = min_xval
        self.max_xval = max_xval
        self.name = name
        self.units = units
        self.size_xbin = (self.max_xval - self.min_xval)/self.num_xbins

        xedges = self.min_xval + np.arange(0,self.num_xbins+1)*self.size_xbin
        xcenters = 0.5*(xedges[0:num_xbins]+xedges[1:self.num_xbins+1])
        
        if name is None:
            name = 'n'

        if no_var:
            self.data = xr.Dataset(
                coords = {'xedges'   : xedges,
                        'xcenters' : xcenters}) 
        else:
            self.data = xr.Dataset(
                data_vars = {'n' : (('xcenters'),np.zeros((self.num_xbins),dtype = np.float32))},
                coords = {'xedges'   : xedges,
                        'xcenters' : xcenters}) 

    def add_data_var(self,name=None):
        if name is not None:
            self.data[name] = (('xcenters'),np.zeros((self.num_xbins),dtype = np.float32))

    def get_cumulative(self,name='n'):

        x_vals = self.data['xedges']
        z = np.insert(self.data[name].values,0,0.0)
        cumsum = np.cumsum(z)
        return x_vals,cumsum

    def get_values(self,name='n'):

        x_vals = self.data['xedges']
        return x_vals,self.data[name].values

    def get_tot_num(self,name='n'):
        tot_num = np.sum(self.data[name].values)
        return tot_num

    def add(self,hist_to_add,name='n'):
        '''add histogram data to existing histogram'''
        
        if isinstance(hist_to_add, Hist1D):
            # if it is a Hist1D instance, use the existing combine method.
            self.combine(hist_to_add,name=name)
        else:
            # for now, we assume hist_to_add is either and xarray, or a numpy array
            # takes bin match up on faith!
            #to do: for the xarray case, chould add code to check bins
            if type(hist_to_add) is np.ndarray:
                h = hist_to_add
            elif isinstance(hist_to_add,xr.DataArray):
                h = hist_to_add.data
            else:
                h = hist_to_add  # this  is the hail mary case

            self.data[name]=self.data[name] + h

    def add_data(self,x,name='n'):
        z = np.isfinite(x)
        hist_to_add,xedges = np.histogram(x[z],
                                        bins=self.num_xbins,
                                        range=[self.min_xval, self.max_xval])
        self.add(hist_to_add,name=name)

    def compatible(self,z):

        '''checks to make x and y cooordinates are the same
            returns True if so
            returns False if not compatible, or if z is not an
            instance of the Hist2D class '''
        hist_compatible = True
        

        if isinstance(z,Hist1D):
            attr_to_check = ['num_xbins','min_xval','max_xval']
            for attr in attr_to_check:
                try:
                    if getattr(self,attr) != getattr(z,attr):
                        print(f'attr missmatch {attr}: {getattr(self,attr)}, {getattr(z,attr)}')
                        hist_compatible = False
                except KeyError:
                    print(f'Missing attr {attr}')
                    hist_compatible = False  #missing critical attr
        else:
            hist_compatible = False

        return hist_compatible

    def combine(self,self2,name='n',name2 = 'n'):

        #make sure histograms are compatible
        # try:
        #     hists_compatible = self.compatible(self2)
        # except:
        #     raise ValueError('Hist1D compatible failed, can not combine')

        # if not hists_compatible:
        #     raise ValueError('Hist1D objects not compatible, can not combine')

        hist_to_add = self2.data[name2]
        self.add(hist_to_add,name=name)

    def to_netcdf(self,filename = None):

        self.data.to_netcdf(path  = filename)

    def from_netcdf(nc_file = None,varname = None,name='',units='m/s'):

        try:
            ds = xr.open_dataset(nc_file)
        except:
            raise FileExistsError(f'File {nc_file} not found or not netcdf file')

        

        if '_w_' in varname:
            xedges   = ds['xedges_spd']
            xcenters = ds['xcenters_spd'] 
        elif (('_u_' in varname) or ('_v_' in varname)):
            xedges   = ds['xedges_vec']
            xcenters = ds['xcenters_vec'] 
        else:
            print('Warning -- var type no understood - assuming scalar')
            xedges   = ds['xedges_spd']
            xcenters = ds['xcenters_spd'] 

        min_xval = np.nanmin(xedges)
        max_xval = np.nanmax(xedges)
        num_xbins = (xcenters.shape)[0]

        if name is None:
            name=varname
        self = Hist1D(num_xbins = num_xbins,min_xval = min_xval,max_xval=max_xval,
                 name=name,units=units)


        if varname is None:
            varname = f"n_{var}_ccmp_{compare_sat}"
            
        z = ds[varname]

        self.add(z)

        return self

    def match_to(self,hist_to_match,attenuate_near_zero = False,atten_thres = 5.0,name='n'):
        '''constructs an additive matching function that when added to the data that resulted in the histogram in self, with result in a
        new dataset with a histigram that matches the histogram  in hist2'''

        x,hist_in_cumsum =  self.get_cumulative(name=name)
        x,hist_to_match_cumsum = hist_to_match.get_cumulative(name=name)

        hist_in_cumsum = hist_in_cumsum/hist_in_cumsum[-1]
        hist_to_match_cumsum = hist_to_match_cumsum/hist_to_match_cumsum[-1]

        x_values = self.data.xedges.values
        z = np.interp(hist_in_cumsum,hist_to_match_cumsum,x_values)

        z1 = 0.5*(z[0:-1]+z[1:])
        x1 = 0.5*(x_values[0:-1]+x_values[1:]) 


        if  attenuate_near_zero:
            diff = z1-x1
            # reduce differences below threshold set by atten_thres in m/s

            squelcher = np.ones_like(z1)
            num = np.sum(np.abs(x1 < atten_thres))
            a = -1.0/(atten_thres*atten_thres)
            b = 2.0/atten_thres

            squelcher[0:num] = a*np.square(x1[0:num]) + b*x1[0:num]  
            diff = diff*squelcher
            z1 = x1+diff

        z1[self.data['n'].values < 10.0] = np.nan

        return x1,z1

    def plot(self, fig = None, 
                ax = None, 
                name='n',
                rangex = None,
                rangey = None,
                title=None, 
                xtitle=None, 
                ytitle=None,
                label=None,
                semilog=False,
                fontsize=16,
                panel_label=None,
                panel_label_loc=[0.04,0.92]):
        if label is None:
            label = name

        if xtitle is None:
            xtitle = self.units
        if ytitle is None:
            ytitle = 'Number of Obs'
        if title is None:
            title = ''

        created_fig = False
        if ax is None:
            fig,ax = plt.subplots(figsize=(10,8))
            ax.set_xlim([self.min_xval,self.max_xval])
            created_fig = True

        if title is not None:
            ax.set_title(title)
        if xtitle is not None:
            ax.set_xlabel(xtitle)
        if ytitle is not None:
            ax.set_ylabel(ytitle)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(fontsize)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(0.8*fontsize)
        
        left,right = self.data['xedges'][:-1],self.data['xedges'][1:]
        X = np.array([left,right]).T.flatten()
        

        y = self.data[name]
        Y = np.array([y,y]).T.flatten()

        if semilog:
            ax.semilogy(X,Y,label=label)
        else:
            ax.plot(X,Y,label=label)
        
        if rangex is None:
            ax.set_xlim(self.min_xval,self.max_xval)
        else:
            ax.set_xlim(rangex[0],rangex[1])

        if rangey is None:
            if semilog:
                ax.set_ylim(10.0,2.0*np.nanmax(y))
            else:
                ax.set_ylim(0.0,1.2*np.nanmax(y))
        else:
            ax.set_ylim(rangey[0],rangey[1])
        
        ax.legend(loc='best',fontsize=10)

        if panel_label is not None:
            plt.text(panel_label_loc[0],panel_label_loc[1],panel_label,transform=ax.transAxes,fontsize=16)

        return fig,ax

if __name__ == '__main__':
    print()
   
   