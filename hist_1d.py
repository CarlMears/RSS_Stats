'''class for defining, accumulating, and displaying 1D histograms'''
import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt

class Hist1D():

    def __init__(self,num_xbins = 200,min_xval = -10.0,max_xval=10.0,
                 name='',units=''):
        self.num_xbins = num_xbins
        self.min_xval = min_xval
        self.max_xval = max_xval
        self.name = name
        self.units = units
        self.size_xbin = (self.max_xval - self.min_xval)/self.num_xbins

        xedges = self.min_xval + np.arange(0,self.num_xbins+1)*self.size_xbin
        xcenters = 0.5*(xedges[0:num_xbins]+xedges[1:self.num_xbins+1])
        
        self.data = xr.Dataset(
            data_vars = {'n' : (('xcenters'),np.zeros((self.num_xbins),dtype = np.float32))},
            coords = {'xedges'   : xedges,
                      'xcenters' : xcenters})  

    def get_cumulative(self):

        x_vals = self.data['xedges']
        z = np.insert(self.data['n'].values,0,0.0)
        cumsum = np.cumsum(z)
        return x_vals,cumsum

    def get_tot_num(self):
        tot_num = np.sum(self.data['n'].values)
        return tot_num

    def add(self,hist_to_add):
        '''add histogram data to existing histogram'''
        
        if isinstance(hist_to_add,Hist1D):
            h = hist_to_add.data['n'].values
        else:
            #for now, we assume hist_to_add is either and xarray, or a numpy array
            try:
                h = hist_to_add.dat
            except:
                h = hist_to_add

        if type(h) is not np.ndarray:
            print('array to add must be xarray or numpy array')
            raise(ValueError)

        self.data['n']=self.data['n'] + h

    def add_data(self,x):
        z = np.isfinite(x)
        hist_to_add,xedges = np.histogram(x[z],
                                        bins=self.num_xbins,
                                        range=[self.min_xval, self.max_xval])
        self.add(hist_to_add)

    def to_netcdf(self,filename = None):

        self.data.to_netcdf(path  = filename)

    def match_to(self,hist_to_match,attenuate_near_zero = False,atten_thres = 5.0):
        '''constructs an additive matching function that when added to the data that resulted in the histogram in self, with result in a
        new dataset with a histigram that matches the histogram  in hist2'''

        x,hist_in_cumsum =  self.get_cumulative()
        x,hist_to_match_cumsum = hist_to_match.get_cumulative()

        hist_in_cumsum = hist_in_cumsum/hist_in_cumsum[-1]
        hist_to_match_cumsum = hist_to_match_cumsum/hist_to_match_cumsum[-1]

        x_values = self.data.xedges.values
        z = np.interp(hist_in_cumsum,hist_to_match_cumsum,x_values)

        z1 = 0.5*(z[0:-1]+z[1:])
        x1 = 0.5*(x_values[0:-1]+x_values[1:]) 


        if  attenuate_near_zero:
            diff = z1-x1
            # reduce differences below threshold of about 5 m/s
            squelcher = np.ones_like(z1)
            num = np.sum(np.abs(x1 < atten_thres))
            a = -1.0/(atten_thres*atten_thres)
            b = 2.0/atten_thres

            squelcher[0:num] = a*np.square(x1[0:num]) + b*x1[0:num]  
            diff = diff*squelcher
            z1 = x1+diff

        z1[self.data['n'].values < 10.0] = np.nan
        
        '''fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.plot(x1,z1)
        ax.set_xlim((0,50))
        ax.set_ylim((0,50))'''


        '''additive_correction = z1-x1
        multiplicative_correction = z1/x1

        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.plot(x1,additive_correction)

        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        ax.plot(x1,multiplicative_correction)

        

        plt.show()

        print'''
        return x1,z1




  
    def plot(self, fig = None, ax = None, title=None, xtitle=None, ytitle=None,label=None,semilog=False):

        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import numpy as np
        import copy
        import sys

        sys.path.append('C:/job_CCMP/python/eval_ncep_monthly_bias/')
        sys.path.append('C:/job_CCMP/python/plotting_and_analysis/')
        sys.path.append('C:/job_CCMP/python/binned_stats/')
        sys.path.append('C:/job_CCMP/python/map_stats/')
        sys.path.append('C:/job_CCMP/python/ncep/')
        sys.path.append('C:/job_CCMP/python/eval_era5/')
        sys.path.append('C:/job_CCMP/python/era5/')
        sys.path.append('C:/job_CCMP/python/buoys/')
        sys.path.append('B:/job_CCMP/python/bytemaps/')
        sys.path.append('B:/job_CCMP/python/quikscat/')
        sys.path.append('B:/job_CCMP/python/ascat/')
        sys.path.append('B:/job_CCMP/python/ssmi/')
        sys.path.append('B:/job_CCMP/python/ssmis/')
        sys.path.append('B:/job_CCMP/python/amsre/')
        sys.path.append('B:/job_CCMP/python/amsr2/')
        sys.path.append('B:/job_CCMP/python/windsat/')
        sys.path.append('../')

        if label is None:
            label = self.name

        if xtitle is None:
            xtitle = self.units
        if ytitle is None:
            ytitle = 'Number of Obs'
        if title is None:
            title = ''

        created_fig = False
        if ax is None:
            fig,ax = plt.subplots(figsize=(10,8))
            ax.set_title(title)
            ax.set_xlabel(xtitle)
            ax.set_ylabel(ytitle)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(20)
            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(16)
            ax.set_xlim([self.min_xval,self.max_xval])
            created_fig = True


        
        left,right = self.data['xedges'][:-1],self.data['xedges'][1:]
        X = np.array([left,right]).T.flatten()
        

        y = self.data['n']
        Y = np.array([y,y]).T.flatten()
        if np.abs(self.min_xval) < 0.0001:
            if semilog:
                ax.semilogy(X,Y,label=label)
                ax.set_xlim(0.0,50.0)
            else:
                ax.plot(X,Y,label=label)
                ax.set_xlim(0.0,30.0)
        else:
            if semilog:
                ax.semilogy(X,Y,label=label)
            else:
                ax.plot(X,Y,label=label)
                
            ax.set_xlim(self.min_xval,self.max_xval)
        
        ax.legend(loc='best',fontsize=13)
        if created_fig:
            return fig,ax
        else:
            return ax

if __name__ == '__main__':

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    '''test histogram matching'''

    plot_path = 'C:/job_CCMP/python/plotting_and_analysis/histogram_matching/'

    #y1 = np.random.normal(loc = 10.0,scale = 2.0,size=10000000)
    #y2 = 1.1*np.copy(y1)

    ds = xr.open_dataset('C:/job_CCMP/compare_era5_vs_ASCAT/nc_files/NS_oscar/ERA5_vs_ASCAT_A_all_winds_NS_Oscar_2017_01.nc')
    y1 = ds['w_model'].values
    y1 = y1[1:]
    y2 = ds['w_ascat'].values
    y2 = y2[1:]

    hist1 = Hist1D(min_xval= 0.0,max_xval=50.0,num_xbins=250)
    hist2 = Hist1D(min_xval= 0.0,max_xval=50.0,num_xbins=250)

    hist1.add_data(y1)
    hist2.add_data(y2)

    hist_fig,ax = hist1.plot(label='ERA5')
    ax = hist2.plot(fig = hist_fig,ax=ax,label='ASCAT-A')
    png_file = plot_path+'ascat_era5_example_inputs.png'
    hist_fig.savefig(png_file)

    hist_fig_log,ax = hist1.plot(semilog=True,label='ERA5')
    ax = hist2.plot(fig = hist_fig_log,ax=ax,semilog=True,label='Histogram to match')
    png_file = plot_path+'ascat_era5_example_inputs_semilog.png'
    hist_fig_log.savefig(png_file)

    x,hist_in_cumsum =  hist1.get_cumulative()
    x,hist_to_match_cumsum = hist2.get_cumulative()

    hist_in_cumsum = hist_in_cumsum/hist_in_cumsum[-1]
    hist_to_match_cumsum = hist_to_match_cumsum/hist_to_match_cumsum[-1]

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(x,hist_in_cumsum,label = 'ERA5')
    ax.plot(x,hist_to_match_cumsum,label = 'ASCAT-A')
    ax.set_xlim(0.0,40.0)
    ax.legend()
    ax.set_title('Cumulative Histograms')
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Fraction of Observations less than X value')
       
    plot_path = 'C:/job_CCMP/python/plotting_and_analysis/histogram_matching/'
    png_file = plot_path+'ascat_era5_example_cumulative_hists.png'
    fig.savefig(png_file)

    #plt.show()

    x,z = hist1.match_to(hist2)
    corr_fig = plt.figure(figsize=(6.5,5))
    ax = corr_fig.add_subplot(111)
    ax.plot(x,z)
    ax.set_xlim(0.0,40.0)
    ax.set_ylim(0.0,40.0)
    ax.plot([0.0,40.0],[0.0,40.0],color='grey',linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('ERA5')
    ax.set_ylabel('Adjusted Value')
    ax.set_title('Adjustment Function F')
    png_file = plot_path+'ascat_era5_example_adj_function.png'
    corr_fig.savefig(png_file)

    alpha = z/x
    alpha_fig = plt.figure(figsize=(6.5,5))
    ax = alpha_fig.add_subplot(111)
    ax.plot(x,alpha)
    ax.set_xlim(0.0,40.0)
    ax.set_ylim(0.8,1.6)
    ax.plot([0.0,40.0],[1.0,1.0],color='grey',linewidth=0.5)
    ax.set_xlabel('ERA5')
    ax.set_ylabel('Adjustment Factor')
    ax.set_title('Adjustment Factor Alpha')
    png_file = plot_path+'ascat_era5_example_adj_factor.png'
    alpha_fig.savefig(png_file)
  
    interp = interp1d(x,z,
                      kind='linear',
                      fill_value='extrapolate')
    
    y3 = interp(y1)

    hist3 = Hist1D(min_xval= 0.0,max_xval=50.0,num_xbins=250)
    hist3.add_data(y3)

    hist_fig,ax = hist1.plot(label='ERA5')
    ax = hist2.plot(fig = hist_fig,ax=ax,label='ASCAT-A')
    ax = hist3.plot(fig = hist_fig,ax=ax,label='ERA5, Adjusted')
    png_file = plot_path+'ascat_era5_example_final_match.png'  
    hist_fig.savefig(png_file)
  
    hist_fig_semilog,ax = hist1.plot(semilog=True,label='ERA5')
    ax = hist2.plot(fig = hist_fig_semilog,ax=ax,semilog=True,label='Hist to Match')
    ax = hist3.plot(fig = hist_fig_semilog,ax=ax,semilog=True,label='ERA5, Adjusted')
    png_file = plot_path+'ascat_era5_example_final_semilog.png'  
    hist_fig_semilog.savefig(png_file)

    plt.show()

    print()