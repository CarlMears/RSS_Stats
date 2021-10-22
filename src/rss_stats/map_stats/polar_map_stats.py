from sea_ice_plotting import plot_polar_stereographic
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d
from polar_grids import polarstereo_inv

class PolarMapStat:
    ''' 
    #Overview
    Class for accumulating and reporting map statistics on a polar stereographic grid.  
    Maps are assumed to be on a 25 km polarstereographic grid, as defined by NSIDC for each pole.  
    The number of grid cells is determined by the pole being represented.
    
    Each map contains maps of the number of observations, the total of the observations, and the total square of the observations.  
    This is to make it easy to combined maps, e.g. combine daily maps into a monthly average.  

    There is also an option to have the map refer to a discontinuous "ice type" variable.  
    
    Unlike the rectangular MapStat class, the data here is represented as an xarray.  

    #To Do
      - Add write methods
      - Add type_map to compatiblity method
    
    #Arguments:
      - pole:  pole to represent - 'north' or 'south'  
      - type_map:  set to True is the map is of ice_type  

    '''

    def __init__(self,pole = 'north',type_map = False):

        if pole == 'north':
            num_x = 304
            num_y = 448
            x_range = [-154.0, 149.0]
            y_range = [-214.0, 233.0]
            std_lat = 70.0
            std_lon = -45.0
        elif pole == 'south':
            num_x = 316
            num_y = 332
            x_range = [-158.0, 157.0]
            y_range =[-158.0, 173.0]
            std_lat = -70.0
            std_lon = 180.0
        else:
            raise(ValueError,f"pole: {pole} not defined")
        self.pole = pole
        '''Pole represented - "north" or "south"'''
        self.type_map = type_map
        '''Set to True is a ice type map'''
        self.num_x = num_x
        #Number cells in the X direction'''
        self.num_y = num_y
        #Number of cells in the Y direction'''
        self.x_range = x_range
        #extent of the X coordinate, in cell index'''
        self.y_range = y_range
        #Extent of the Y coordinate, in cell index'''
        

        if type_map:
            self.num_stats = 4
            #Number statistics represented'''
            self.stat_type = ['ocean','new','fy','my']
            #Names of the stats represented'''
        else:
            self.num_stats = 3
            self.stat_type = [0,1,2]

        iy_list = np.arange(self.y_range[0], self.y_range[1]+1, dtype=np.float32)
        ix_list = np.arange(self.x_range[0], self.x_range[1]+1, dtype=np.float32)
        xdist_list = (ix_list+0.5)*25000.0
        ydist_list = (iy_list+0.5)*25000.0
        self.data =  xr.Dataset(
            data_vars={},
            coords={'ygrid': ydist_list,
                    'xgrid': xdist_list,
                    'stat_type': self.stat_type})

        if pole == 'north':
            self.data.xgrid.attrs = {'valid_range':[-3850000.0,3750000.0],
                                 'units' : 'meters',
                                 'long_name' : 'projection_grid_x_centers',
                                 'standard_name' : 'projection_x_coordinate',
                                 'axis' : 'X'}
        
            self.data.ygrid.attrs = {'valid_range':[-5350000.0,5850000.0],
                                 'units' : 'meters',
                                 'long_name' : 'projection_grid_y_centers',
                                 'standard_name' : 'projection_y_coordinate',
                                 'axis' : 'Y'}

        else:
            self.data.xgrid.attrs = {'valid_range':[-3950000.0,3950000.0],
                                 'units' : 'meters',
                                 'long_name' : 'projection_grid_x_centers',
                                 'standard_name' : 'projection_x_coordinate',
                                 'axis' : 'X'}
        
            self.data.ygrid.attrs = {'valid_range':[-3950000.0,4350000.0],
                                 'units' : 'meters',
                                 'long_name' : 'projection_grid_y_centers',
                                 'standard_name' : 'projection_y_coordinate',
                                 'axis' : 'Y'}

        self.data['Latitude'] = xr.DataArray(np.zeros((self.num_y,self.num_x),dtype=np.float32), 
                                                coords=[self.data.ygrid,self.data.xgrid], 
                                                dims=['ygrid', 'xgrid'])
        self.data['Longitude'] = xr.DataArray(np.zeros((self.num_y,self.num_x),dtype=np.float32), 
                                                coords=[self.data.ygrid,self.data.xgrid], 
                                                dims=['ygrid', 'xgrid'])

        self.data['X'] = xr.DataArray(np.zeros((self.num_y,self.num_x),dtype=np.float32), 
                                                coords=[self.data.ygrid,self.data.xgrid], 
                                                dims=['ygrid', 'xgrid'])
        
        self.data['Y'] = xr.DataArray(np.zeros((self.num_y,self.num_x),dtype=np.float32), 
                                                coords=[self.data.ygrid,self.data.xgrid], 
                                                dims=['ygrid', 'xgrid'])

        lat_temp = np.zeros((self.num_y, self.num_x),dtype=np.float32)
        lon_temp = np.zeros((self.num_y, self.num_x),dtype=np.float32)
        x_temp = np.zeros((self.num_y, self.num_x),dtype=np.float32)
        y_temp = np.zeros((self.num_y, self.num_x),dtype=np.float32)

        for j in np.arange(0, ydist_list.shape[0]):
            x_temp[j,:] = xdist_list
        for i in np.arange(0, xdist_list.shape[0]):
            y_temp[:,i] = ydist_list

        if pole == 'north':
            lon_temp, lat_temp = polarstereo_inv(x_temp/1000.0,y_temp/1000.0, std_parallel = std_lat, lon_y = std_lon)
        else:
            lon_temp, lat_temp = polarstereo_inv(-x_temp/1000.0, y_temp/1000.0, std_parallel = -std_lat, lon_y = std_lon)
            lat_temp = -lat_temp

        self.data['Longitude'].values = lon_temp
        self.data['Latitude'].values  = lat_temp
        self.data['X'].values = x_temp
        self.data['Y'].values = y_temp

    def add_map(self,map_name = 'none',dtype = np.float32):
        '''Add data in a 2D polar map to the object, assuming each grid cell has 1 observation'''
        data = np.zeros((self.num_y,self.num_x,self.num_stats),dtype=dtype)
        self.data[map_name] = xr.DataArray(data, coords=[self.data.ygrid,self.data.xgrid,self.data.stat_type], dims=['ygrid', 'xgrid','stat'])
    
    def add_data_to_map_lat_lon(self,map_name='none',values = None,lat = None, lon = None, percent_land = None,land_thres = 1.0,pole = 'north'):
        '''Add data to map from a np arrays of values, latitudes and longitudes  
        If percent_land is presents, only add data where percent land is < land_threshold  '''
        from polar_projections.polar_grids import polarstereo_fwd,polarstereo_fwd_SP
        if pole == 'north':
            x, y = polarstereo_fwd(lat, lon)
        else:
            x, y = polarstereo_fwd_SP(lat, lon)

        ix = np.floor(x * 0.04).astype(np.int32)
        iy = np.floor(y * 0.04).astype(np.int32)

        map = np.zeros((self.num_y, self.num_x, self.num_stats))

        if percent_land is None:
            #ignore land  mask because not present
            ok = np.all([(np.isfinite(values)),
                         (ix >= self.x_range[0]),
                         (ix <= self.x_range[1]),
                         (iy >= self.y_range[0]),
                         (iy <= self.y_range[1])], axis=0)
        else:
            ok  = np.all([(np.isfinite(values)),
                          (ix >= self.x_range[0]),
                          (ix <= self.x_range[1]),
                          (iy >= self.y_range[0]),
                          (iy <= self.y_range[1]),
                          (percent_land <= land_thres)],axis=0)

        rng = np.array([[self.y_range[0] - 0.5,self.y_range[1] + 0.5],
               [self.x_range[0] - 0.5,self.x_range[1] + 0.5]])

        
        map[:, :, 0], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], values[ok],
                                                                      statistic='count',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)
        map[:, :, 1], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], values[ok],
                                                                      statistic='sum',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)
        map[:, :, 2], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], np.square(values[ok]),
                                                                      statistic='sum',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)

        self.data[map_name] = self.data[map_name] + map

    def add_data_to_map(self,map_name='none',values = None,ix = None, iy = None, percent_land = None,land_thres = 1.0):
        '''Add data to map from a np arrays of values, x and y locations  
        ix and iy should already be converted to integer cell positions  
        If percent_land is presents, only add data where percent land is < land_threshold  '''
        map = np.zeros((self.num_y, self.num_x, self.num_stats))
        if percent_land is None:
            #ignore land  mask because not present
            ok = np.all([(np.isfinite(values)),
                         (ix >= self.x_range[0]),
                         (ix <= self.x_range[1]),
                         (iy >= self.y_range[0]),
                         (iy <= self.y_range[1])], axis=0)
        else:
            ok  = np.all([(np.isfinite(values)),
                          (ix >= self.x_range[0]),
                          (ix <= self.x_range[1]),
                          (iy >= self.y_range[0]),
                          (iy <= self.y_range[1]),
                          (percent_land <= land_thres)],axis=0)

        rng = np.array([[self.y_range[0] - 0.5,self.y_range[1] + 0.5],
               [self.x_range[0] - 0.5,self.x_range[1] + 0.5]])

        
        map[:, :, 0], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], values[ok],
                                                                      statistic='count',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)
        map[:, :, 1], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], values[ok],
                                                                      statistic='sum',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)
        map[:, :, 2], xedges, yedges, binnumber = binned_statistic_2d(iy[ok], ix[ok], np.square(values[ok]),
                                                                      statistic='sum',
                                                                      bins=[self.num_y, self.num_x],
                                                                      range=rng)

        self.data[map_name] = self.data[map_name] + map

    def add_type_data_to_map(self,map_name='none',values = None,ix = None, iy = None, percent_land = None,land_thres = 1.0):
        '''Add data to map from a np arrays of ice types, latitudes and longitudes  
        If percent_land is presents, only add data where percent land is < land_threshold  '''

        if not self.type_map:
            raise(ValueError,'Map object is not an ice type object')

        map = np.zeros((self.num_y, self.num_x, self.num_stats))

    def compatible(self,self2):
        '''Check to see if maps are compatible  
        Often called before combining'''
        if isinstance(self2,PolarMapStat):
            attr_to_check = ['pole', 'num_x', 'num_y','x_range','y_range','std_lat','std_lon']
            maps_compatible = True
            for attr in attr_to_check:
                try:
                    if getattr(self,attr) != getattr(self2,attr):
                        maps_compatible = False
                except KeyError:
                    maps_compatible = False
        else:
            maps_compatible=False

        return maps_compatible



    def combine(self,self2,verbose=False):
        '''Combine data from second PolarMapStat object into self'''

        keys = dict(self.data).keys()
        keys2 = dict(self.data).keys()
        common_keys = set(keys) & set(keys2)
        for unneeded_key in ['Latitude','Longitude','X','Y']:
            if unneeded_key in common_keys:
                common_keys.discard(unneeded_key)
        chan_list = list(common_keys)
        chan_list.sort()
        for chan in chan_list:
            if verbose:
                print(f'Combining Map for channel {chan}')
            self.data[chan] += self2.data[chan]

    def num_obs(self):
        '''Return map of number of observations'''
        if self.type_map:
            return np.sum(self.data[map_name],axis=2)
        else:
            return self.data[map_name][0,:,:]


    def mean(self,map_name):
        '''Return mean map'''
        if self.type_map:
            raise ValueError('Error - mean makes no sense for type maps')
        else:
            mean_map = np.zeros_like(self.data[map_name][:,:,0])
            np.divide(self.data[map_name][:,:,1], 
                        self.data[map_name][:,:,0], 
                        out=mean_map, 
                        where=self.data[map_name][:,:,0] > 0)
            mean_map[self.data[map_name][:,:,0] == 0.0] = np.nan
        return mean_map

    def plot(self,map_name=None,zrange=(0.0,1.0),stat = 'mean',title=None,units=None,coast_color = 'w',cmap=None):
        '''Plot polar map using matplotlib/cartopy'''
        if title is None:
            title = map_name
        if stat=='mean':
            stat_to_plot = self.mean(map_name)
            fig = plot_polar_stereographic(stat_to_plot,zrange=zrange,title=title,units=units,coast_color=coast_color,pole=self.pole,cmap=cmap)
            
        else:
            raise ValueError('Unsupported stat, supported stats: mean')
        return fig

    #def variance(self):


    #def stddev(self):
  

