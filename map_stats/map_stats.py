import numpy as np
import xarray as xr


class MapStat():

    '''#Overview
    Class for accumulating and reporting map statistics.  
    Maps are assumed to be on a regular lat/lon grids, with ilat=0 corresponding to the southernmost point.  
    Each map contains maps of the number of observations, the total of the observations, and the total square of the observations.
    This is to make it easy to combined maps, e.g. combine daily maps into a monthly average.  

    #Args:  
    - num_lats:   number of latitide grid cells  
    - num_lons:   number of longitude grid cells  
    - min_lat:    minimum latitude  
    - max_lat:    maximum latitude  
    - min_lon:    minimum longitude  
    - max_lon:    maximum longitude  

    #Internal Representation:
      - self.num numpy array, number of observations in grid cell  
      - self.tot numpy array, total of observations in grid cell
      - self.tot_sqr numpy array, total square of observations in grid cell
    '''
    

    def __init__(self, num_lats = 720,num_lons = 1440,min_lat = -89.875,max_lat = 89.875,min_lon = -179.875,max_lon=179.875):
        '''Initialize MapStat Object'''
        self.num_lats = num_lats
        #'''Number of latitude cells in map'''
        self.num_lons = num_lons
        #'''Number of longitude cells in map'''
        self.min_lat  = min_lat
        #'''Minimum latitude in map'''
        self.max_lat  = max_lat
        #'''Maximum latitude in map'''
        self.min_lon  = min_lon
        #'''Minimum longitude in map'''
        self.max_lon  = max_lon
        #'''Maximum longitude in map'''
        self.dlat = (max_lat - min_lat) / (num_lats - 1)
        #'''Latitude spacing in map'''
        self.dlon = (max_lon - min_lon) / (num_lons - 1)
        #'''Longitude spacing in map'''
        
        self.lats = self.min_lat + np.arange(0,self.num_lats)*self.dlat 
        self.lons = self.min_lon + np.arange(0,self.num_lons)*self.dlon

        self.num =    np.zeros((num_lats,num_lons),dtype = np.int64)
        '''number of observations in grid cell'''
        self.tot =    np.zeros((num_lats,num_lons),dtype = np.float64)
        '''total of observations in grid cell'''
        self.totsqr = np.zeros((num_lats,num_lons),dtype = np.float64)
        '''total of squared observations in grid cell'''

    def from_np_triple(*,n,tot,totsqr,num_lats = 720,num_lons = 1440,min_lat = -89.875,max_lat = 89.875,min_lon = -179.875,max_lon=179.875):
        '''Initilzes MapStat from 3 numpy 2D arrays (n,tot,totsqr)'''

        stats =  MapStat(num_lats = num_lats,num_lons = num_lons,min_lat = min_lat,max_lat = max_lat,min_lon = min_lon,max_lon=max_lon)
        stats.num += n
        stats.tot += tot
        stats.totsqr += totsqr
        
        return stats

    def from_netcdf_triple(nc_file = None,speed_only = False):
        '''Opens a netcdf file and reads U, V, and W maps, each with total and tot_sqr data.
        U data must be named u_tot and u_tot_sqr in the file
        V data must be named v_tot and v_tot_sqr
        W data must be named w_tot and w_tot_sqr

        If any data is missing, the corresponding MapStat object is returned empty

        Unless speed_only is set True -- in this case, only the W variables are loaded
        '''
        try:
            ds = xr.open_dataset(nc_file)
        except:
            return [np.nan,np.nan,np.nan]

        try:
            lats = ds['Latitude'].values
            lons = ds['Longitude'].values
            num = ds['num'].values
        except:
            raise ValueError('Critical map data missing from ' + nc_file)

        min_lat = float(lats[0])
        max_lat = float(lats[-1])
        min_lon = float(lons[0])
        max_lon = float(lons[-1])

        num_lats = lats.shape[0]
        num_lons = lons.shape[0]
        dlat = float(lats[1]-lats[0])
        dlon = float(lons[1]-lons[0])

        if not speed_only:
            try:
                u_tot = ds['u_tot'].values
                u_tot_sqr = ds['u_tot_sqr'].values
                u_map_stats = MapStat(num_lats = num_lats,num_lons = num_lons,min_lat = min_lat,max_lat = max_lat,min_lon = min_lon,max_lon=max_lon)
                u_map_stats.num  = num
                u_map_stats.tot  = u_tot
                u_map_stats.totsqr = u_tot_sqr
            except:
                print('U data missing from '+nc_file)
                u_map_stats = MapStat(num_lats = num_lats,num_lons = num_lons,min_lat = min_lat,max_lat = max_lat,min_lon = min_lon,max_lon=max_lon)

            try:
                v_tot = ds['v_tot'].values
                v_tot_sqr = ds['v_tot_sqr'].values
                v_map_stats = MapStat(num_lats = num_lats,num_lons = num_lons,min_lat = min_lat,max_lat = max_lat,min_lon = min_lon,max_lon=max_lon)
                v_map_stats.num  = num
                v_map_stats.tot  = v_tot
                v_map_stats.totsqr = v_tot_sqr
            except:
                print('V data missing from '+nc_file)
                v_map_stats =MapStat(num_lats = num_lats,num_lons = num_lons,min_lat = min_lat,max_lat = max_lat,min_lon = min_lon,max_lon=max_lon)
        
        
        try:
            w_tot = ds['w_tot'].values
            w_tot_sqr = ds['w_tot_sqr'].values
            w_map_stats = MapStat(num_lats = num_lats,num_lons = num_lons,min_lat = min_lat,max_lat = max_lat,min_lon = min_lon,max_lon=max_lon)
            w_map_stats.num  = num
            w_map_stats.tot  = w_tot
            w_map_stats.totsqr = w_tot_sqr
        except:
            print('W data missing from '+nc_file)
            w_map_stats =MapStat(num_lats = num_lats,num_lons = num_lons,min_lat = min_lat,max_lat = max_lat,min_lon = min_lon,max_lon=max_lon)
        if not speed_only:
            return u_map_stats,v_map_stats,w_map_stats
        else:
            return w_map_stats



    def compatible(self,self2):
        '''Checks to make sure that 2 MapStat objects are compatible, usually called before combining
        two maps are combined'''
        maps_compatible = True
        if isinstance(self2,MapStat):
            attr_to_check = ['num_lats','num_lons','max_lat','min_lat','max_lon','min_lon']
            for attr in attr_to_check:
                try:
                    if getattr(self,attr) != getattr(self2,attr):
                        maps_compatible = False
                except KeyError:
                    maps_compatible = False
        else:
            maps_compatible=False

        return maps_compatible

    def combine(self,self2):
        '''Combine the data in two MapStat objects'''
        #make sure maps are compatible
        maps_compatible = self.compatible(self2)
        try:
            maps_compatible = self.compatible(self2)
        except:
            raise ValueError('Map objects not compatible, can not combine')

        if not maps_compatible:
            raise ValueError('Map objects not compatible, can not combine')

        self.num = self.num + self2.num
        self.tot = self.tot + self2.tot
        self.totsqr = self.totsqr + self2.totsqr
            

    def add_data(self,lats=None,lons=None,values=None):

        '''Adds data from numpy arrays of lats,lons and values to the map'''

        if np.any([(lons.shape != values.shape),(lons.shape != lats.shape)]):
            raise ValueError('array sizes do not match')
        
        
        ilats = np.floor((lats-(self.min_lat-self.dlat/2.0))/self.dlat).astype(np.int32)
        ilons = np.floor((lons-(self.min_lon-self.dlon/2.0))/self.dlon).astype(np.int32)

        ok  = np.all([(ilats>=0),(ilats < self.num_lats),
                      (ilons>=0),(ilons < self.num_lons),
                      (np.isfinite(values))
                    ],axis=(0))

        values_ok = values[ok].tolist()
        values_sqr_ok = np.square(values[ok]).tolist()
        ilats_ok = ilats[ok].tolist()
        ilons_ok = ilons[ok].tolist()
        #num_vals = len(values_ok)
        for ilat,ilon,val,val_sqr in zip(ilats_ok,ilons_ok,values_ok,values_sqr_ok):
            self.num[ilat,ilon] += 1.0
            self.tot[ilat,ilon] += val
            self.totsqr[ilat,ilon] += val_sqr

            
    def add_map(self,map):

        '''Add data from a lat/lon array of values -- assume that num is 1 for the map to be added'''

        sz = map.shape
        map_compatible = True
        if len(sz)!=2:
            map_compatible = False
        if sz[0] != self.num_lats:
            map_compatible = False
        if sz[1] != self.num_lons:
            map_compatible = False
        
        if not map_compatible:
            raise ValueError('map size not compatible, can not be added')

        # if we get to here, then the arrays are the same size

        # for now, assume the  lats and  lons add up.
        # probably map should be an xarray and we could check to make sure
        # coordinates match

        ok_map = np.zeros((sz))
        ok = np.isfinite(map)
        if np.nansum(ok) > 0:
            try:
                self.num[ok] += 1
                self.tot[ok] += map[ok]
                self.totsqr[ok] += np.square(map[ok])
            except:
                print()

    def num_obs(self):
        '''Return the num map as a numpy array'''
        return self.num

    def mean(self):
        '''Return a map of mean values'''

        mean_map = np.full_like(self.tot, np.nan)
        np.divide(self.tot, self.num, out=mean_map, where=self.num > 0)
        return mean_map

    def variance(self):
        '''Return a map of variances'''
        #mean_map = self.tot/self.num
        #variance_map = self.totsqr/self.num - np.square(mean_map)
        #variance_map[self.num < 3] = np.nan 
        mean_map = np.full_like(self.tot, np.nan)
        variance_map = np.full_like(self.tot, np.nan)
        np.divide(self.tot, self.num, out=mean_map, where=self.num > 2)
        np.divide(self.totsqr, self.num, out=variance_map, where=self.num > 2)
        variance_map -= np.square(mean_map) 
        return variance_map

    def stddev(self):
        '''Return a map of standard deviations'''
        return np.sqrt(self.variance())

    def zonal_sum(self):
        '''Return a zonal sum'''  
        zonal_num = np.sum(self.num,axis=1)
        zonal_tot = np.sum(self.tot,axis=1)
        zonal_tot_sqr = np.sum(self.totsqr,axis=1)
        z_sum = {"num" : zonal_num,
                 "tot" : zonal_tot,
                 "totsqr" : zonal_tot_sqr
                }
        return z_sum

    def zonal_num_good_pix(self):
        '''Return zonal sum of good pix'''
        temp = np.zeros_like(self.num)
        temp[self.num > 0] = 1
        num_good_pix = np.sum(temp,axis=1)
        return num_good_pix

    def zonal_mean(self,min_good_pix = None):
        '''Return zonal mean'''
        z_sum = self.zonal_sum()
        z_mean = z_sum['tot']/z_sum['num']
        z_mean[z_sum['num'] == 0] = np.nan

        if min_good_pix is not None:
            z_num = self.zonal_num_good_pix
            z_mean[z_num() < min_good_pix] = np.nan

        return z_mean

    def zonal_variance(self,min_good_pix = None):
        '''Return zonal variance'''
        z_sum = self.zonal_sum()
        z_mean = z_sum['tot']/z_sum['num']
        z_variance = z_sum['totsqr']/z_sum['num'] - np.square(z_mean)
        z_variance[z_sum['num'] < 3.0] = np.nan
        if min_good_pix is not None:
            z_num = self.zonal_num_good_pix
            z_variance[z_num() < min_good_pix] = np.nan
        
        return z_variance

    def zonal_stddev(self,min_good_pix = None):
        '''Return zonal standard deviation'''
        z_variance = self.zonal_variance()
        z_stddev = np.sqrt(z_variance)

        if min_good_pix is not None:
            z_num = self.zonal_num_good_pix
            z_stddev[z_num() < min_good_pix] = np.nan
        return z_stddev

    def as_xr(self):
        '''Returns MapStat objects as an xarray'''
        ds = xr.Dataset(
            data_vars={'num'    : (('lats', 'lons'), self.num),
                       'tot'    : (('lats', 'lons'), self.tot),
                       'totsqr'    : (('lats', 'lons'), self.totsqr)
                       },
            coords={ 'lats': self.lats,
                     'lons': self.lons})
        return ds

