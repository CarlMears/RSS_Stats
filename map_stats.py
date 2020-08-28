import numpy as np
import xarray as xr

class MapStat():

    ''' Class for accumulating and reporting average map statistics

    Args:
        num_lats:   number of latitide grid cells
        num_lons:   number of longitude grid cells
        min_lat:    minimum latitude
        max_lat:    maximum latitude
        min_lon:    minimum longitude
        max_lon:    maximum longitude

    '''

    def __init__(self, num_lats = 720,num_lons = 1440,min_lat = -89.875,max_lat = 89.875,min_lon = -179.875,max_lon=179.875):

        self.num_lats = num_lats
        self.num_lons = num_lons
        self.min_lat  = min_lat
        self.max_lat  = max_lat
        self.min_lon  = min_lon
        self.max_lon  = max_lon
        self.dlat = (max_lat - min_lat) / (num_lats - 1)
        self.dlon = (max_lon - min_lon) / (num_lons - 1)
        
        self.lats = self.min_lat + np.arange(0,self.num_lats)*self.dlat 
        self.lons = self.min_lon + np.arange(0,self.num_lons)*self.dlon

        self.num =    np.zeros((num_lats,num_lons),dtype = np.int64)
        self.tot =    np.zeros((num_lats,num_lons),dtype = np.float64)
        self.totsqr = np.zeros((num_lats,num_lons),dtype = np.float64)

    def from_netcdf_triple(nc_file = None):

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
        
        return u_map_stats,v_map_stats,w_map_stats

        

    def combine(self,self2):

        #make sure maps are compatible
        try:
            assert(isinstance(self2,MapStat))
            assert(self.num_lats == self2.num_lats)
            assert(self.num_lons == self2.num_lons)
            assert(self.max_lat == self2.max_lat) 
            assert(self.min_lat == self2.min_lat)
            assert(self.max_lon == self2.max_lon) 
            assert(self.min_lon == self2.min_lon)
        except:
            raise ValueError('Map objects not compatible, can not combine')

        self.num = self.num + self2.num
        self.tot = self.tot + self2.tot
        self.totsqr = self.totsqr + self2.totsqr
            

    def add_map(self,map):

        sz = map.shape
        assert(len(sz)==2)
        assert(sz[0] == self.num_lats)
        assert(sz[1] == self.num_lons)

        # if we get to here, then the arrays are the same size

        # for now, assume the  lats and  lons add up.
        # probably map should be an xarray and we could check to make sure
        # coordinates match

        ok_map = np.zeros((sz))
        ok = np.isfinite(map)

        self.num[ok] = self.num[ok] + 1
        self.tot[ok] = self.tot[ok] + map[ok]
        self.totsqr[ok] = self.totsqr[ok] + np.square(map[ok])

    def num_obs(self):
        return self.num

    def mean(self):
        mean_map = self.tot/self.num
        return mean_map

    def variance(self):
        mean_map = self.tot/self.num
        variance_map = self.totsqr/self.num - np.square(mean_map)
        variance_map[self.num < 3] = np.nan 
        return variance_map

    def stddev(self):
        stddev_map = np.sqrt(self.variance())
        return stddev_map

    def zonal_sum(self):
        zonal_num = np.sum(self.num,axis=1)
        zonal_tot = np.sum(self.tot,axis=1)
        zonal_tot_sqr = np.sum(self.totsqr,axis=1)
        z_sum = {"num" : zonal_num,
                 "tot" : zonal_tot,
                 "totsqr" : zonal_tot_sqr
                }
        return z_sum

    def zonal_num_good_pix(self):
        temp = np.zeros_like(self.num)
        temp[self.num > 0] = 1
        num_good_pix = np.sum(temp,axis=1)
        return num_good_pix

    def zonal_mean(self,min_good_pix = None):
        z_sum = self.zonal_sum()
        z_mean = z_sum['tot']/z_sum['num']
        z_mean[z_sum['num'] == 0] = np.nan

        if min_good_pix is not None:
            z_num = self.zonal_num_good_pix
            z_mean[z_num() < min_good_pix] = np.nan

        return z_mean

    def zonal_variance(self):
        z_sum = self.zonal_sum()
        z_mean = z_sum['tot']/z_sum['num']
        z_variance = z_sum['totsqr']/z_sum['num'] - np.square(z_mean)
        z_variance[z_sum['num'] < 3.0] = np.nan
        
        return z_variance

    def zonal_stddev(self,min_good_pix = None):
        z_variance = self.zonal_variance()
        z_stddev = np.sqrt(z_variance)

        if min_good_pix is not None:
            z_num = self.zonal_num_good_pix
            z_stddev[z_num() < min_good_pix] = np.nan
        return z_stddev

    def as_xr(self):
        ds = xr.Dataset(
            data_vars={'num'    : (('lats', 'lons'), self.num),
                       'tot'    : (('lats', 'lons'), self.tot),
                       'totsqr'    : (('lats', 'lons'), self.totsqr)
                       },
            coords={ 'lats': self.lats,
                     'lons': self.lons})
        return ds
        



















