'''#Overview
Helper functions to compuse simple stats from numpy arrays representing maps.  
These functions are deprecated
#To do
  - Remove these from current code and replace with MapStats instances where needed  
'''
def zonal_mean(map,num_thres = 50):
    import numpy as np
    sz = map.shape
    #print(sz)
    num_lats = sz[0]
    zonal_mean  = np.zeros((num_lats))*np.nan
    for ilat in range(0,num_lats):
        row = map[ilat,:]
        num_good = np.sum(np.isfinite(row))
        if num_good > num_thres:
            zonal_mean[ilat] = np.nanmean(row)
        #print(ilat,num_good,zonal_mean[ilat])
    
    return zonal_mean

def global_mean(map):
    import numpy as np
    sz = map.shape
    num_lats = sz[0]
    num_lons = sz[1]
    if (num_lats % 2) == 0:
        delta_lat = 180.0/num_lats
        lats = -90.0 + delta_lat/2.0 + np.arange(0,num_lats)*delta_lat
    else:
        delta_lat = 180.0/(num_lats -1)
        lats = -90.0 + np.arange(0,num_lats)*delta_lat
        
    lat_wt_map = np.zeros((num_lats,num_lons))
    for ilat in np.arange(0,num_lats):
        lat_wt_map[ilat,:] = np.ones((num_lons))*np.cos(2.0*np.pi*lats[ilat]/360.0)
        
    #global_map(lat_wt_map, vmin=-1.0, vmax=1.0, cmap=cmap_diff, plt_colorbar=True,
    #                                 title='lat wts')
    
    wted_map = map*lat_wt_map
    
    mn  = np.nanmean(wted_map)
    stddev = np.nanstd(wted_map)
    return mn,stddev