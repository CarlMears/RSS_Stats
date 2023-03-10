# RSS_Stats

This module contains several classes that accumulate statistics for use in analysis of satellite data

## binned_stats.py 
Contains BinnedStat, a class for accumulating data to calculate binned means and stddevs for "cross" talk plots

## binned_stats_2D 
Contains BinnedStat2D, a class for accumulating data to calculate binned means and stddevs for "cross" talk plots

## hist_1d.py 
Contains Hist1D, a class for accumulating data in the form of 1 dimensional histograms

## hist_2d.py
Contains Hist2D, a class for accumulating data in the for of 2 dimensional histograms

## map_stats.py
Contains MapStat, a class for accumalating data in rectangular geographical maps

## polar_map_stats.py 
Contains PolarMapStat, a class for accumulating data in polar-gridded maps

There are also several legacy routines in binned_means.py that calculate, combine, and plot binned means.

*plot_2d_hist.py has been moved to the plotting project*

---
# Class Reference
---
## binned_stats.py 
Contains BinnedStat, a class for accumulating data to calculate binned means and stddevs for "cross" talk plots.
The binned number data points, the binned total of the data point values, and the binned square of the 
datapoint values are accumulated, making it easy to combine objects and add additional data.

---
### Initialize
bs = BinnedStats(num_bins = 40, x_rng=[0.0,1.0])

---
### Attributes
- self.num_bins 
- self.x_rng
- self.binned_x_tot  
- self.binned_y_tot           
- self.binned_y_totsqr       
- self.binned_num            

- self.overall_tot          
- self.overall_totsqr        
- self.overall_num         

- self.xbin_size 
- self.xbin_centers 
- self.xbin_edges 

___
### Methods
---
#### BinnedStats.add_data(self,x,y,mask=None,verbose=False)
Adds data to BinnedStat object
- x: arraylike, x values to be added
- y: arraylike, same size as x, y values to be added
- mask: arraylike, same size as x and y, set to > 0.5 to exclude data
- verbose: boolean, if True, verbose reporting
---
#### BinnedStats.combine(bs2)
Combines two BinnedStat object if xedges are identical
- bs2: BinnedStat object to be combined into the original Object, which is modified in place

- Return Value: BinnedStat object containing the combined results

- Usage:
~~~python 
bs.combine(bs2) # combines in place in bs, or 
bs3 = BinnedStat.combine(bs1,bs2) #makes a new instance
~~~
---
#### BinnedStats.calc_stats(self,return_as_xr = False)
Returns dictionary of binned stats calculated from the current state of the object.  If return_as_xr is true, returns an xarray Dataset object.
~~~python
binned_stats.keys() = [binned_x_means,
                       binned_y_means,
                       binned_y_stddev,
                       binned_num,
                       overall_mean,
                       overall_stddev,
                       overall_rms,
                       overall_num,
                       xbin_centers,
                       xbin_edges,
                       xrng]
~~~


