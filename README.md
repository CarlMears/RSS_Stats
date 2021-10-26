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

# Class Reference

## binned_stats.py 
Contains BinnedStat, a class for accumulating data to calculate binned means and stddevs for "cross" talk plots
### Initialize
bs = BinnedStats(num_bins = 40, x_rng=[0.0,1.0])
### Methods
#### BinnedStats.add_data(self,x,y,mask=None,verbose=False):
- x: arraylike, x values to be added
- y: arraylike, same size as x, y values to be added
- mask: arraylike, same size as x and y, set to > 0.5 to exclude data
- verbose: boolean, if True, verbose reporting



