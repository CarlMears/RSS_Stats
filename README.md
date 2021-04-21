RSS_Stats

This module contains several classes that accumulate statistics for use in analysis of satellite data

binned_stats.py contains BinnedStat
    A class for accumulating data to calculate binned means and stddevs for "cross" talk plots

hist_1d.py contains Hist1D
    A class for accumulating data in the form of 1 dimensional histograms

hist_2d.py contains Hist2D
    A class for accumulating data in the for of 2 dimensional histograms

map_stats.py contains MapStat
    A class for accumalating data in rectangular geographical maps

polar_map_stats.py contains PolarMapStat
    A class for accumulating data in polar-gridded maps

plot_2d_hist.py contains helper functions for Hist2D - should probably be included in hist_2d.py
