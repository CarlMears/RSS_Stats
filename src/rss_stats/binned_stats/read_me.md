
    Class to accumulate binned statistics, and some helper functions

    The main class accumulates 1-dimensional binned statistics.   
    Each bin is described by:   
        the bin edges
        the bin center
        the number of item accumulated,
        the total x values of the items accumulated
        the total y values of the items accumulated
        the total squared y values of the items accumlated
    

        overall statistics are also accumulated.

    How to import and use example:

    from rss_stats.binned_stats import BinnedStats

    bs = BinnedStats()

    Class Attributes
    ----------
    num_bins : int
        number of bins
    x_rng : [float,float]
        lower and upper limit to be put into bins
    binned_x_tot: numpy float64 array[num_bins]
        total of the x values acculated into each bin
    binned_y_tot: numpy float64 array[num_bins]
        total of the y values acculated into each bin
    binned_y_totsqr: numpy float64 array[num_bins]
        total of the square of the y values acculated into each bin
    binned_num: numpy float64 array[num_bins]
        total number of data points in each bin
    overall_tot : float64
        total of all y values
    overall_totsqr : float64
        total of the sqaure of all y values
    overall_num : float 64
        total number of datapoints accumulated
    xbin_size : float64
        width of each bin, the same for each bin
    xbin_centers : np float64 array[num_bins]
        center of each bin (different from average x values!)
    xbin_edges : np float64 array[num_bins+1]
        edges of each bin

    Methods
    --------
    __init__(num_bin=40,x_rng=[0.0,1.0])
        initializes a class instance
    add_data(x,y,mask=None,verbose=False)
        adds data at points x with values y to the binned stat object
    plot()

    '''