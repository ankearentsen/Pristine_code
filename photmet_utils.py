# photmet_utils.py

"""
A utility module containing functions for the photometric metallicity pipeline.
"""

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel, convolve
import polars as pl
import matplotlib.pyplot as plt


## Function to go from a training set to a reference grid (for giants & dwarfs separately)

def makegrid(trainsubsetfile, sub, bf, step, minx, maxx, polyfol):

    ## read training sample
    TSsel1 = pd.read_csv(trainsubsetfile)

    ## Define grid
    xplot_gi = TSsel1['BP_0'] - TSsel1['RP_0']
    yplot_gi = (TSsel1['CaHK_0'] - TSsel1['G_0']) - bf*(TSsel1['BP_0'] - TSsel1['RP_0'])
    fehplot = TSsel1['feh_lit']

    ## --------------------------
    ### MAKE METALLICITY GRID
    ## --------------------------

    ## Set x and y steps and range in CCD to use for the model
    xstep = step
    ystep = step
    X = np.arange(0.35+xstep/2, 1.7+xstep/2, xstep)
    Y = np.flip(np.arange(-1.8+ystep/2, 0.2+ystep/2, ystep))


    ## Make (empty) grid in the same shape
    data_gi = {"Xgrid": [], "Ygrid": [], "FeH_values": [], "empty":[], "FeH": []}
    data_gi["Xgrid"] = np.array([]);  data_gi["Ygrid"] = np.array([])

    for x in X:
        for y in Y:
            data_gi["Xgrid"] = np.append(data_gi["Xgrid"], x)
            data_gi["Ygrid"] = np.append(data_gi["Ygrid"], y)

    N = len(data_gi["Xgrid"])   #length array

    data_gi["FeH_values"] = (np.ones(N)*np.nan).astype(object)
    data_gi["FeH"] = np.ones(N)*np.nan
    data_gi["empty"] = np.zeros(N)

    ## Loop over the grid and derive metallicities for each cell
    for n in range(N):

        indlist = ((xplot_gi)>(data_gi["Xgrid"][n])-xstep/2) & ((xplot_gi)<(data_gi["Xgrid"][n] + xstep/2)) & ((yplot_gi)<(data_gi["Ygrid"][n])+ystep/2) & ((yplot_gi)>(data_gi["Ygrid"][n]-ystep/2))
        ind = indlist[indlist == True].index.values
        data_gi["FeH_values"][n] = fehplot[ind]   # assign values to cell

        if len(ind)> 4:  # if there are more than 4 stars, we take the mean with outlier detection
            data_gi["FeH"][n] = sigma_clipped_stats(fehplot[ind], sigma=2)[0]
        elif (len(ind)>= 2):
            data_gi["FeH"][n] = np.mean(fehplot[ind])   #no point in outlier detection if there are fewer stars
        else:
            data_gi["empty"][n] = 1   #cell is empty or has only 1 star

    ##  If a cell is empty, assigning metallicity value from the closest cell
    for n in range(N):
        xdist = np.ones(N)*np.nan; ydist= np.ones(N)*np.nan   #to store distances
        xval = data_gi["Xgrid"][n] ;  yval = data_gi["Ygrid"][n]
        if data_gi["empty"][n] == 1:  #check if cell is empty
            for i in range(N):
                if data_gi["empty"][i] == 0:  #if cell is not empty, distance is calculated
                    xdist[i] = abs(data_gi["Xgrid"][i]  - xval)
                    ydist[i] = abs(data_gi["Ygrid"][i]  - yval)
            dist = np.sqrt(xdist**2 + ydist**2)
            ind = np.nanargmin(dist)  #find index of closests cell (ignoring NaNs)
            if (xdist[ind] < 20*xstep) & (ydist[ind] < 20*ystep) &  (dist[ind] < 15*xstep):
                data_gi["FeH"][n] = data_gi["FeH"][ind]   #assign metallicity value


    # make montonically decreasing in the y-direction for grid points with [Fe/H] < -1.5 and BP-RP > 1.0
    # to avoid high metallicity contamination scattering in low-metallicity regime
    for n in range(len(X)):
        n *= len(Y)
        for i in range(1,len(Y)-1):
            if (data_gi["FeH"][i+n] > data_gi["FeH"][i+n-1]) & (data_gi["FeH"][i+n-1] < -1.5) & (data_gi["Xgrid"][i+n] > 1.0):
                data_gi["FeH"][i+n] = data_gi["FeH"][i+n-1]

    ## iso-metallicity lines for [Fe/H] = -3.0 and zero metallicity (= Black Body, BB)
    ## (were originally derived for the colour combination (CaHK - G) - 2.7*(BP-RP), hence the (bf-2.7) factor at the end)
    if (sub == 'all') | (sub == 'dwarfs'):
        ybb_gi = 0.346945 - 3.19276*(data_gi["Xgrid"]) + 1.85539*(data_gi["Xgrid"])**2 - 0.519518*(data_gi["Xgrid"])**3  - (bf-2.7)*data_gi["Xgrid"]
        y3_gi = 0.613234 - 4.18968*(data_gi["Xgrid"]) + 3.01172*(data_gi["Xgrid"])**2 - 0.813665*(data_gi["Xgrid"])**3 - (bf-2.7)*data_gi["Xgrid"]
    elif sub == 'giants':
        ybb_gi = 0.587261 - 3.70764*(data_gi["Xgrid"]) + 1.92016*(data_gi["Xgrid"])**2 - 0.323654*(data_gi["Xgrid"])**3 - (bf-2.7)*data_gi["Xgrid"]
        y3_gi = 0.836432 - 4.55978*(data_gi["Xgrid"]) + 2.84748*(data_gi["Xgrid"])**2 - 0.549882*(data_gi["Xgrid"])**3 - (bf-2.7)*data_gi["Xgrid"]


    ## assign [Fe/H] = -4.0 for stars closer to or above BB line
    data_gi["FeH"] = np.where(abs(data_gi["Ygrid"]  - ybb_gi) < (abs(data_gi["Ygrid"]  - y3_gi)), -4, data_gi["FeH"]) # if closer to zero-metallicity line
    data_gi["FeH"] = np.where(data_gi["Ygrid"]  < ybb_gi, -4, data_gi["FeH"]) # if above BB line

    ## Load 0.0 polynomial to limit the grid on the bottom & assign [Fe/H] = 0.0 for stars from +0.1 mag below the available training sample
    ## this polynomial has been created in part 2 of the "prepare training" code
    ppar = np.loadtxt('poly/polypar_0.0_{}BPRP_{}.txt'.format(round(bf,1),sub))
    pol = np.poly1d(ppar)
    data_gi["FeH"] = np.where(data_gi["Ygrid"]  > pol(data_gi["Xgrid"])+0.1, 0.0, data_gi["FeH"])

    ## Gaussian smoothing
    fehplot_gi = data_gi["FeH"].reshape((len(X), len(Y))).T  #reshaping data to 2D array

    kernel = Gaussian2DKernel(x_stddev=2)
    fehplot2_gi = convolve(fehplot_gi, kernel)

    ## corrections after smoothing
    data_gi["FeH"] = (fehplot2_gi.T).flatten()

    data_gi["FeH"] = np.where((data_gi["Ygrid"]  < ybb_gi) & (data_gi["Ygrid"]  > (ybb_gi - 0.2)), -4, data_gi["FeH"])
    data_gi["FeH"] = np.where(data_gi["Ygrid"]  < (ybb_gi -0.2), np.nan, data_gi["FeH"])

    cond = (data_gi["Ygrid"] <= (ybb_gi -0.2)) | (data_gi["Xgrid"]  < minx) | (data_gi["Xgrid"]  > maxx) | (data_gi["Ygrid"] >= pol(data_gi["Xgrid"])+0.3)
    data_gi["FeH"] = np.where(cond, np.nan, data_gi["FeH"])

    fehplot_gi = data_gi["FeH"].reshape((len(X), len(Y))).T  #final grid

    return data_gi, X, Y


    
def polynomial_model(TnormA0feh, const, a, b, c, a2, b2, c2, ab, ac, bc, a3, b3, c3, a2b, ab2, a2c, ac2, b2c, bc2, abc):

    # a is factor for T_norm; b is factor for A0; c is factor for [Fe/H]
    # T_norm = Teff/5040

    Tnorm, A0, feh = TnormA0feh

    val1 = const + a*Tnorm + b*A0 + c*feh + a2*Tnorm**2 + b2*A0**2 + c2*feh**2 + a3*Tnorm**3 + b3*A0**3 + c3*feh**3
    val2 = ab*Tnorm*A0 + ac*Tnorm*feh + bc*A0*feh
    val3 = a2b*Tnorm**2*A0 + ab2*Tnorm*A0**2 + a2c*Tnorm**2*feh + ac2*Tnorm*feh**2 + b2c*A0**2*feh + bc2*A0*feh**2
    val4 = abc*Tnorm*A0*feh

    return val1 + val2 + val3 + val4

def polynomial_model_Gaia(TnormA0feh, const, a, b, c, d,e,f,g,h,k):

    # a is factor for T_norm; b is factor for A0; c is factor for [Fe/H]
    # T_norm = Teff/5040

    #feh is not used here but is kept for consistency

    Tnorm, A0, feh = TnormA0feh

    val1 = const + a*Tnorm + b*Tnorm**2 + c*Tnorm**3 + d*A0 + e*A0**2 + f*A0**3 
    val2 = g*Tnorm*A0 + h*Tnorm**2*A0 + k*Tnorm*A0**2

    return val1 + val2


def Teff_casagrande_marginalised(X, feh):
    ## Function to derive photometric temperatures, following the IRFM by Casagrande et al. (2021)
    ## where X = (BP-RP)_0

    factor_logg = 3 # marginalise over logg by fixing it to a "middle value"

    a0, a1, a2, a3, a4, a5 = 7928, -3663.11, 803.3, -9.37, 0, 325.13
    a6, a7, a8, a9, a10 = -500.11, 279.48, -53.51, 0, -2.42
    a11, a12, a13, a14 = -128.03, 49.49, 5.91, 41.37

    teff_cas = a0 + a1*X + a2*X**2 + a3*X**3 + a4*X**5 + a5*factor_logg + a6*factor_logg*X + a7*factor_logg*X**2 +\
        a8*factor_logg*X**3 + a9*factor_logg*X**5 + a10*feh + a11*feh*X + a12*feh*X**2 + a13*feh*X**3 + a14*feh*factor_logg*X

    ## stay within validity of Casagrande relation
    teff_cas[teff_cas < 3500] = 3500
    teff_cas[teff_cas > 9000] = 9000

    return teff_cas



# Helper function for applying final rules
def apply_final_metallicity_rules(feh_array, bprp_array, y_color_array, sub, bf, maxx, minx, step):
    """Applies the set of NaN and edge-case rules."""
    if sub == 'giants':
        ybb_gi = 0.587261 - 3.70764*bprp_array + 1.92016*bprp_array**2 - 0.323654*bprp_array**3 - (bf-2.7)*bprp_array
    else:
        ybb_gi = 0.346945 - 3.19276*bprp_array + 1.85539*bprp_array**2 - 0.519518*bprp_array**3 - (bf-2.7)*bprp_array
    feh_cleaned = np.copy(feh_array)
    nan_mask = np.isnan(feh_cleaned)
    edge_case_mask = ( (bprp_array <= maxx + step) & (bprp_array >= minx - step) & (y_color_array < ybb_gi - 0.1) )
    feh_cleaned[nan_mask & edge_case_mask] = -4.0
    if sub == 'giants':
        giant_edge_case_mask = ( (y_color_array < -0.5) & (feh_cleaned != -4.0) & (bprp_array >= maxx - step / 2) )
        feh_cleaned[np.isnan(feh_cleaned) & giant_edge_case_mask] = -2.0
    return np.nan_to_num(feh_cleaned, nan=0.0)



def plot_heatmap(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    what: str = None, # type: ignore
    limits: list = None, # type: ignore
    shape: int = 128,
    log_scale: bool = False,
    xlabel: str = None, # type: ignore
    ylabel: str = None, # type: ignore
    title: str = None, # type: ignore
    colorbar_label: str = None, # type: ignore
    cmap: str = 'viridis',
    vmin: float = None, # type: ignore
    vmax: float = None, # type: ignore
    figsize: tuple = (9, 6)
):
    """
    A helper function to create 2D heatmaps from a Polars DataFrame.
    
    """
    plot_cols = [x_col, y_col]
    if what:
        plot_cols.append(what)
        
    df_clean = df.filter(pl.all_horizontal(pl.col(c).is_finite() for c in plot_cols))
    
    if df_clean.is_empty():
        print("Warning: No finite data to plot after cleaning and filtering.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title if title else 'No Data to Display')
        ax.set_xlabel(xlabel if xlabel else x_col)
        ax.set_ylabel(ylabel if ylabel else y_col)
        ax.text(0.5, 0.5, "No valid data in the selected range.", ha='center', va='center', transform=ax.transAxes)
        if limits:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
        return fig, ax

    x = df_clean[x_col].to_numpy()
    y = df_clean[y_col].to_numpy()

    hist_range = [[min(lim), max(lim)] for lim in limits] if limits else None

    if what is None:
        counts, x_edges, y_edges = np.histogram2d(x, y, bins=shape, range=hist_range)
        grid = counts
        if log_scale:
            grid = np.log10(grid, out=np.full_like(grid, -np.inf), where=(grid > 0))
    else:
        weights = df_clean[what].to_numpy()
        sums, x_edges, y_edges = np.histogram2d(x, y, bins=shape, range=hist_range, weights=weights)
        counts, _, _ = np.histogram2d(x, y, bins=shape, range=hist_range)
        grid = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=(counts != 0))

    fig, ax = plt.subplots(figsize=figsize)
    
    # The extent is now always derived from the histogram edges, ensuring the
    # axes are labeled with the correct data values.
    plot_extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    # -----------------------

    im = ax.imshow(
        grid.T,
        origin='lower',
        extent=plot_extent,
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    
    # If the user provided inverted limits, we respect that in the final view.
    if limits:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    
    cbar = fig.colorbar(im)
    if colorbar_label: cbar.set_label(colorbar_label)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
        
    return fig, ax
