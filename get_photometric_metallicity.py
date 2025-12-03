#!/usr/bin/env python
"""
Pristine Photometric Metallicity Estimator

This script calculates photometric metallicities for stars using CaHK photometry
and Gaia photometry. It performs iterative dereddening, Monte Carlo error estimation,
and outputs metallicity estimates for both giant and dwarf solutions.

Based on the methodology described in Martin, Starkenburg et al. 2023.

Authors: 
v1: Else Starkenburg (2017) - original IDL code, using SDSS photometry
v2: Anna Esselink and Akshara Viswanathan (2022) - Python translation of v1, Jupyter notebook format
v3: Anke Ardern-Arentsen (2023) - major upgrade of v2. Adapted for Gaia, used for Pristine DR1. 
v4: Anke Ardern-Arentsen (July 2025) - updated to use Polars instead of vaex, improved handling of large files
v5: Anke Ardern-Arentsen (December 2025) - converted v4 Jupyter notebook to Python script with command-line arguments
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import polars as pl
import numpy as np
import os
import time
import argparse

import matplotlib
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

from scipy.interpolate import RegularGridInterpolator

from dustmaps.bayestar import BayestarQuery
from dustmaps.sfd import SFDQuery
from dustmaps.gaia_tge import GaiaTGEQuery

import pickle
import fitsio

from photmet_utils import (
    makegrid,
    polynomial_model,
    Teff_casagrande_marginalised,
    apply_final_metallicity_rules,
    plot_heatmap
)

# Set matplotlib font size
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 20}
matplotlib.rc('font', **font)


# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

## Example usage (leaving all other settings to the defaults):
# ./get_photometric_metallicity.py --datafile ../catalogues/Pristine_merged_detections_all_until_24Bm05+basicGaia.fits --outformat fits --cat Pristine --cat2 internal --version DR2_v1 --outfol ../photmet_out/

def parse_arguments():
    """
    Parse command-line arguments for the photometric metallicity estimator.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Calculate photometric metallicities for stars using CaHK and Gaia photometry.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output files and catalogue settings
    parser.add_argument('--datafile', type=str,
                        default='../catalogues/Pristine_merged_detections_all_until_24Bm05+basicGaia.fits',
                        help='Input file for which to derive photometric metallicities (.csv or .fits)')
    
    parser.add_argument('--outformat', type=str,
                        default='csv',
                        help='Output file format (csv or fits)')
    
    parser.add_argument('--version', type=str,
                        default='DR2_v1',
                        help='Run name for the sample, will be used in output filenames.')
    
    parser.add_argument('--cat', type=str, choices=['CaHKsyn', 'Pristine'],
                        default='Pristine',
                        help='Which CaHK magnitudes to use, synthetic or real Pristine')
    
    parser.add_argument('--cat2', type=str, choices=['internal', 'public'],
                        default='internal',
                        help='Pristine catalogue type: "internal" for internal Pristine catalogue, "public" otherwise')

    # C* calculation
    parser.add_argument('--compute-cstar', choices=['true','false'],
                        default='true',
                        help='Compute and add Cstar (and Cstar_1sigma). Use true or false.')
    
    # Monte Carlo settings
    parser.add_argument('--n-mc', type=int,
                        default=100,
                        help='Number of Monte Carlo iterations for [Fe/H] uncertainties')
    
    # Extinction settings
    parser.add_argument('--getext-data', action='store_true',
                        default=True,
                        help='Derive new E(B-V) from dustmap')
    
    parser.add_argument('--no-getext-data', dest='getext_data', action='store_false',
                        help='Use pre-existing E(B-V) column instead of calculating from dustmap')
    
    parser.add_argument('--ext-type', type=str,
                        choices=['bay', 'sfd', 'tge', 'zero'],
                        default='sfd',
                        help='Dustmap to use: "bay" (Bayestar19), "sfd" (SFD), "tge" (Gaia TGE), or "zero" (no correction)')
    
    parser.add_argument('--ext-col', type=str,
                        default='e_bv_bay8',
                        help='Name of existing E(B-V) column to use (only if --no-getext-data is set)')
    
    parser.add_argument('--tge-fudge', type=float,
                        default=0.76,
                        help='Fudge factor for TGE extinction map')

    # Other files and paths (don't change if following the standard directory structure)
    parser.add_argument('--train-dwarfs', type=str,
                        default='catalogues/TrainingSample_may2023_v4+CaHK_dwarfs.csv',
                        help='Training sample file for dwarfs')
    
    parser.add_argument('--train-giants', type=str,
                        default='catalogues/TrainingSample_may2023_v4+CaHK_giants.csv',
                        help='Training sample file for giants')
    
    parser.add_argument('--extpol-dwarfs', type=str, 
                        default='extpol/final_dwarf_polynomial_parameters_anke+v.csv',
                        help='Extinction polynomial file for dwarfs')

    parser.add_argument('--extpol-giants', type=str,
                        default='extpol/final_giant_polynomial_parameters_anke+v.csv',
                        help='Extinction polynomial file for giants')
    
    parser.add_argument('--poly-folder', type=str,
                        default='poly/',
                        help='Folder with training sample polynomials')
    
    parser.add_argument('--outfol', type=str,
                        default='photmet_out/',
                        help='Output folder for photometric metallicity results')
    
    return parser.parse_args()


# Parse command-line arguments
args = parse_arguments()

# ------------------------------------------------------------------------------
# Set parameters based on input above
# ------------------------------------------------------------------------------

datafile = args.datafile
cat = args.cat
cat2 = args.cat2
samp = f'{cat}-{args.version}'
allout = f'{args.outfol}results_{samp}_g+dw.{args.outformat}'
compute_Cstar = (args.compute_cstar == 'true')
trainsubsetfile_dwarfs = args.train_dwarfs
trainsubsetfile_giants = args.train_giants
polyfol = args.poly_folder
n_MC = args.n_mc
getext_data = args.getext_data
getext_type_science = args.ext_type
extcol = args.ext_col
extpol_dwarfs = args.extpol_dwarfs
extpol_giants = args.extpol_giants

if getext_type_science == 'tge':
    tge_fudge = args.tge_fudge

# ------------------------------------------------------------------------------
# define colour space (Don't touch these unless you know what you are doing!)
# ------------------------------------------------------------------------------

bf = 2.5       # (CaHK-G) - bf*(BP-RP)
step = 0.02    # pixel size in the photometric metallicity grid

minx = 0.45    # grid min limit (BP-RP)
maxx = 1.55    # grid max limit (BP-RP)

print (allout)


raise SystemExit()

# ==============================================================================
# SETUP - EXTINCTION PREP
# ==============================================================================

## load extinction polynomials (for details see Martin, Starkenburg et al. 2023, Section 4)
df_dwarf = pd.read_csv(extpol_dwarfs)
df_giant = pd.read_csv(extpol_giants)

fit_parameters_dwarf = np.array(df_dwarf)[:,2:22]
fit_parameters_giant = np.array(df_giant)[:,2:22]

## Query the relevant dustmap(s)

sfd = SFDQuery()
if getext_type_science == 'bay':
    bayestar = BayestarQuery(version='bayestar2019')
elif getext_type_science == 'tge':
    tge = GaiaTGEQuery()

## The SFD map is overestimating E(B-V) by 14% (see e.g. Schlafly & Finkbeiner 2011), so this factor is necessary to rescale
sfdfac = 0.86

## Extinction law value RV
RV_law = 3.1


# ==============================================================================
# SETUP - REFERENCE [Fe/H] GRID
# ==============================================================================

# ---------------------------------------------
# Get input for the reference grid
# ---------------------------------------------

# Define the filenames for our cached grid data
giants_grid_file = 'precomputed_grid_giants.pkl'
dwarfs_grid_file = 'precomputed_grid_dwarfs.pkl'

# --- Handle the GIANTS grid ---
if os.path.exists(giants_grid_file):
    print(f"Loading pre-computed giants grid from: {giants_grid_file}")
    with open(giants_grid_file, 'rb') as f:
        # Load the tuple of (data, X, Y) from the file
        data_gi_giants, X_giants, Y_giants = pickle.load(f)
else:
    print("Pre-computed giants grid not found. Computing now...")
    # If the file doesn't exist, run the original function
    data_gi_giants, X_giants, Y_giants = makegrid(
        trainsubsetfile_giants, 'giants', bf, step, minx, maxx, polyfol
    )
    print(f"Saving computed giants grid to: {giants_grid_file}")
    # Save the results as a tuple into a new .pkl file
    with open(giants_grid_file, 'wb') as f:
        pickle.dump((data_gi_giants, X_giants, Y_giants), f)

# --- Handle the DWARFS grid ---
if os.path.exists(dwarfs_grid_file):
    print(f"Loading pre-computed dwarfs grid from: {dwarfs_grid_file}")
    with open(dwarfs_grid_file, 'rb') as f:
        data_gi_dwarfs, X_dwarfs, Y_dwarfs = pickle.load(f)
else:
    print("Pre-computed dwarfs grid not found. Computing now...")
    data_gi_dwarfs, X_dwarfs, Y_dwarfs = makegrid(
        trainsubsetfile_dwarfs, 'dwarfs', bf, step, minx, maxx, polyfol
    )
    print(f"Saving computed dwarfs grid to: {dwarfs_grid_file}")
    with open(dwarfs_grid_file, 'wb') as f:
        pickle.dump((data_gi_dwarfs, X_dwarfs, Y_dwarfs), f)


# ==============================================================================
# SETUP - CREATE INTERPOLATORS
# ==============================================================================

# RegularGridInterpolator requires the grid coordinates to be strictly ascending.
# Our Y coordinates are descending, so we flip them and the data.
Y_giants_asc = Y_giants[::-1]
feh_grid_giants = data_gi_giants["FeH"].reshape(len(X_giants), len(Y_giants)).T
feh_grid_giants_asc = feh_grid_giants[::-1, :]

Y_dwarfs_asc = Y_dwarfs[::-1]
feh_grid_dwarfs = data_gi_dwarfs["FeH"].reshape(len(X_dwarfs), len(Y_dwarfs)).T
feh_grid_dwarfs_asc = feh_grid_dwarfs[::-1, :]


# Create the interpolator objects ONCE
# Note the (Y, X) order because the grid shape is (n_y, n_x).
interpolator_giants = RegularGridInterpolator(
    (Y_giants_asc, X_giants), feh_grid_giants_asc,
    bounds_error=False, fill_value=np.nan, method='linear'
)
interpolator_dwarfs = RegularGridInterpolator(
    (Y_dwarfs_asc, X_dwarfs), feh_grid_dwarfs_asc,
    bounds_error=False, fill_value=np.nan, method='linear'
)


print("Interpolators created successfully.")


# ==============================================================================
# OPTIONAL: PLOT THE RESULTING GRID
# ==============================================================================

def plot_reference_grids():
    """
    Plot the reference [Fe/H] grids for giants and dwarfs.
    This is optional and can be called if you want to visualize the grids.
    """
    # Set limits for the colour bar
    maxc = 0.0
    minc = -4.0

    for sub, dati in zip(['giants','dwarfs'],[[data_gi_giants, X_giants, Y_giants],[data_gi_dwarfs, X_dwarfs, Y_dwarfs]]):

        data_gi, X, Y = dati

        fig = figure(figsize= (10,7))
        frame = fig.add_subplot(1,1,1)

        fehplot_gi = data_gi["FeH"].reshape((len(X), len(Y))).T  #final grid

        ## Plot
        image = frame.imshow(fehplot_gi,cmap='jet', interpolation='none', origin = 'lower',
                             extent=[X[0]-step/2,X[-1]-step/2,Y[0]+step/2,Y[-1]+step/2],
                            vmin = minc, vmax=maxc, aspect='auto')
        fig.colorbar(image, label='[Fe/H]')

        xxs = np.arange(0.4, 1.6, 0.01)

        if (sub == 'all') | (sub == 'dwarfs'):
            ybb_gi = 0.346945 - 3.19276*xxs + 1.85539*xxs**2 - 0.519518*xxs**3 - (bf-2.7)*xxs
            y3_gi = 0.613234 - 4.18968*xxs + 3.01172*xxs**2 - 0.813665*xxs**3 - (bf-2.7)*xxs
        elif sub == 'giants':
            ybb_gi = 0.587261 - 3.70764*xxs + 1.92016*xxs**2 - 0.323654*xxs**3 - (bf-2.7)*xxs
            y3_gi = 0.836432 - 4.55978*xxs + 2.84748*xxs**2 - 0.549882*xxs**3 - (bf-2.7)*xxs

        ppar = np.loadtxt('{}polypar_0.0_{}BPRP_{}.txt'.format(polyfol, round(bf,1),sub))
        pol = np.poly1d(ppar)

        ## Set axes
        plt.xlabel('(BP-RP)$_0$')
        plt.ylabel('(CaHK-G)$_0$ - {}(BP-RP)$_0$'.format(round(bf,1)))
        plt.title(sub)

        plt.show()


# Uncomment the following line if you want to plot the grids:
# plot_reference_grids()


# ==============================================================================
# PHOTOMETRIC [Fe/H] FOR DATA
# ==============================================================================

# ==============================================================================
# Step 2: LOAD & PREPARE DATA
# ==============================================================================
print("--- Step 2: Loading and Preparing Data from FITS ---")
t_start_load = time.time()

# --- A. Read the file ---

#  Check the file extension
file_extension = os.path.splitext(datafile)[1].lower()

if file_extension == '.csv':
    # --- METHOD 1: Fast in-memory loading for (small) CSV files ---
    print(f"Detected CSV file. Loading '{datafile}' with Pandas...")
    pdf = pd.read_csv(datafile)
    df = pl.from_pandas(pdf)
    print(f"Successfully loaded {len(df)} rows.")

elif file_extension in ['.fits', '.fit']:
    # --- METHOD 2: Memory-safe chunking for (large) FITS files ---
    print(f"Detected FITS file. Loading '{datafile}' in chunks...")

    chunk_size = 1_000_000
    data_chunks = []

    with fitsio.FITS(datafile) as fits:
        data_hdu = fits[1]
        total_rows = data_hdu.get_nrows()
        total_chunks = int(np.ceil(total_rows / chunk_size))
        
        for i in range(total_chunks):
            start = i * chunk_size
            end = start + chunk_size

            if (i + 1) % 10 == 0 or (i + 1) == total_chunks:
                print(f"  Processed {i + 1}/{total_chunks} chunks...")
            
            numpy_chunk = data_hdu[start:end]
            
            # Convert to native byte order to prevent errors
            native_dtype = numpy_chunk.dtype.newbyteorder('=')
            native_chunk = numpy_chunk.astype(native_dtype)
            
            # Use Pandas as the robust bridge to Polars
            pandas_chunk = pd.DataFrame(native_chunk)
            polars_chunk = pl.from_pandas(pandas_chunk)
            data_chunks.append(polars_chunk)

    df = pl.concat(data_chunks)
    print(f"Successfully loaded {len(df)} rows.")

else:
    # Handle unknown file types
    raise ValueError(f"Unsupported file type: '{file_extension}'. Please provide a .csv or .fits file.")


print(f"Loading took {time.time() - t_start_load:.2f} seconds.")


# --- B. Prepare the DataFrame (Renaming, etc.) ---

rename_map = {}
    
if cat == 'CaHKsyn' :
    if 'CaHK_syn' in df.columns:
        rename_map.update({'CaHK_syn': 'CaHK'})
    if 'd_CaHK_syn' in df.columns:
        rename_map.update({'d_CaHK_syn': 'd_CaHKorig'})
    elif 'cahkerr' in df.columns:
        rename_map.update({'cahkerr': 'd_CaHKorig'})
if cat == 'Pristine' and cat2 == 'public':
    rename_map.update({'CaHK_Pr': 'CaHK', 'd_CaHK_Pr': 'd_CaHKorig'})
if cat == 'Pristine' and cat2 == 'internal':
    # rename_map.update({'e_CaHK': 'd_CaHK'})
    rename_map.update({'merged_CaHK': 'CaHK', 'merged_d_CaHK': 'd_CaHKorig', 'merge_flag': 'merged_CASU_flag'})
    if 'RA_CaHK' not in df.columns:
        rename_map.update({'RA':'RA_CaHK', 'Dec':'Dec_CaHK'})
    # if 'pvar' in df.columns :
    #     rename_map.update({'pvar': 'Pvar'})
if 'RAdeg' in df.columns:
    cds_rename_dict_full = { 'RAdeg': 'ra', 
                       'DEdeg': 'dec', 
                       'Source': 'source_id2', 
                       'e_RAdeg': 'ra_error', 
                       'e_DEdeg': 'dec_error', 
                       'Plx': 'parallax', 
                       'e_Plx': 'parallax_error', 
                       'RPlx': 'parallax_over_error', 
                       'PM': 'pm', 
                       'pmRA': 'pmra', 
                       'e_pmRA': 'pmra_error', 
                       'pmDE': 'pmdec', 
                       'e_pmDE': 'pmdec_error', 
                       'NgAL': 'astrometric_n_good_obs_al', 
                       'gofAL': 'astrometric_gof_al', 
                       'chi2AL': 'astrometric_chi2_al', 
                       'epsi': 'astrometric_excess_noise', 
                       'sepsi': 'astrometric_excess_noise_sig', 
                       'Solved': 'astrometric_params_solved', 
                       'pscol': 'pseudocolour', 
                       'e_pscol': 'pseudocolour_error', 
                       'Nper': 'visibility_periods_used', 
                       'Dup': 'duplicated_source', 
                       'FG': 'phot_g_mean_flux', 
                       'e_FG': 'phot_g_mean_flux_error', 
                       'Gmag': 'phot_g_mean_mag', 
                       'FBP': 'phot_bp_mean_flux', 
                       'e_FBP': 'phot_bp_mean_flux_error', 
                       'BPmag': 'phot_bp_mean_mag', 
                       'FRP': 'phot_rp_mean_flux', 
                       'RPmag': 'phot_rp_mean_mag', 
                       'E(BP/RP)': 'phot_bp_rp_excess_factor', 
                       'BP-RP': 'bp_rp', 
                       'e_Gmag': 'phot_g_mean_mag_error', 
                       'e_BPmag': 'phot_bp_mean_mag_error', 
                       'e_RPmag': 'phot_rp_mean_mag_error', 
                       'o_Gmag': 'phot_g_n_obs', 
                       'RFG': 'phot_g_mean_flux_over_error', 
                       'o_BPmag': 'phot_bp_n_obs', 
                       'RFBP': 'phot_bp_mean_flux_over_error', 
                       'o_RPmag': 'phot_rp_n_obs', 
                       'e_FRP': 'phot_rp_mean_flux_error', 
                       'RFRP':'phot_rp_mean_flux_over_error'}
    
    # Check which columns exist and only rename those
    cds_rename_dict = {}
    missing_cds_cols = []
    for old_col, new_col in cds_rename_dict_full.items():
        if old_col in df.columns:
            cds_rename_dict[old_col] = new_col
        else:
            missing_cds_cols.append(old_col)
    
    # if missing_cds_cols:
    #     print(f"Note (will not break the code though): The following CDS columns are not present in the data: {missing_cds_cols}")
    
    rename_map.update(cds_rename_dict)

df = df.rename(rename_map)

# Add the 'd_CaHK' column, removing negative original CaHK errors
df = df.with_columns(
    pl.when(pl.col('d_CaHKorig') < 0).then(99.9).otherwise(pl.col('d_CaHKorig')).alias('d_CaHK')
)

# --- C. Check necessary columns ---

required_columns = ['ra', 'dec', 'CaHK', 'd_CaHK', 
                    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                    'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error', 'phot_g_mean_mag_error',
]

missing_columns = [col for col in required_columns if col not in df.columns]
has_ext_col = 'ebv' in df.columns or 'a0' in df.columns

if missing_columns:
    print(f"WARNING: The following required columns are missing from df: {missing_columns}")
elif not has_ext_col:
    if not getext_data:
        print("WARNING: extcol specified but not found in df. Switching to getext_data = True.")
        getext_data = True
else:
    print("All required columns for calculating photometric metallicities are present in df.")

if 'phot_bp_rp_excess_factor' not in df.columns:
    print('WARNING: Cstar cannot be calculated, phot_bp_rp_excess_factor is missing from df.')

print("Data loading complete.")


# --- D.  Add extinction ---

if getext_data:
    print("Calculating extinction...")
    coords = SkyCoord(df['ra'].to_numpy()*u.deg, df['dec'].to_numpy()*u.deg, frame='icrs', distance=8*u.kpc)
    
    if getext_type_science == 'bay':
        ebv_bay_data = bayestar(coords, mode='mean')
        df = df.with_columns(pl.Series("ebv", ebv_bay_data))

    elif getext_type_science == 'sfd':
        ebv_sfd_data = sfd(coords)
        df = df.with_columns(pl.Series("ebv", ebv_sfd_data))

    elif getext_type_science == 'tge':
        a0_tge = tge(coords) * tge_fudge
        df = df.with_columns(pl.Series("a0", a0_tge))

    print("Extinction calculation complete.")

else:
    # If using a pre-existing column, just rename it
    df = df.with_columns(pl.col(extcol).alias("ebv"))


# ==============================================================================
# Step 3: ITERATIVE DEREDDENING
# ==============================================================================
print("--- Step 3: Starting Iterative Dereddening ---")

for sub in ['giants', 'dwarfs']:
    t_dered = time.time()
    lab = '_dw' if sub == 'dwarfs' else ''
    print(f"Processing {sub}...")

    # A. Extract data from Polars to NumPy ONCE.
    CaHK_orig = df['CaHK'].to_numpy()
    G_orig = df['phot_g_mean_mag'].to_numpy()
    BP_orig = df['phot_bp_mean_mag'].to_numpy()
    RP_orig = df['phot_rp_mean_mag'].to_numpy()

    if getext_type_science == 'tge':
        A0val = df['a0'].to_numpy()
        ebv_np = A0val/(RV_law/0.95)
    else:
        ebv_np = df['ebv'].to_numpy()
        A0val = RV_law * ebv_np * sfdfac
    
    # B. Select parameters and interpolator
    if sub == 'giants':
        fit_params = fit_parameters_giant
        interpolator = interpolator_giants
    else:
        fit_params = fit_parameters_dwarf
        interpolator = interpolator_dwarfs
    model_g, model_bp, model_rp, model_cahk, model_a0av = (fit_params[i] for i in range(5))

    # C. Perform the iterative loop
    BP_RP_guess = BP_orig - RP_orig - ebv_np
    G_guess = G_orig - 2.0 * ebv_np
    CaHK_guess = CaHK_orig - 4.0 * ebv_np

    for i in range(5):
        y_color_guess = (CaHK_guess - G_guess) - bf * BP_RP_guess
        points_to_interp = np.column_stack((y_color_guess, BP_RP_guess))

        # Perform the raw interpolation
        phot_metal_corr_raw = interpolator(points_to_interp)
        
        # --- Apply metallicity rules for boundary conditions etc. ---
        phot_metal_corr = apply_final_metallicity_rules(phot_metal_corr_raw, BP_RP_guess, y_color_guess, sub, bf, maxx, minx, step)
        # ---------------------------------------------------------

        Teff_corr = Teff_casagrande_marginalised(BP_RP_guess, phot_metal_corr)

        if getext_type_science in ['bay', 'sfd']:
            A0AV_guess = polynomial_model((Teff_corr/5040, A0val, phot_metal_corr), *model_a0av)
            A0val = RV_law * A0AV_guess * ebv_np * sfdfac

        kg, kbp, krp, kcahk = (polynomial_model((Teff_corr/5040, A0val, phot_metal_corr), *p) for p in (model_g, model_bp, model_rp, model_cahk))
        
        G_guess, BP_guess, RP_guess, CaHK_guess = G_orig - kg * A0val, BP_orig - kbp * A0val, RP_orig - krp * A0val, CaHK_orig - kcahk * A0val
        BP_RP_guess = BP_guess - RP_guess

    A0AV_guess = polynomial_model((Teff_corr/5040, A0val, phot_metal_corr), *model_a0av)

    # D. Add final results back to the Polars DataFrame.
    df = df.with_columns([
        pl.Series(f"G_0{lab}", G_guess), pl.Series(f"BP_0{lab}", BP_guess),
        pl.Series(f"RP_0{lab}", RP_guess), pl.Series(f"CaHK_0{lab}", CaHK_guess),
        pl.Series(f"fehguess{lab}", phot_metal_corr)
    ])

    if (getext_type_science == 'tge'):
        df = df.with_columns([
            pl.Series(f"ebv", A0val/(RV_law/A0AV_guess))
        ])

    print(f"Dereddening for {sub} complete in {time.time() - t_dered:.2f} seconds.")


# ==============================================================================
# Step 4: C* CALCULATION
# ==============================================================================
if compute_Cstar:
    print("--- Step 4: Calculating C* ---")
    
    df = df.with_columns(
        # Create a temporary column for the BP-RP color
        bprp_temp = pl.col('phot_bp_mean_mag') - pl.col('phot_rp_mean_mag')
    ).with_columns(
        # Use Polars' powerful when/then/otherwise to apply the conditional logic
        Cstar_factor = pl.when(pl.col('bprp_temp') < 0.5)
                        .then(1.154360 + 0.033772*pl.col('bprp_temp') + 0.032277*pl.col('bprp_temp').pow(2))
                        .when((pl.col('bprp_temp') >= 0.5) & (pl.col('bprp_temp') < 4.0))
                        .then(1.162004 + 0.011464*pl.col('bprp_temp') + 0.049255*pl.col('bprp_temp').pow(2) - 0.005879*pl.col('bprp_temp').pow(3))
                        .otherwise(1.057572 + 0.140537*pl.col('bprp_temp'))
    ).with_columns(
        # Calculate the final Cstar and Cstar_1sigma columns
        Cstar = pl.col('phot_bp_rp_excess_factor') - pl.col('Cstar_factor'),
        Cstar_1sigma = 0.0059898 + 8.817481e-12 * pl.col("phot_g_mean_mag").pow(7.618399)
    ).drop(['bprp_temp', 'Cstar_factor']) # Drop the temporary columns to keep the DataFrame clean

else:
    # If not computing, add NaN columns so the DataFrame schema is consistent
    print("--- Step 4: Skipping C* Calculation ---")
    df = df.with_columns([
        pl.lit(np.nan, dtype=pl.Float64).alias("Cstar"),
        pl.lit(np.nan, dtype=pl.Float64).alias("Cstar_1sigma")
    ])


# ==============================================================================
# Step 5: MONTE CARLO SIMULATION
# ==============================================================================
print("--- Step 5: Starting Memory-Safe Monte Carlo Simulation ---")
t_mc = time.time()

# The helper function to process a single batch
def process_monte_carlo_batch(df_batch: pl.DataFrame, sub: str) -> pl.DataFrame:
    """
    Processes a small batch of the data to perform the MCMC, keeping memory usage low.
    """
    Ndata = len(df_batch)
    if Ndata == 0: return pl.DataFrame()

    lab = '_dw' if sub == 'dwarfs' else ''
    interpolator = interpolator_giants if sub == 'giants' else interpolator_dwarfs
    
    source_cols = [f'BP_0{lab}', f'RP_0{lab}', f'CaHK_0{lab}', f'G_0{lab}', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error', 'd_CaHK', 'phot_g_mean_mag_error']
    bp0, rp0, cahk0, g0, bp_err, rp_err, cahk_err, g_err = (df_batch[c].to_numpy() for c in source_cols)

    # Perform the Monte Carlo for the batch
    bp_i = bp0[:, None] + np.random.normal(scale=bp_err[:, None], size=(Ndata, n_MC))
    rp_i = rp0[:, None] + np.random.normal(scale=rp_err[:, None], size=(Ndata, n_MC))
    g_i = g0[:, None] + np.random.normal(scale=g_err[:, None], size=(Ndata, n_MC))
    cahk_i = cahk0[:, None] + np.random.normal(scale=cahk_err[:, None], size=(Ndata, n_MC))
    
    x_i = bp_i - rp_i
    y_i = cahk_i - g_i - bf * x_i
          
    points_mc = np.column_stack((y_i.ravel(), x_i.ravel()))
    interp_FeH_alli = interpolator(points_mc).reshape(Ndata, n_MC)
    
    # Calculate statistics 
    feh_50, feh_16, feh_84 = np.nanpercentile(interp_FeH_alli, [50, 16, 84], axis=1)
    mc_frac = np.sum(~np.isnan(interp_FeH_alli), axis=1) / n_MC
    
    # Calculate the direct value
    x_direct = bp0 - rp0
    y_direct = (cahk0 - g0) - bf * x_direct
    
    # Provide the correct (Y, X) coordinates.
    points_direct = np.column_stack((y_direct, x_direct))
    
    feh_direct_raw = interpolator(points_direct)
    
    return pl.DataFrame({
        f"FeH_{cat}{lab}_50th": feh_50,
        f"FeH_{cat}{lab}_16th": feh_16,
        f"FeH_{cat}{lab}_84th": feh_84,
        f"mcfrac_{cat}{lab}": mc_frac,
        f"FeH_{cat}{lab}": feh_direct_raw
    })


# --- Main loop ---
all_mc_results = []
for sub in ['giants', 'dwarfs']:
    print(f"Processing MC for {sub} in batches...")
    
    chunk_size = 250_000
    batch_results = []

    total_chunks = int(np.ceil(len(df) / chunk_size))
    
    for i, df_batch in enumerate(df.iter_slices(n_rows=chunk_size)):
        # Print progress every 10 chunks or at the end
        if (i + 1) % 10 == 0 or (i + 1) == total_chunks:
            print(f"  Processed {i + 1}/{total_chunks} chunks...")
        result_chunk = process_monte_carlo_batch(df_batch, sub=sub)
        batch_results.append(result_chunk)
    
    mc_results_df = pl.concat(batch_results)
    all_mc_results.append(mc_results_df)

# Horizontally concatenate the original DataFrame with all the new result columns
df = pl.concat([df] + all_mc_results, how="horizontal")

print(f"Monte Carlo simulation complete in {time.time() - t_mc:.2f} seconds.")


# ==============================================================================
# Step 6: FINAL EXPORT
# ==============================================================================
print(f"\n--- Step 6: Writing final catalog to {allout} ---")
t_export = time.time()

## select columns to save
cols = ['source_id2','ra','dec','ebv','CaHK','d_CaHK','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag',
       'phot_g_mean_mag_error' ,'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error']

for k in ['Pvar','RUWE', 'Cstar','Cstar_1sigma','RA_CaHK','Dec_CaHK','angDist']:
    if k in df.columns:
        cols.append(k)
    else:
        print(f'Column {k} not found in DataFrame, not added')

for lab in ['', '_dw']:
    cols.extend(['G_0{}'.format(lab),'BP_0{}'.format(lab), 'RP_0{}'.format(lab),'CaHK_0{}'.format(lab), 
                 'FeH_{}{}'.format(cat,lab),
                 'FeH_{}{}_50th'.format(cat,lab),'FeH_{}{}_16th'.format(cat,lab), 'FeH_{}{}_84th'.format(cat,lab),
                 'mcfrac_{}{}'.format(cat,lab)])
    
if cat == "Pristine":
    cols.append('merged_CASU_flag')

## save - check file extension to determine format
output_extension = os.path.splitext(allout)[1].lower()

if output_extension == '.csv':
    df.select(cols).write_csv(allout)
    print(f"Saved as CSV")
    
elif output_extension in ['.fits', '.fit']:
    print("Writing FITS file in chunks to manage memory...")
    
    chunk_size_export = 1_000_000  # Adjust based on available memory
    total_rows = len(df)
    total_chunks = int(np.ceil(total_rows / chunk_size_export))
    
    # Process in chunks
    for i, df_chunk in enumerate(df.select(cols).iter_slices(n_rows=chunk_size_export)):
        if (i + 1) % 10 == 0 or (i + 1) == total_chunks:
            print(f"  Written {i + 1}/{total_chunks} chunks...")
        
        # Convert chunk to pandas, then to numpy recarray
        chunk_pandas = df_chunk.to_pandas()
        chunk_recarray = chunk_pandas.to_records(index=False)
        
        if i == 0:
            # First chunk: create new file
            fitsio.write(allout, chunk_recarray, clobber=True)
        else:
            # Subsequent chunks: append to existing file
            with fitsio.FITS(allout, 'rw') as fits_file:
                fits_file[1].append(chunk_recarray)
        
        # Clean up memory
        del chunk_pandas, chunk_recarray
    
    print("Saved as FITS (chunked)")
    
else:
    # Default to CSV if extension not recognized
    print(f"Warning: Unrecognized extension '{output_extension}', defaulting to CSV")
    df.select(cols).write_csv(allout)

print(f"Export complete in {time.time() - t_export:.2f} seconds.")


# ==============================================================================
# OPTIONAL: PLOTTING FUNCTIONS
# ==============================================================================

def plot_results():
    """
    Generate diagnostic plots of the photometric metallicity results.
    This includes density plots and metallicity distributions in color-color space.
    """
    # =========================================================================
    # Iso-metallicity Lines and Plot Settings
    # =========================================================================

    bf = 2.5
    polyfol = 'poly/'

    xxs2 = np.arange(0.55, 1.4, 0.01)

    # Dwarf BB and -3.0 lines
    ybb_gi2_dw = 0.346945 - 3.19276*xxs2 + 1.85539*xxs2**2 - 0.519518*xxs2**3 - (bf-2.7)*xxs2
    y3_gi2_dw = 0.613234 - 4.18968*xxs2 + 3.01172*xxs2**2 - 0.813665*xxs2**3 - (bf-2.7)*xxs2

    # Giants BB and -3.0 lines
    ybb_gi2_g = 0.587261 - 3.70764*xxs2 + 1.92016*xxs2**2 - 0.323654*xxs2**3 - (bf-2.7)*xxs2
    y3_gi2_g = 0.836432 - 4.55978*xxs2 + 2.84748*xxs2**2 - 0.549882*xxs2**3 - (bf-2.7)*xxs2

    # Dwarf solar metallicity line
    ppar_d = np.loadtxt(f'{polyfol}polypar_0.0_{round(bf,1)}BPRP_dwarfs.txt')
    pol_d = np.poly1d(ppar_d)

    # Giants solar metallicity line
    ppar_g = np.loadtxt(f'{polyfol}polypar_0.0_{round(bf,1)}BPRP_giants.txt')
    pol_g = np.poly1d(ppar_g)
        
    # Plot settings for the colour bar
    cmap = matplotlib.cm.get_cmap('jet')
    maxc = -0.5
    minc = -3.0
    m3 = cmap(-(-3.0 - minc)/(minc + maxc))


    # =========================================================================
    # Plot 1: Density of all stars in the CCD
    # =========================================================================
    print("Plotting overall star density...")

    df2 = df.filter(
        (pl.col(f"ebv") < 0.5)
    )

    df_plot1 = df2.with_columns(
        x_plot = pl.col('BP_0') - pl.col('RP_0'),
        y_plot = (pl.col('CaHK_0') - pl.col('G_0')) - bf * (pl.col('BP_0') - pl.col('RP_0'))
    )

    # Using a helper function from photmet_utils
    fig, ax = plot_heatmap(
        df=df_plot1,
        x_col="x_plot",
        y_col="y_plot",
        limits=[[-0.2, 2.5], [0.2, -1.9]],
        shape=100,
        log_scale=True,
        cmap='viridis',
        xlabel=f'(BP-RP)$_0$',
        ylabel=f'(CaHK-G)$_0$ - {round(bf,1)}(BP-RP)$_0$',
        colorbar_label='log(density)'
    )

    # Add the overlay lines
    ax.plot(xxs2, y3_gi2_g, color=m3, linewidth=2)
    ax.plot(xxs2, ybb_gi2_g, color='black', lw=2)
    ax.plot(xxs2, pol_g(xxs2)+0.1, color='magenta', ls='--', lw=1.5)

    # ax.invert_yaxis()  # Invert y-axis to match the CCD orientation

    # plt.savefig(f'figures/CCD_density_{samp}.pdf')
    plt.show()


    # =========================================================================
    # Plot 2: Giant metallicities after quality cuts
    # =========================================================================
    print("Plotting giant metallicities with quality cuts...")

    df_giants_filtered = df.filter(
        (pl.col(f"FeH_{cat}") < 0) &
        (pl.col(f"FeH_{cat}") > -4) &
        (pl.col(f"mcfrac_{cat}") > 0.8) &
        (((pl.col(f"FeH_{cat}_84th") - pl.col(f"FeH_{cat}_16th")) / 2) < 0.5) &
        # (pl.col("Pvar") < 0.3) &
        # (pl.col("RUWE") < 1.4) &
        (pl.col("Cstar") < 5*pl.col("Cstar_1sigma"))
    )

    df_giants_filtered = df_giants_filtered.with_columns(
        x_plot = pl.col('BP_0') - pl.col('RP_0'),
        y_plot = (pl.col('CaHK_0') - pl.col('G_0')) - bf * (pl.col('BP_0') - pl.col('RP_0'))
    )

    fig, ax = plot_heatmap(
        df=df_giants_filtered,
        x_col="x_plot",
        y_col="y_plot",
        what=f"FeH_{cat}",
        limits=[[0.4, 1.6], [0.0, -1.7]],
        shape=100,
        cmap='jet',
        vmin=-3,
        vmax=0,
        xlabel=f'(BP-RP)$_0$',
        ylabel=f'(CaHK-G)$_0$ - {round(bf,1)}(BP-RP)$_0$',
        title='Giant solution',
        colorbar_label=f'Mean [Fe/H]'
    )

    # Add the overlay lines
    ax.plot(xxs2, y3_gi2_g, color=m3, linewidth=2)
    ax.plot(xxs2, ybb_gi2_g, color='black', linewidth=2)
    ax.plot(xxs2, pol_g(xxs2)+0.1, color='magenta', ls='--', lw=1.5)

    # ax.invert_yaxis()  # Invert y-axis to match the CCD orientation


    # plt.savefig(f'figures/CCD_FeHgiant_{samp}.pdf')
    plt.show()


    # =========================================================================
    # Plot 3: Dwarf metallicities after quality cuts
    # =========================================================================
    print("Plotting dwarf metallicities with quality cuts...")

    df_dwarfs_filtered = df.filter(
        (pl.col(f"FeH_{cat}_dw") < 0) &
        (pl.col(f"FeH_{cat}_dw") > -4) &
        (pl.col(f"mcfrac_{cat}_dw") > 0.8) &
        (((pl.col(f"FeH_{cat}_dw_84th") - pl.col(f"FeH_{cat}_dw_16th")) / 2) < 0.5) &
        # (pl.col("Pvar") < 0.3) &
        # (pl.col("RUWE") < 1.4) &
        (pl.col("Cstar") < 5*pl.col("Cstar_1sigma"))
    )

    df_dwarfs_filtered = df_dwarfs_filtered.with_columns(
        x_plot = pl.col('BP_0_dw') - pl.col('RP_0_dw'),
        y_plot = (pl.col('CaHK_0_dw') - pl.col('G_0_dw')) - bf * (pl.col('BP_0_dw') - pl.col('RP_0_dw'))
    )

    fig, ax = plot_heatmap(
        df=df_dwarfs_filtered,
        x_col="x_plot",
        y_col="y_plot",
        what=f"FeH_{cat}_dw",
        limits=[[0.4, 1.6], [0.0, -1.7]],
        shape=100,
        cmap='jet',
        vmin=-3,
        vmax=0,
        xlabel=f'(BP-RP)$_0$',
        ylabel=f'(CaHK-G)$_0$ - {round(bf,1)}(BP-RP)$_0$',
        title='Dwarf solution',
        colorbar_label=f'Mean [Fe/H]'
    )

    # Add the overlay lines
    ax.plot(xxs2, y3_gi2_dw, color=m3, linewidth=2)
    ax.plot(xxs2, ybb_gi2_dw, color='black', linewidth=2)
    ax.plot(xxs2, pol_d(xxs2)+0.1, color='magenta', ls='--', lw=1.5)

    # ax.invert_yaxis()  # Invert y-axis to match the CCD orientation


    # plt.savefig(f'figures/CCD_FeHdwarf_{samp}.pdf')
    plt.show()


# Uncomment the following line to generate plots:
# plot_results()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Pristine Photometric Metallicity Calculator")
    print("="*80 + "\n")
    
    print("Processing complete! Results saved to:")
    print(f"  {allout}")
    print("\nTo generate diagnostic plots, uncomment the plot_results() call at the end of the script.")
