# Pristine photometric metallicity code 

The details of the method are described in Martin, Starkenburg et al. (The Pristine survey -- XXIII. Data Release 1, 2024, A&A, 692A, 115)

The main code (Pristine_get-photometric-metallicity.ipynb) goes from input magnitudes (CaHK + Gaia broad-band) to photometric metallicities with uncertainties. 

It uses a "training sample" (a sample for which spectroscopic metallicities are available) to build a photometric metallicity grid in the Pristine colour-colour space (x = (BP-RP) and y = (CaHK-G) - 2.5*(BP-RP)). The training sample can be rebuilt using the code in the "make_training" folder, or one can use the pre-computed giants and dwarfs training sets in the "catalogues" folder. All observed stars are interpolated on this grid to assign a photometric metallicity. The training sample is also used to create solar metallicity colour-colour space lines, which can be found in the "poly" folder. 

The photometry is iteratively dereddened in the code, because the extinction coefficients for G, BP and RP depend on the effective temperature, metallicity and extinction itself. Martin, Starkenburg et al. (2023) describe the derivation of the extinction polynomials used (which can be found in the "extpol" folder). 

Uncertainties on the photometric metallicities are derived using Monte Carlo sampling of the uncertainties for each of the photometric bands. 

--------------------

A test input sample "CaHKsyn_CaHKPr_v0.9_2023_08_01_dCaHK01-1percent+GDR3.csv" (= random 1% subset of stars in the Martin, Starkenburg et al. (2023) Pristine data release with d_CaHK_syn < 0.1 cross-matched with Gaia DR3) can be downloaded from: 
https://drive.google.com/file/d/1zfrGh9-55wCZPAWlSMbW_-SkL6xhZd9C/view?usp=sharing 

Once the full repository (the code as well as extinction polynomials, training sample etc.) plus the test input file have been downloaded (into the "catalogues" folder), one should be able to run the main code directly. Some customisations can be made in the "Initial settings" cell of the notebook, according to the preferences of the user. 

-----------------------

## Updates since version 1 (used for Pristine DR1)

This is an updated version from the DR1 code, swapping out the Python vaex package for the Python polars package (both meant to efficiently deal with large catalogues). 

A new interpolation function is used to derive metallicities from the photometric metallicity grid. This has no significant effect for the vast majority of stars. There is a small effect (0.1-0.2 dex maximum) for hot EMP stars only, see the Pristine DR2 paper (Yuan, Ardern-Arentsen et al. 2026). 

For the user, the code works the same way and requires the same input/training files as the DR1 code. Apart from the package and interpolation swaps, these are some other changes:

- the output format for the metallicities has been updated to match that of DR1 (e.g. FeH_Pristine_50th or FeH_CaHKsyn_50th instead of FeH_gaia_direct)
- the first time the reference grid is computed from the training sample it is saved, the next time it is loaded directly rather than being recomputed. 
- some of the column renaming has been updated, but may require editing based on exactly what kind of file is given as input. A cell has been added to check that all the necessary columns are present before moving on. 
- some of the helper functions have been moved to a separate file, photmet_utils.py

--------------------

The code uses the following packages:

pandas
polars
numpy
matplotlib
astropy
scipy
pickle
fitsio
dustmaps

--------------------

NOTE: If you are running dustmaps for the first time, you will need to download the dustmap(s) you want to use (see the dustmaps documentation for more details) e.g. for the SFD map:

import dustmaps.sfd
dustmaps.sfd.fetch()

