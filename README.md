# Pristine photometric metallicity code 

The details of the method are described in Martin, Starkenburg et al. (2023) [arXiv:2308.01344]. 

The main code (Pristine_get-photometric-metallicity.ipynb) goes from input magnitudes (CaHK + Gaia broad-band) to photometric metallicities with uncertainties. 

It uses a "training sample" (a sample for which spectroscopic metallicities are available) to build a photometric metallicity grid in the Pristine colour-colour space (x = (BP-RP) and y = (CaHK-G) - 2.5*(BP-RP)). The training sample can be rebuilt using the code in the "make_training" folder, or one can use the pre-computed giants and dwarfs training sets in the "catalogues" folder. All observed stars are interpolated on this grid to assign a photometric metallicity.

The photometry is iteratively dereddened in the code, because the extinction coefficients for G, BP and RP depend on the effective temperature, metallicity and extinction itself. 

Uncertainties on the photometric metallicities are derived using Monte Carlo sampling of the uncertainties for each of the photometric bands. 

The code uses the following packages:
- numpy
- pandas
- matplotlib
- vaex
- astropy
- scipy
- dustmaps
