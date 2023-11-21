# Pristine photometric metallicity code 

The details of the method are described in Martin, Starkenburg et al. (The Pristine survey -- XXIII. Data Release 1) [arXiv:2308.01344]. 

The main code (Pristine_get-photometric-metallicity.ipynb) goes from input magnitudes (CaHK + Gaia broad-band) to photometric metallicities with uncertainties. 

It uses a "training sample" (a sample for which spectroscopic metallicities are available) to build a photometric metallicity grid in the Pristine colour-colour space (x = (BP-RP) and y = (CaHK-G) - 2.5*(BP-RP)). The training sample can be rebuilt using the code in the "make_training" folder, or one can use the pre-computed giants and dwarfs training sets in the "catalogues" folder. All observed stars are interpolated on this grid to assign a photometric metallicity. The training sample is also used to create solar metallicity colour-colour space lines, which can be found in the "poly" folder. 

The photometry is iteratively dereddened in the code, because the extinction coefficients for G, BP and RP depend on the effective temperature, metallicity and extinction itself. Martin, Starkenburg et al. (2023) describe the derivation of the extinction polynomials used (which can be found in the "extpol" folder). 

Uncertainties on the photometric metallicities are derived using Monte Carlo sampling of the uncertainties for each of the photometric bands. 

--------------------

A test input sample "CaHKsyn_CaHKPr_v0.9_2023_08_01_dCaHK01-1percent+GDR3.csv" (= random 1% subset of stars in the Martin, Starkenburg et al. (2023) Pristine data release with d_CaHK_syn < 0.1 cross-matched with Gaia DR3) can be downloaded from: 
https://drive.google.com/file/d/1zfrGh9-55wCZPAWlSMbW_-SkL6xhZd9C/view?usp=sharing 

Once the full repository (the code as well as extinction polynomials, training sample etc.) plus the test input file have been downloaded (into the "catalogues" folder), one should be able to run the main code directly. Some customisations can be made in the "Initial settings" cell of the notebook, according to the preferences of the user. 

--------------------

The code uses the following packages (and it was tested with version numbers given in brackets, as well as slightly earlier versions):
- numpy (1.23.5)
- pandas (1.5.3)
- matplotlib (3.4.3)
- vaex (4.16.1)
- astropy (5.1)
- scipy (1.10.0)
- dustmaps (1.0.12)

Some issues were found when running with a newer version of vaex (4.17). 

