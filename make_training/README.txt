------------
README
------------

part 1 = combine all the spectroscopic input samples into a master file & match to CaHK 

part 2 = select dwarf/giant subsets from master file and apply photometric quality cuts, plus fitting a polynomial to training sample with [Fe/H] = 0.0 for use in the photometric metallicity code


Inputs for the training sample (can all be found in the 'samples' folder except SSPP and APOGEE, which can be downloaded from the SDSS website):
- SSPP
- APOGEE
- Aguado+19
- LAMOST VMP/Li (cleaned using LAMOST DR8 instead of DR3)
- Pristine HR follow-up (two tables)
- Bootes stars
- PASTEL (file prepared by Y. Huang)
