# GenMeteorSporadics

## Introduction
This is the supplementary code to the paper:
> Vida, D., Brown, P.G., Campbell-Brown, M. (2017) "Generating realistic synthetic meteoroid orbits", Icarus, 296, pp. 197 - 215

DOI: https://doi.org/10.1016/j.icarus.2017.06.020

Use this code to generate all graphs and datasets in the aforementioned paper.

## Requirements
The code is written in Python 3, and here are the libraries used:

- Numpy
- Matplotlib
- Scipy
- Cython (optional - only used for calculating nearest neighbour D_SH valules)


## Usage
Open and run the SporadicsGenerateSynthetic.py file - it will present graphs of meteor orbits generated using the Jopek & Bronikowska (2016) method E, and graphs of all three synthetic datasets (10, 0.1 and nonscalar bandwidth matrices) generated using the Kernel Density Estimation method.

If you want to use the generated orbits for your own work, here are the steps to take:

1. Use the function from LoadData.py module for parsing the CAMS or UFOorbit format data.
2. Use the generateKDEOrbits function from the SporadicsGenerateSynthetic.py to generate synthetic meteor orbits using the Kernel Density Estimation method.
3. Use the provided funtions for analyzing the data, or use them in your own analysis.

All functions are well documented for your convenience.

## Academic citing
For academic use, please cite the paper:
> Vida, D., Brown, P.G., Campbell-Brown, M. (2017) "Generating realistic synthetic meteoroid orbits", Icarus, 296, pp. 197 - 215

## References
- Jopek, T. J., & Bronikowska, M. (2016). "Probability of coincidental similarity among the orbits of small bodies – I. Pairing." Planetary and Space Science.
