# struvana
A package to analyze data from StratusUV setup from Cerullo lab ([link](https://www.femtosecond.fisi.polimi.it/)). 

The jupyter notebooks explain basic usage. In particular:
* `Preprocess.ipynb` : explains how to load and preprocess raw data;
* `OscillationsAnalysis.ipynb` : explains how to look for oscillations in already preprocessed data. 
* `SVD Oscillations Analysis.ipynb` : explains how to clean data from noise with SVD and look for oscillations.

Note that this package require the following python packages:
* numpy
* matplotlib
* pandas

If you don't have them you can install with `pip install numpy matplotlib pandas`. If you are using a Anaonda python ditribution then is preferable to install with `conda`.
