# Rascal
Rapid sampler for large covariance matrices

This code is based on the method described in O'Connell et al. ([arxiv:1510.01740](https://arxiv.org/abs/1510.01740)). 
If you use this code in a project, please cite that paper.
*Rascal* uses importance sampling to quickly estimate the covariance matrices for 2-point correlation functions.
The survey geometry and 2-point correlation function are each described in separate libraries,
to facilitate applications to a wide variety of situations.

Usage is simply:
`python Rascal_rmu.py config.txt`

In this example `config.txt` is a file that sets the parameters for a run of Rascal. In the 
`Survey` section we specify the range of radial bins considered (in units of *h*<sup>-1</sup>Mpc),
the number of *r* and *μ* bins, and the number of cores to use. The `Output` section specifies 
the location and prefix for output files.

A run of `Rascal_rmu.py` will produce a covariance matrix for a 2-point correlation function
estimated in bins of *r* and *μ*. The output is stored in a "pickle" file (.pkl), which can be loaded in Python with the `pickle`
module. Loading it will provide a dictionary with the 2-point, 3-point, and 4-point contributions
to the covariance matrix, as well as many other parameters describing that run of *Rascal*.
My hope is that putting all of this information in a single file will be convenient. To generate a 
Gaussian covariance matrix from a .pkl file, do something like this:

```
import pickle
rascal_run = pickle.load(open('Rascal_Output.pkl'))
gaussian_covariance = rascal_run['c2'] + rascal_run['c3'] + rascal_run['c4']
```

The `Survey` and `Corr` sections describe the survey geometry and correlation function, 
respectively. These sections always specify `surveylib` and `corrlib`, the python libraries
that describe the survey geometry and correlation function. Additional options provided
in these sections are passed through to those libraries.

## "Simple" Libraries

We currently provide two sets of libraries. The first are numerically very fast, but do not
describe a realistic survey. `Simple_Survey.py` describes a survey with uniform number density,
observed in a box described by boundaries at fixed RA and dec, and a fixed range in redshift.
`Linear_xi.py` provides a linear theory correlation function. It takes as arguments *b*<sup>2</sup>*σ*<sub>8</sub>,
which specifies the amplitude of the correlation function for your tracer of choice, and 
*β*, which specifies the level of redshift space distortions. An example of use of there libraries
can be found in `config_simple.txt`.

## "Realistic" Libraries

The other set of libraries, `BOSS_Like_Survey.py` and `BOSS_Like_xi.py`, use a realistic
survey geometry (redshift distribution and survey mask) and correlation function. `BOSS_Like_Survey.py`
takes in a file describing the *n*(*z*) and a separate file specifying the survey mask. An example 
*n*(*z*) is provided here (`nbar_DR12v5_CMASS_North_om0p31_Pfkp10000.dat`) which describes
the redshift distribution of galaxies in the northern galactic cap for the CMASS sample of BOSS.
The *n*(*z*) is based on galaxy counts in redshift bins, and will be updated dynamically to
accommodate if you change the survey cosmology. The corresponding mask file is larger and is 
*not* provided here. Instead, it can be downloaded from `http://data.sdss3.org/sas/dr12/boss/lss/mask_DR12v5_CMASS_North.ply`
The method used to generate these files is described in Reid et al., ([arxiv:1509.06529](https://arxiv.org/abs/1509.06529)).

`BOSS_Like_xi.py` requires a correlation function evaluated on an *r*-*μ* grid. To approximate
the non-linear 2-point correlation function we use the average redshift-space correlation function 
observed in 1,000 QPM mocks. The mocks are described in White et al., ([arxiv:1309.5532](http://arxiv.org/abs/1309.5532)).