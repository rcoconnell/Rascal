"""
This is a simple linear theory, redshift space correlation function. It takes as options
b^2 sigma_8 (to set the overall amplitude of the correlation function) and beta (which
sets the mu-dependence).
"""

from scipy.interpolate import interp1d
from numpy import concatenate, loadtxt
from Rascal_Sampler import *

def L2(x):
	"""
	Scipy's Legendre polynomials seem insane, so we'll make our own.
	"""
	return 0.5 * (3*x**2 - 1)

def L4(x):
	"""
	Scipy's Legendre polynomials seem insane, so we'll make our own.
	"""
	return 0.125*(35*x**4 - 30*x**2 +3)

def init(opts):
	# SimpleMultipoles was generated from a matter transfer function generated with CAMB.
	# The power spectrum use to generate SimpleMultipoles.csv was found to have sigma8 = 0.8.
	xifile = "SimpleMultipoles.csv"
	sigma8 = 0.8
	
	b2sigma8 = float(opts['b2sigma8'])
	beta = float(opts['beta'])
	
	rv,xi0,xi2,xi4 = loadtxt(xifile)
	
	# We need to do provide sensible values all the way down to r=0
	rv = concatenate(( [0.],rv ))
	xi0 = concatenate(( [0.],xi0 ))
	xi2 = concatenate(( [0.],xi2 ))
	xi4 = concatenate(( [0.],xi4 ))
	
	# Set the desired amplitude
	xi0 *= b2sigma8 / sigma8
	xi2 *= b2sigma8 / sigma8
	xi4 *= b2sigma8 / sigma8
	
	# Generate continuous functions for each multipole
	xi0_i = interp1d(rv,xi0,bounds_error = False,fill_value = 0.)
	xi2_i = interp1d(rv,xi2,bounds_error = False,fill_value = 0.)
	xi4_i = interp1d(rv,xi4,bounds_error = False,fill_value = 0.)

	# Add them all together
	def xi(r,mu):
		outv  = (1. + (2./3)*beta + (1./5)*beta**2) * xi0_i(r)
		outv += ((4./3)*beta + (4./7)*beta**2) * xi2_i(r) * L2(mu)
		outv += ((8./35)*beta) * xi4_i(r) * L4(mu)
		return outv
	
	# Need to generate r2xi as well
	
	rv_bdy = array(rv)
	rv_bdy[1:] += 1
	r2xi = pinv_pair( (xi0 * rv**2)[1:] , rv_bdy )
	
	return xi, r2xi