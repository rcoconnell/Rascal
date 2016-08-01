"""
Simple_Survey describes the geometry of a simple survey. Options are passed to it from Rascal 
via New_Survey. These options describe a simple box in RA and dec, and the number density of
the survey. Since the number density is uniform, weights are all 1. It also takes in parameters 
describing a LCDM cosmology. Redshift boundaries are implemented in New_Survey.

It also serves as a model survey library for users who want to implement their own more detailed
survey geometries. Note that it returns several functions that take in 3-component vectors 
(nwr_fn, nw_fn, and w_fn) and return the appropriate quantity, as well as the approriate distance 
function d(z).
"""

from Rascal_Sampler import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from numpy import loadtxt,linspace,arccos,arctan2,mean,array,transpose,ones,cross
from numpy.random import rand

def init(opts):

	Omega_m = float(opts['omega_m'])
	h = float(opts['h'])

	c = 299792.458

	ramin  = float(opts['ramin'])
	ramax  = float(opts['ramax'])
	decmin = float(opts['decmin']) +90.
	decmax = float(opts['decmax']) +90.
	
	n_density = float(opts['n_density'])

	#Distance function
	def Einv(z):
		return 1./sqrt( Omega_m*(1+z)**3 + (1-Omega_m) )

	nz = 10000
	ztable = linspace(0.42,0.71,nz)
	dtable = zeros(nz)
	for i in range(nz):
		dtable[i] = (c/100) * quad(Einv,0,ztable[i])[0]

	d = interp1d(ztable,dtable)

	def nw_fn(r_vec):
		"""
		nw_fn takes in an array of 3-component vectors and returns the product n*w evaluated
		at those points. In this simple survey, weights are assumed uniform.
		"""
		r = anorm(r_vec)
		return n_density*ones(len(r))

	def w_fn(r_vec):
		"""
		w_fn takes in an array of 3-component vectors and returns the weighting function evaluated
		at those points. In this simple survey, weights are assumed uniform.
		"""
		r = anorm(r_vec)
		return ones(len(r))
	
	return nw_fn,w_fn,d