"""
Needs some docstrings!
"""

from Rascal_Sampler import *
import mangle
from scipy.integrate import quad
from scipy.interpolate import interp1d
from numpy import loadtxt,linspace,arccos,arctan2,mean,array,transpose,ones,cross
from numpy.random import rand

def init(opts):

	Omega_m = float(opts['omega_m'])
	h = float(opts['h'])

	c = 299792.458
	maskcut = 1.

	ramin  = float(opts['ramin'])
	ramax  = float(opts['ramax'])
	decmin = float(opts['decmin']) +90.
	decmax = float(opts['decmax']) +90.
	zmin   = float(opts['zmin'])
	zmax   = float(opts['zmax'])

	nbar_file = opts['nbar_file']	#Currently it's assumed that this n(z) is defined from 0.43-0.7
	mask_file = opts['mask_file']

	try:
		eff_area = opts['eff_area']
	except KeyError:
		eff_area = 6.851419E+03
	
	try:
		P_fkp = float(opts['p_fkp'])
	except KeyError:
		P_fkp = 0	#When no P_fkp is specified, use wfkp from the nbar file

	# Loading the nbar file
	zcen,zlo,zhi,nbar,wfkp,shell_vol,ngal = loadtxt(nbar_file,unpack=True,)

	#Distance function
	def Einv(z):
		return 1./sqrt( Omega_m*(1+z)**3 + (1-Omega_m) )

	nz = 10000
	ztable = linspace(min(zlo),max(zhi),nz)
	dtable = zeros(nz)
	for i in range(nz):
		dtable[i] = (c/100) * quad(Einv,0,ztable[i])[0]

	d = interp1d(ztable,dtable)

	# Computing our own nbar
	rlo = d(zlo)
	rhi = d(zhi)

	my_shell_vol = (1./3) * (rhi**3 - rlo**3) * eff_area * (4*pi/41253) #Last bit is the deg^2 -> steradians conversion
	my_nbar = ngal/shell_vol
	nbar_all = interp1d(d(zcen),my_nbar,bounds_error=False,fill_value=0)
	
	f = where( (zcen>zmin) & (zcen<zmax) )[0]
	my_rvals = zeros(len(f)+2)
	my_rvals[0] = d(zmin)
	my_rvals[-1] = d(zmax)
	my_rvals[1:-1] = d(zcen[f])
	nbar_r = interp1d(my_rvals,nbar_all(my_rvals),bounds_error=False,fill_value=0)
	
	if (P_fkp != 0):
		def weight_r(r):
			return 1/(1+P_fkp * nbar_r(r))
	else:
		wbar_all = interp1d(d(zcen),wfkp,bounds_error=False,fill_value=0)
		weight_r = interp1d(my_rvals,wbar_all(my_rvals),bounds_error=False,fill_value=0)
	
	# Load the mask
	mask = mangle.Mangle(mask_file)

	#########

	def nw_fn(r_vec):
		r = anorm(r_vec)
		x,y,z = transpose(r_vec)
		dec = arccos(x/r)*(180/pi) - 90
		ra  = arctan2(z,y) * (180/pi)
		f = where(ra<0)
		ra[f] += 360
		mask_w = mask.get_weights(ra,dec)
		if maskcut != 1:
			f = where(w>maskcut)
			w[f] = 1.
		n = nbar_r(r)
		w = weight_r(r)
		return n*w*mask_w

	def w_fn(r_vec):
		r = anorm(r_vec)
		w = weight_r(r)
		return w
	
	return nw_fn,w_fn,d
