from RASCAL_Sampler import *
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
#	zmin   = float(opts['zmin'])
#	zmax   = float(opts['zmax'])

	nzw_file  = opts['nzw_file']	#Currently it's assumed that this n(z) is defined from 0.43-0.7
	mask_file = opts['mask_file']

	#Distance function
	def Einv(z):
		return 1./sqrt( Omega_m*(1+z)**3 + (1-Omega_m) )

	nz = 10000
	ztable = linspace(0.42,0.71,nz)
	dtable = zeros(nz)
	for i in range(nz):
		dtable[i] = (c/100) * quad(Einv,0,ztable[i])[0]

	d = interp1d(ztable,dtable)

	# Working on n(z)
	zcen,nbar,wvec = loadtxt(nzw_file,unpack=True)
	dz = zcen[0]-0.43
	zbdy = zeros(len(zcen)+1)
	zbdy[:-1] = zcen-dz
	zbdy[-1]  = zcen[-1]+dz

	mask = mangle.Mangle(mask_file)

	f = where( ( zcen>=0.43) & (zcen<=0.7) )
	g = where( (zbdy >= 0.43) & (zbdy <= 0.7) )

	#########

	nVol = sum(mask.weights * mask.areas) * trapz(d(zcen[f])**2 * nbar[f] , d(zcen[f]))

	n1inv,n1pdf = pinv_pair(nbar[f]    * d(zcen[f])**2,d(zbdy[g]))
	n2inv,n2pdf = pinv_pair(nbar[f]**2 * d(zcen[f])**2 * wvec[f]**2,d(zbdy[g]))
	n3inv,n3pdf = pinv_pair(nbar[f]**3 * d(zcen[f])**2,d(zbdy[g]))
	n4inv,n4pdf = pinv_pair(nbar[f]**4 * d(zcen[f])**2,d(zbdy[g]))

	nb_lohi = zeros(2)
	nb_lohi[0] = nbar[0] + ( (nbar[1]-nbar[0])/(zcen[1]-zcen[0]) )*(0.43-zcen[0])
	nb_lohi[1] = nbar[-1] + ( (nbar[-1]-nbar[-2])/(zcen[-1]-zcen[-2]) )*(0.7-zcen[-1])
	wv_lohi = zeros(2)
	wv_lohi[0] = wvec[0] + ( (wvec[1]-wvec[0])/(zcen[1]-zcen[0]) )*(0.43-zcen[0])
	wv_lohi[1] = wvec[-1] + ( (wvec[-1]-wvec[-2])/(zcen[-1]-zcen[-2]) )*(0.7-zcen[-1])


	nb_lohi = interp([0.43,0.7],zcen,nbar)
	wv_lohi = interp([0.43,0.7],zcen,wvec)
	f = where( (zcen > 0.43) & (zcen<0.7) )
	zfid = zeros(len(f[0])+2)
	nfid = zeros(shape(zfid))
	wfid = zeros(shape(zfid))
	zfid[0],nfid[0],wfid[0] = 0.43,nb_lohi[0],wv_lohi[0]
	zfid[-1],nfid[-1],wfid[-1] = 0.7,nb_lohi[1],wv_lohi[1]
	zfid[1:-1],nfid[1:-1],wfid[1:-1] = zcen[f],nbar[f],wvec[f]

	nbar_r = interp1d(d(zfid),nfid,bounds_error=False,fill_value=0)
	weight_r = interp1d(d(zfid),wfid,bounds_error=False,fill_value=0)

	def nwr_fn(r_vec):
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
		return n*w*mask_w*r

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
	
	return nwr_fn,nw_fn,w_fn,d