"""
This version of Rascal generates a covariance matrix for a 2-point correlation function
evaluated in bins of (r,mu). It implements the method described in O'Connell et al, arxiv:1510.01740

It has been modified to compute a model version of a jackknife covariance.
"""

from ConfigParser import SafeConfigParser
from multiprocessing import Pool
from Rascal_Sampler import *
from numpy import loadtxt,savetxt,max,min,interp,abs,linspace,meshgrid,zeros,ones,cumsum,where,reshape,shape,array,mean,arange,std,trace,log,diag,outer,arccos
from numpy.linalg import det,inv,eigvals
from scipy.interpolate import RectBivariateSpline,interp1d
from scipy.optimize import fmin
from time import time
import pickle
from numpy.random import seed
import Rascal_Classes as Rascal_classes
import sys
import healpy as hp

if len(sys.argv)!=2:
	print "Config file?"
	sys.exit()

Config = SafeConfigParser()
Config.read(sys.argv[1])

#Setting up the survey geometry and correlation function
survey_opts = dict(Config.items('Survey'))
corr_opts = dict(Config.items('Corr'))
mysurvey = Rascal_classes.Survey(corr_opts,survey_opts)

#Setting up the sampler
rmin  = Config.getfloat('Sampler','rmin')
rmax  = Config.getfloat('Sampler','rmax')
nr    = Config.getfloat('Sampler','nr')
rmins = linspace(rmin,rmax,nr,endpoint=False)

nmu = Config.getfloat('Sampler','nmu')
mumins = linspace(0,1,nmu,endpoint=False)

#Setting up the output
fileprefix = Config.get('Output','fileprefix')

#Report on how things are set up:
print "Redshift range is "+str(mysurvey.zmin)+"-"+str(mysurvey.zmax)
print "Mask is bounded in RA by "+str(mysurvey.RAmin)+"-"+str(mysurvey.RAmax)
print "Mask is bounded in dec by "+str(mysurvey.decmin)+"-"+str(mysurvey.decmax)

#######

#Some flags
RASCAL_version = '0.96'

#More sampler options
ncores  = Config.getint('Sampler','ncores')
NSamp   = 4e5
repeats = 4
nloops  = 10

#######

nbins = len(rmins)*len(mumins)

bnl = []
dmu = mumins[1] - mumins[0]
dr  = rmins[1]  - rmins[0]
for ri in range(len(rmins)):
	for mui in range(len(mumins)):
		bnl.append( bin(rmins[ri],mumins[mui],dr,dmu) )

######

def quick_px(r_vec):
	#Copied from BOSS_Like_Survey
	r = anorm(r_vec)
	x,y,z = transpose(r_vec)
	dec = arccos(x/r)*(180/pi) - 90
	ra  = arctan2(z,y) * (180/pi)
	f = where(ra<0)
	ra[f] += 360
	
	#Copied from splitter
	theta = (dec+90)*pi/180
	phi = ra*pi/180
	px = hp.ang2pix(8,theta,phi)

	return px

######

def dosamples(bn):
	"""
	dosamples takes in a bin and estimates the covariance matrix entry associated with that bin. 
	It returns a dictionary with R_a, the correlation function, and the 2-, 3-, and 4-point 
	contributions to the covariance matrix.
	"""
	from numpy.random import seed
	seed()

	r_center = bn['rmin'] + 0.5*bn['dr']
	mu_center = bn['mumin'] + 0.5*bn['dmu']

	Ra  = 0
	ctx = 0
	ct2 = 0
	ct3 = zeros(nbins)
	ct4 = zeros(nbins)

	rij,p = flat_draw(NSamp,bn)
	r4n2 , dp , ri_vec , rj_vec ,wi,wj= mysurvey.maskpair(rij)
	p *= dp
	ri_px = quick_px(ri_vec)
	rj_px = quick_px(rj_vec)
	rij_jack = (ri_px==rj_px).astype(int)

	rij_mag,rij_mu = cleanup_l(ri_vec-rj_vec,0.5*(ri_vec+rj_vec))

#	Ra  = mean( r4n2 / p )
#	ct2 = mean( r4n2 * wi*wj * (1+mysurvey.xi(rij_mag,rij_mu)) / p )
#	ctx = mean( r4n2 * mysurvey.xi(rij_mag,rij_mu) / p )

	Ra  = mean( rij_jack * r4n2 / p )
	ct2 = mean( rij_jack * r4n2 * wi*wj * (1+mysurvey.xi(rij_mag,rij_mu)) / p )
	ctx = mean( rij_jack * r4n2 * mysurvey.xi(rij_mag,rij_mu) / p )

	for rpt in range(repeats):
		rjk,p_jk = r2xi_draw(NSamp,mysurvey.r2xi)
		ril,p_il = r2xi_draw(NSamp,mysurvey.r2xi)

		rk_vec = rj_vec - rjk
		rl_vec = ri_vec - ril
		rk = anorm(rk_vec)
		rl = anorm(rl_vec)
		rk_px = quick_px(rk_vec)
		rl_px = quick_px(rl_vec)
		rkl_jack = (rk_px==rl_px).astype(int)
		rik_jack = (ri_px==rk_px).astype(int)
		rjl_jack = (rj_px==rl_px).astype(int)

		rkl_mag,rkl_mu = cleanup_l(rk_vec-rl_vec,0.5*(rk_vec+rl_vec))
		rjk_mag,rjk_mu = cleanup_l(rj_vec-rk_vec,0.5*(rj_vec+rk_vec))
		ril_mag,ril_mu = cleanup_l(ri_vec-rl_vec,0.5*(ri_vec+rl_vec))
		rik_mag,rik_mu = cleanup_l(ri_vec-rk_vec,0.5*(ri_vec+rk_vec))
		rjl_mag,rjl_mu = cleanup_l(rj_vec-rl_vec,0.5*(rj_vec+rl_vec))
			
		m = where( (r4n2 != 0) & (((rkl_mag > min(rmins)) & (rkl_mag<max(rmins)+4)) | ((rik_mag>min(rmins)) & (rik_mag<max(rmins)+4)) | ((rjl_mag>min(rmins)) & (rjl_mag<max(rmins)+4)) ) )
		NHits = len(m[0])

		#Evaulating xi is also slow, so we minimize the number of times we do it.
		xijk = zeros(shape(rjk_mag))
		xiil = zeros(shape(ril_mag))
		xijk[m] = mysurvey.xi(rjk_mag[m],rjk_mu[m])
		xiil[m] = mysurvey.xi(ril_mag[m],ril_mu[m])
		
		
		nwk = zeros(len(rk))
		nwl = zeros(len(rl))
		nwall = mysurvey.nw( concatenate((rk_vec[m],rl_vec[m])) )
		nwk[m] = nwall[:NHits]
		nwl[m] = nwall[NHits:]

		for rj in range(len(rmins)):
			for muj in range(len(mumins)):
				j = muj + 10*rj

				#4-point term
				f = where( (rkl_mag>rmins[rj]) & (rkl_mag<rmins[rj]+4) & 
						   (abs(rkl_mu)>mumins[muj]) & (abs(rkl_mu)<mumins[muj]+0.1) )

				#ct4[j] += sum( ((r4n2 * rjk_mag**2 * xijk * nwk * ril_mag**2 * xiil * nwl)/(p*p_jk*p_il))[f] )/NSamp
				ct4[j] += sum( ((rij_jack * rkl_jack * r4n2 * rjk_mag**2 * xijk * nwk * ril_mag**2 * xiil * nwl)/(p*p_jk*p_il))[f] )/NSamp

				#First 3-point term
				f = where( (rik_mag>rmins[rj]) & (rik_mag<rmins[rj]+4) & 
						   (abs(rik_mu)>mumins[muj]) & (abs(rik_mu)<mumins[muj]+0.1) )

				#ct3[j] += sum( ((r4n2 * rjk_mag**2 * xijk * nwk * wi)/(p*p_jk))[f] )/NSamp
				ct3[j] += sum( ((rij_jack * rik_jack * r4n2 * rjk_mag**2 * xijk * nwk * wi)/(p*p_jk))[f] )/NSamp

				#Second 3-point term
				f = where( (rjl_mag>rmins[rj]) & (rjl_mag<rmins[rj]+4) & 
						   (abs(rjl_mu)>mumins[muj]) & (abs(rjl_mu)<mumins[muj]+0.1) )

				#ct3[j] += sum( ((r4n2 * ril_mag**2 * xiil * nwl * wj)/(p*p_il))[f] )/NSamp
				ct3[j] += sum( ((rij_jack * rjl_jack * r4n2 * ril_mag**2 * xiil * nwl * wj)/(p*p_il))[f] )/NSamp
				
	ct4 /= repeats
	ct3 /= repeats
	
	outd = {}
	outd['Ra']  = Ra
	outd['ctx'] = ctx
	outd['ct2'] = ct2
	outd['ct3'] = ct3
	outd['ct4'] = ct4
	outd['r_center'] = r_center
	outd['mu_center'] = mu_center
	
	return outd

mo = []
big_Ra  = zeros((nloops,nbins))
big_ctx = zeros((nloops,nbins))
big_ct2 = zeros((nloops,nbins))
big_ct3 = zeros((nloops,nbins,nbins))
big_ct4 = zeros((nloops,nbins,nbins))
for loopct in range(nloops):
	if __name__ == '__main__':
		pool = Pool(processes=ncores)
		tstart = time()
		myout = pool.map(dosamples, bnl)
		tfin = time()
	print "Run "+str(loopct)+" done, took "+str(tfin-tstart)
	mo.append(myout)

	Ra  = zeros(nbins)
	ctx = zeros(nbins)
	ct2 = zeros(nbins)
	ct3 = zeros((nbins,nbins))
	ct4 = zeros((nbins,nbins))
	r_center = zeros(nbins)
	mu_center = zeros(nbins)

	for i in range(nbins):
		Ra[i]  = myout[i]['Ra']
		ctx[i] = myout[i]['ctx']
		ct2[i] = myout[i]['ct2']
		ct3[i] = myout[i]['ct3']
		ct4[i] = myout[i]['ct4']
		r_center[i] = myout[i]['r_center']
		mu_center[i] = myout[i]['mu_center']
	
	cx = ctx / Ra
	c2 = diag(ct2) / outer(Ra,Ra)
	c3 = 0.5*( ct3 + transpose(ct3) ) / outer(Ra,Ra)
	c4 = 0.5*( ct4 + transpose(ct4) ) / outer(Ra,Ra)
				
	c_out = {}
	c_out['descriptive_keys'] = ['descriptive_keys','rmins','mumins','NSamp','repeats','RASCAL_version']

	c_out['Ra'] = Ra	
	c_out['c2'] = c2
	c_out['c3'] = c3
	c_out['c4'] = c4
	c_out['cx'] = cx
	c_out['r_center'] = r_center
	c_out['mu_center'] = mu_center
	
	c_out['rmins']    = rmins
	c_out['mumins']   = mumins
	c_out['NSamp']    = NSamp
	c_out['NSamp']    = NSamp
	c_out['repeats']  = repeats
	c_out['RASCAL_version'] = RASCAL_version

	pickle.dump( c_out, open( fileprefix+'_'+str(loopct)+'.pkl', "wb" ),pickle.HIGHEST_PROTOCOL)
	
	big_Ra[loopct]  = Ra
	big_ctx[loopct] = ctx
	big_ct2[loopct] = ct2
	big_ct3[loopct] = ct3
	big_ct4[loopct] = ct4


Ra  = mean(big_Ra,axis=0)
ctx = mean(big_ctx,axis=0)
ct2 = mean(big_ct2,axis=0)
ct3 = mean(big_ct3,axis=0)
ct4 = mean(big_ct4,axis=0)

cx = ctx/Ra
c2 = diag(ct2)/outer(Ra,Ra)
c3 = 0.5*(ct3+transpose(ct3))/outer(Ra,Ra)
c4 = 0.5*(ct4+transpose(ct4))/outer(Ra,Ra)

c_out = {}
c_out['descriptive_keys'] = ['descriptive_keys','NSamp','repeats','RASCAL_version','corr_opts','survey_opts']

c_out['Ra'] = Ra	
c_out['c2'] = c2
c_out['c3'] = c3
c_out['c4'] = c4
c_out['cx'] = cx
c_out['r_center'] = r_center
c_out['mu_center'] = mu_center

c_out['rmins']    = rmins
c_out['mumins']   = mumins
c_out['NSamp']    = NSamp
c_out['repeats']  = repeats
c_out['RASCAL_version'] = RASCAL_version
c_out['corr_opts'] = corr_opts
c_out['survey_opts'] = survey_opts

pickle.dump( c_out, open( fileprefix+'_'+'mean'+'.pkl', "wb" ),pickle.HIGHEST_PROTOCOL)
