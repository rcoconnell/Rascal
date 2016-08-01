"""
This defines the 'Survey' class for Rascal.
"""

from Rascal_Sampler import anorm,aortho,pinv_pair
from numpy import array,transpose,linspace,sqrt,cos,sin,pi,concatenate
from numpy.random import rand

#Status: init runs, need to test that r_vecdraw matches original, then see if maskpair matches as well

class Survey(object):
	"""
	Instances of this class describe the geometrical properties of a survey, and the dynamical properties
	of the objects studied in the survey. Options, including the options that specify which libraries
	will provide the correlation function and geometry, are passed through from Rascal.

	Instances then provide a number of methods used by Rascal. At present, these are xi, r2xi, nw, and maskpair.
	"""
	def __init__(self,corr_opts,sgeo_opts):
		corr = __import__(corr_opts['corrlib'])
		sgeo = __import__(sgeo_opts['surveylib'])
		self.xi,self.r2xi = corr.init(corr_opts) #In the long term, r2xi should be computed in this class
		self.nw,self.w,self.dz  = sgeo.init(sgeo_opts)
		self.zmin   = float(sgeo_opts['zmin'])
		self.zmax   = float(sgeo_opts['zmax'])
		self.RAmin  = float(sgeo_opts['ramin'])
		self.RAmax  = float(sgeo_opts['ramax'])
		self.decmin = float(sgeo_opts['decmin']) + 90
		self.decmax = float(sgeo_opts['decmax']) + 90

		# Set up n2inv, n2pdf
		los_RA  = (self.RAmax + self.RAmin)/2
		los_cth = (cos(self.decmax) + cos(self.decmin*pi/180))/2
		zbdy = linspace(self.zmin,self.zmax,55)
		zcen = (zbdy + (zbdy[1]-zbdy[0])/2)[:-1]
		dcen = self.dz(zcen)
		dbdy = self.dz(zbdy)
		r_vec = array([ dcen*los_cth, dcen*sqrt(1-los_cth**2)*cos(los_RA*pi/180), dcen*sqrt(1-los_cth**2)*sin(los_RA*pi/180) ])
		r_vec = transpose(r_vec)
		nwcen = self.nw(r_vec)
		self.n2inv,self.n2pdf = pinv_pair( nwcen**2 * dcen**2 , dbdy )

		#later on, set up r2xi here instead of loading it from corr
	
	def r_vecdraw(self,r):
		"""
		r_vecdraw takes in an array of reals (line-of-sight distances to points in the survey)
		and converts them into 3-component vectors with angular coordinates inside the bounding box
		described by RAmin, RAmax, decmin, and decmax. It returns the 3-component vectors as well as
		the probability density associated with those angular coordinates.
		"""
		ni = len(r)
		phi_draw = self.RAmin + (self.RAmax-self.RAmin)*rand(ni)
		cth_draw = cos(self.decmin*pi/180) + ( cos(self.decmax*pi/180) - cos(self.decmin*pi/180) ) * rand(ni)
		r_vec = array([ r*cth_draw,r*sqrt(1-cth_draw**2)*cos(phi_draw*pi/180),r*sqrt(1-cth_draw**2)*sin(phi_draw*pi/180) ])
		r_vec = transpose(r_vec)
		return r_vec,1/( (self.RAmax-self.RAmin)*(pi/180) * ( cos(self.decmin*pi/180) - cos(self.decmax*pi/180) ) )
	
	def maskpair(self,rij):
		"""
		maskpair takes in an array of vectors r_ij and produces pairs of vectors within the survey volume,
		r_i and r_j, whose difference is r_ij. It returns several quantities. First r^2 * r_ij^2 * n^2 * w^2, where
		n^2 and w^2 are the number density and weights evaluated at r_i and r_j, then the probability of choosing 
		each pair of vectors r_i and r_j, the vectors r_i and r_j themselves, and the weight function evaluated at
		r_i and r_j.
		"""
		rij_mag = anorm(rij)
		ns = len(rij_mag)
		los_mag = self.n2inv(rand(ns))
		los,rho = self.r_vecdraw(los_mag)
		p = self.n2pdf(los_mag)*rho
		lhat,ltran1,ltran2 = aortho(los)
	
		#rij is set up with the l.o.s. in the \hat{x} direction. Need to put it along lhat instead.
		rij_rot = rij[:,0]*transpose(lhat) + rij[:,1]*transpose(ltran1) + rij[:,2]*transpose(ltran2)
		rij_rot = transpose(rij_rot)

		ri = los + 0.5*rij_rot
		rj = los - 0.5*rij_rot
		ri_mag = anorm(ri)
		rj_mag = anorm(rj)
		
		nwij = self.nw(concatenate((ri,rj)))
		nwi  = nwij[:ns]
		nwj  = nwij[ns:2*ns]
		
		wij = self.w(concatenate((ri,rj)))
		wi  = wij[:ns]
		wj  = wij[ns:2*ns]

		return los_mag**2 * rij_mag**2 * nwi*nwj , p , ri,rj,wi,wj