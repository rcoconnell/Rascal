from Rascal_Sampler import *
from numpy import loadtxt,max,min,interp,abs,meshgrid,zeros,ones,where,reshape,shape,array,mean,arange
from scipy.interpolate import RectBivariateSpline

def load_xi(xiload):
	global xi,r2xi,xifile

	xifile = xiload
	xirmu = loadtxt(xifile)
	nr = len(xirmu)
	nmu = len(xirmu[0])

	#We assume 1 Mpc/h bins, and evenly spaced mu bins from mu=0 to mu=1
	rr = 0.5+arange(nr)
	mumu = (0.5+arange(nmu))/(1.*nmu)

	rr0 = zeros(1001)
	rr0[1:] = 0.5+arange(1000)

	rmax = max(rr0)+1


	xirmu_mult = zeros((len(rr0),len(mumu)))
	xirmu_mult[1:nr+1] = xirmu
	for i in range(1,len(xirmu[0])):
		xirmu_mult[nr+1:,i] = xirmu_mult[nr,i]*(rr0[nr]/rr0[nr+1:])**4
		#print xirmu_mult[-1,i]

	xirmu_mult *= reshape(rr0**2,(1001,1))

	xi_RBS = RectBivariateSpline(rr0,mumu,xirmu_mult,bbox=[0,1000,0,1])
	def xi(r,mu):
		mu = abs(mu)
		xi_t = xi_RBS.ev(r,mu)
		xi_t /= r**2
		return xi_t

	#Build the vectors that drive the importance sampling

	ri = len(xirmu)
	pdf_mult = zeros(1001)
	pdf_mult[1:ri+1] = abs(mean(xirmu,axis=1))
	pdf_mult[ri+1:] = pdf_mult[ri]*(rr0[ri]/rr0[ri+1:])**4
	pdf_mult *= rr0**2

	rr_cdf = array(rr0)
	rr_cdf[1:] += ones(len(rr0)-1)
	r2xi = pinv_pair(pdf_mult[1:],rr_cdf)

def init(opts):
	xifile = opts['xifile']
	load_xi(xifile)
	return xi,r2xi