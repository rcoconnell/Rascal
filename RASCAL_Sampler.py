"""
The Sampler needs more docstrings, but I don't remember what each of these do.
"""
from numpy import max,min,array,where,reshape,shape,transpose,sqrt,sum,linspace,meshgrid,interp,trapz,abs,zeros,ones,pi,exp,cos,sin,sign,arange,concatenate,sort,cumsum,arctan2,cross,arctan2
from numpy.random import rand,randint
from scipy.interpolate import RectBivariateSpline,interp1d
from scipy.integrate import cumtrapz
#import 

def anorm(v):
	return sqrt(adot(v,v))

def adot(v1,v2):
	return sum(v1*v2,axis=1)

def aortho(v):
	vn = anorm(v)
	u = v / reshape(vn,(len(v),1))
	
	test1 = zeros(( len(v) , 3 ))
	test1[:,0] = 1
	test1 -= u * reshape( adot(u,test1) , (len(v),1) )
	test1 /= reshape(anorm(test1),(len(v),1))
	
	test2 = cross(u,test1)
	
	return u,test1,test2

# def cleanup(v):
# # We'll often set up a vector using the x-hat direction ([0]) as the line-of-sight. 
# # This extracts r and mu in that case.
# 	vn = anorm(v)
# 	#vmu = abs(v[:,0]/vn)
# 	vmu = v[:,0]/vn
# 	return vn,vmu

def cleanup_l(v,los):
	vn = anorm(v)
	lmag = anorm(los)
	lhat,ltran1,ltran2 = aortho(los)
	vmu = adot(v,lhat)/vn
	return vn,vmu

def bin(rmin,mumin,dr=4,dmu=0.1):
	b = {}
	b['dr']     = dr
	b['dmu']    = dmu
	b['rmin']  = rmin
	b['rmax']  = rmin + dr
	b['mumin'] = mumin
	b['mumax'] = mumin + dmu
	return b

def make_pdfv(cdfv):
	pdfv = zeros(len(cdfv)-1)
	pdfv = cdfv[:-1] + 0.5 * (cdfv[1:] - cdfv[:-1])
	return pdfv

def gridder(pdfv,cdfv):
	v_double = zeros(len(pdfv)+2)
	v_double[1:-1] = pdfv
	v_double[0] = cdfv[0]
	v_double[-1] = cdfv[-1]
	ind_double = zeros(len(v_double))
	ind_double[1:-1] = arange(len(pdfv))
	ind_double[-1] = ind_double[-2]
	return interp1d(v_double,ind_double,kind='nearest')	

def pinv_pair(pdf,v_cdf):
	v_double = concatenate((v_cdf,v_cdf[1:-1]))
	v_double = sort(v_double)
	pdf_double = zeros(len(v_double))
	for i in range(len(pdf)):
		pdf_double[2*i] = pdf[i]
		pdf_double[2*i+1] = pdf[i]
	
	pdfi = interp1d(v_double, pdf_double,kind='nearest')
		
	cdf = zeros(len(v_cdf))
#	for i in range(1,len(v_cdf)):
#		cdf[i] = quad(pdfi,v_cdf[0],v_cdf[i])[0]

	delta = v_cdf[1:]-v_cdf[:-1]
	cdf[1:] = cumsum(delta*pdf)
	
	pdf_double /= cdf[-1]
	cdf /= cdf[-1]

	pdfi = interp1d(v_double, pdf_double,kind='nearest')
	invi = interp1d(cdf,v_cdf,kind='linear')

	return invi,pdfi

def flat_draw(n,bn,sw=1):
	dr   = bn['dr']
	dmu  = bn['dmu']
	rij  = bn['rmin']  + rand(n)*dr
	muij = bn['mumin'] + rand(n)*dmu

	phiij = 2*pi*rand(n)

	# Necessary because bins are defined by |mu| not mu (for autocorrelation)
	musign = 2*randint(2,size=n)-1
	muij *= musign

	rijv = array([ rij*muij , rij*sqrt(1-muij**2)*cos(phiij) , rij*sqrt(1-muij**2)*sin(phiij) ])
	rijv = transpose(rijv)
	
	p = ones(n) / (2*dr*dmu*2*pi) #Leading 2 is from the sign on mu
	
	return rijv,p

def r2xi_draw(n,r2xi):
	r2xi_inv,r2xi_pdf = r2xi

	muij = -1 + 2*rand(n)
	phiij = 2*pi*rand(n)
	p = ones(n) / (2*2*pi)

	rij = r2xi_inv(rand(n))
	p *= r2xi_pdf(rij)
	
	rijv = array([ rij*muij , rij*sqrt(1-muij**2)*cos(phiij) , rij*sqrt(1-muij**2)*sin(phiij) ])
	rijv = transpose(rijv)

	return rijv,p