from numpy import max,min,array,where,reshape,shape,transpose,sqrt,sum,linspace,meshgrid,interp,trapz,abs,zeros,ones,pi,exp,cos,sin,sign,arange,concatenate,sort,cumsum,arctan2,cross,arctan2
from numpy.random import rand,randint
from scipy.interpolate import RectBivariateSpline,interp1d
from scipy.integrate import cumtrapz
#import 

def uniformbnd(vmin,vmax):
	y = rand(len(vmin))
	return vmin + y*(vmax-vmin)

def anorm(v):
	return sqrt(adot(v,v))

def adot(v1,v2):
	return sum(v1*v2,axis=1)

# def aortho(v):
# 	vn = anorm(v)
# 	u = v / reshape(vn,(len(v),1))
# 	
# 	test1 = 2*(rand(len(v),3)-1)
# 	test1 -= u * reshape( adot(u,test1) , (len(v),1) )
# 	test1 /= reshape(anorm(test1),(len(v),1))
# 	
# 	#test2 = numpy.cross(u,test1)
# 	
# 	return u,test1 #,test2

def aortho(v):
	vn = anorm(v)
	u = v / reshape(vn,(len(v),1))
	
	test1 = zeros(( len(v) , 3 ))
	test1[:,0] = 1
	test1 -= u * reshape( adot(u,test1) , (len(v),1) )
	test1 /= reshape(anorm(test1),(len(v),1))
	
	test2 = cross(u,test1)
	
	return u,test1,test2

def get_rmuphi_100(v):
	vn = anorm(v)
	vmu = v[:,0]/vn
	vphi = arctan2(v[:,2],v[:,1])
	return vn,vmu,vphi

def get_rmuphi_los(v,los):
	vn = anorm(v)
	lmag = anorm(los)
	lhat,ltran1,ltran2 = aortho(los)
	vtran1 = adot(v,ltran1)
	vtran2 = adot(v,ltran2)
	vmu = adot(v,lhat)/vn
	vphi = arctan2(vtran2,vtran1)
	return vn,vmu,vphi
	


def cleanup(v):
# We'll often set up a vector using the x-hat direction ([0]) as the line-of-sight. 
# This extracts r and mu in that case.
	vn = anorm(v)
	#vmu = abs(v[:,0]/vn)
	vmu = v[:,0]/vn
	return vn,vmu

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

def zbin(rmin,mumin,dmin,dmax,dr=4,dmu=0.1):
	b = {}
	b['dr']     = dr
	b['dmu']    = dmu
	b['rmin']  = rmin
	b['rmax']  = rmin + dr
	b['mumin'] = mumin
	b['mumax'] = mumin + dmu
	b['dmin']  = dmin
	b['dmax']  = dmax
	return b

def bins(r1min,r2min,mu1min,mu2min,dr=4,dmu=0.1):
	b = {}
	b['dr']     = dr
	b['dmu']    = dmu
	b['r1min']  = r1min
	b['r1max']  = r1min + dr
	b['mu1min'] = mu1min
	b['mu1max'] = mu1min + dmu
	b['r2min']  = r2min
	b['r2max']  = r2min + dr
	b['mu2min'] = mu2min
	b['mu2max'] = mu2min + dmu
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

def flat_draw(n,bn,sw):
	dr   = bn['dr']
	dmu  = bn['dmu']
	if sw == 1:
		rij  = bn['r1min']  + rand(n)*dr
		muij = bn['mu1min'] + rand(n)*dmu
	elif sw == 2:
		rij  = bn['r2min']  + rand(n)*dr
		muij = bn['mu2min'] + rand(n)*dmu
	else:
		return 0

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

def short_draw(n,rmax=8):
	rij = rmax*rand(n)
	p = ones(n)/rmax

	muij = -1 + 2*rand(n)
	phiij = 2*pi*rand(n)
	p *= ones(n) / (2*2*pi)

	rijv = array([ rij*muij , rij*sqrt(1-muij**2)*cos(phiij) , rij*sqrt(1-muij**2)*sin(phiij) ])
	rijv = transpose(rijv)

	return rijv,p

def quad_sampler(rpdf,rdmax=499,nb=496):
	#Establish the grids we'll use
	muv_cdf = 1-2*exp(-4*linspace(0,1,11))
	muv_cdf[-1] = 1
	muv_pdf = make_pdfv(muv_cdf)

	rdv_cdf = zeros(nb+16)
	rdv_cdf[:16] = linspace(0,3.75,16)
	rdv_cdf[16:] = linspace(4,rdmax,nb)
	rdv_pdf = make_pdfv(rdv_cdf)
	
	r1v_cdf = array(rdv_cdf)
	r1v_pdf = array(rdv_pdf)

	unv = linspace(0,1,80)
	
	#Set up the gridders
	mu_gridder = gridder(muv_pdf,muv_cdf)
	rdv_gridder = gridder(rdv_pdf,rdv_cdf)
	r1v_gridder = gridder(r1v_pdf,r1v_cdf)
	
# 	drd = rdv[1]-rdv[0]
# 	dmu = muv[1]-muv[0]
# 	dr1 = r1v[1]-r1v[0]

 	nd = len(rdv_pdf)
# 	nmu = len(muv)
# 	nr1 = len(r1v)
# 	nu  = len(unv)

	rdg,mug,r1g = meshgrid(rdv_pdf,muv_pdf,r1v_pdf,indexing='ij')

#	drdv = (rdv[1]-rdv[0])/2
	r2v = sqrt(rdg**2 + r1g**2 - 2*rdg*r1g*mug)
#	r2vp = sqrt((rdg+drdv)**2 + r1g**2 - 2*(rdg+drdv)*r1g*mug)
#	r2vm = sqrt((rdg-drdv)**2 + r1g**2 - 2*(rdg-drdv)*r1g*mug)

	pdfg = (r1g**2) * rpdf(r1g) * rpdf(r2v)

	#This isn't enough to keep the sampler from producing r1 and r2 > rmax. Stupid splines.
#	m = where( (r2v>rmax) | (r2vm>rmax) | (r2vp>rmax) )
#	pdfg[m] = 0

	#We won't have consistency across different axes, I don't think. But we won't actually 
	#use them together, so all this will do is make the final pdf for points a little 
	#different than we wanted, while each pdf/inverter pair will still be correctly matched.

	pdfnorm = trapz( trapz( pdfg, muv_pdf,axis=1 ), r1v_pdf,axis=1 )

	pdfg /= reshape( pdfnorm , (nd,1,1))

# 	pdf_smooth = []
# 	for i in range(nd):
# 		pdf_smooth.append( RectBivariateSpline(muv,r1v,pdfg[i],kx=1,ky=1) )
	
	mupdf = trapz( pdfg,r1v_pdf,axis=2 )
	mu_inv = []
	mu_p = []
	for i in range(len(rdv_pdf)):
		mu_inv_t,mu_p_t = pinv_pair(mupdf[i],muv_cdf)
		mu_inv.append(mu_inv_t)
		mu_p.append(mu_p_t)
	
	r1_inv = []
	r1_p = []
	for i in range(len(rdv_pdf)):
		r1_inv.append([])
		r1_p.append([])
		for j in range(len(muv_pdf)):
			r1_inv_t,r1_p_t = pinv_pair(pdfg[i][j],r1v_cdf)
			r1_inv[-1].append( r1_inv_t )
			r1_p[-1].append( r1_p_t )
	
	return (rdv_gridder,mu_gridder),(mu_inv,mu_p),(r1_inv,r1_p),(rdv_pdf,pdfnorm)
		
#	return (mucdf,muv),r1inv_smooth,(rdv,pdfnorm)


def drawquad_diff(rdswitch,rijv,bn,sampler,rpdf):

	#Unpack the sampler
	rd_gridder,mu_gridder = sampler[0]
	mu_inv,mu_p = sampler[1]
	r1_inv,r1_p = sampler[2]
	rdv_pdf,pdfnorm = sampler[3]
	
	n = len(rijv)
	
	#We'll use these grids in determining the pdf for delta phi
	drv = array([ 0 , 0.05 , 0.1 , 0.5 , 0.9 , 0.95 ]) * bn['dr']
	dmuv = array([ 0 , 0.05 , 0.1 , 0.5 , 0.9 , 0.95 ]) * bn['dmu']
	r1v_cdf = bn['r1min'] + drv
	r2v_cdf = bn['r2min'] + drv
	mu1v_cdf = bn['mu1min'] + dmuv
	mu2v_cdf = bn['mu2min'] + dmuv
	r1v_pdf = make_pdfv(r1v_cdf)
	r2v_pdf = make_pdfv(r2v_cdf)
	mu1v_pdf = make_pdfv(mu1v_cdf)
	mu2v_pdf = make_pdfv(mu2v_cdf)
	tv = linspace(-1,1,11)
	phiv_cdf = (pi/2)*(1 + abs(tv)**(1./3) * sign(tv) )
	phiv_pdf = make_pdfv(phiv_cdf)
	unit = linspace(0,1,5)
	p_kl = ones(n)
	p_jk = ones(n)
	
	#Evaluate r_delta on the grid, compare to the target pdf for r_delta to get a pdf on the grid
	r1g,r2g,mu1g,mu2g,phig = meshgrid(r1v_pdf,r2v_pdf,mu1v_pdf,mu2v_pdf,phiv_pdf,indexing='ij')
	if rdswitch == 1:
		rdg = sqrt( abs(r1g**2 + r2g**2 - 2*r1g*r2g*( mu1g*mu2g + sqrt(1-mu1g**2)*sqrt(1-mu2g**2)*cos(phig) )))
	elif rdswitch == 2:
		rdg = sqrt( abs(r1g**2 + r2g**2 + 2*r1g*r2g*( mu1g*mu2g + sqrt(1-mu1g**2)*sqrt(1-mu2g**2)*cos(phig) )))
	else:
		print "bad rdswitch"
		return -1
	pdfg = interp(rdg,rdv_pdf,pdfnorm)
	
	#Integrate to get a pdf for delta phi, generate the delta phi draws
	phipdf = trapz( trapz( trapz( trapz( pdfg , mu2v_pdf,axis=-2) , mu1v_pdf,axis=-2) , r2v_pdf,axis=-2) , r1v_pdf,axis=-2)
#	phipdf /= trapz(phipdf,phiv)
#	phicdf = zeros(shape(phipdf))
#	phicdf[1:] = cumtrapz(phipdf,phiv)
#	deltaphi = interp(rand(n),phicdf,phiv)
#	p *= interp(deltaphi,phiv,phipdf)
	phi_inv,phi_p = pinv_pair(phipdf,phiv_cdf)
	deltaphi = phi_inv(rand(n))
	p_kl *= phi_p(deltaphi)
	deltaphi *= sign( 2*(rand(n)-0.5) )
	p_kl /= 2

	#Draw the remaining "binned" vector, rkl, and decompose the input vector rij
	dr   = bn['dr']
	dmu  = bn['dmu']
	rij  = anorm(rijv)
	muij = rijv[:,0]/rij
	phiij = arctan2(rijv[:,2],rijv[:,1])
	f = where(phiij < 0)
#	phiij[f] += 2*pi
	rkl  = bn['r2min']  + rand(n)*dr
	mukl = bn['mu2min'] + rand(n)*dmu
	phikl = phiij + deltaphi
	p_kl /= (dr*dmu)

	#With the binned values, compute r_ij, r_kl, r_delta
	rklv = array([ rkl*mukl , rkl*sqrt(1-mukl**2)*cos(phikl) , rkl*sqrt(1-mukl**2)*sin(phikl) ])
	rklv = transpose(rklv)
	if rdswitch == 1:
		rdelv = rklv - rijv
	elif rdswitch == 2:
		rdelv = rklv + rijv
	else:
		print "bad rdswitch -- have you been tinkering with the code?"
		return -2
	rd = anorm(rdelv)

	#Start work on r_jk	
	mudraw = zeros(n)
	r1draw = zeros(n)

	rdi = rd_gridder(rd).astype(int)
	for ri in range(min(rdi),max(rdi)+1):
		f = where(rdi == ri)
		mudraw[f] = mu_inv[ri](rand(len(f[0])))
		p_jk[f] *= mu_p[ri](mudraw[f])

	mui = mu_gridder(mudraw).astype(int)
	for ri in range(min(rdi),max(rdi)+1):
		for mi in range(min(mui),max(mui)+1):
			f = where( (rdi==ri) & (mui==mi) )
			r1draw[f] = r1_inv[ri][mi](rand(len(f[0])))
			p_jk[f] *= r1_p[ri][mi](r1draw[f])
	
	p_jk *= 1/(2*pi)
	
	phijk = 2*pi*rand(len(r1draw))
	rjkv = r1draw * array([ mudraw , sqrt(1-mudraw**2)*cos(phijk) , sqrt(1-mudraw**2)*sin(phijk) ])
	rjkv = transpose(rjkv)
	rjkv = -rjkv
	
	rliv = -(rijv + rjkv + rklv)
	
	return rjkv,rklv,p_jk*p_kl,p_kl




def drawquad_nodiff(rdswitch,n,bn,sampler,rpdf):
	mucdf,muv = sampler[0]
	r1inv = sampler[1]
	rdv,pdfnorm = sampler[2]

	dr   = bn['dr']
	dmu  = bn['dmu']
	p = ones(n)

	if rdswitch == 1:
		binv = linspace( bn['r1min'],bn['r1max'],10)
	elif rdswitch == 2:
		binv = linspace( bn['r2min'],bn['r2max'],10)
	else:
		print "bad rdswitch"
		return -1
	binpdf = interp( binv,rdv,pdfnorm )
	binpdf /= trapz(binpdf,binv)
	bincdf = zeros(shape(binpdf))
	bincdf[1:] = cumtrapz(binpdf,binv)
	
	if rdswitch == 1:
		rij = interp(rand(n),bincdf,binv)
		p *= interp(rij,binv,binpdf)
		rkl  = bn['r2min']  + rand(n)*dr
		p /= dr
	elif rdswitch == 2:
		rkl = interp(rand(n),bincdf,binv)
		p *= interp(rkl,binv,binpdf)
		rij  = bn['r1min']  + rand(n)*dr
		p /= dr
	else:
		print "bad rdswitch -- have you been tinkering with the code?"
		return -2

	muij = bn['mu1min'] + rand(n)*dmu
	phiij = 2*pi*rand(n)
	mukl = bn['mu2min'] + rand(n)*dmu
	phikl = 2*pi*rand(n)
	p /= ( 2*pi*dmu )**2

	rijv = array([ rij*muij , rij*sqrt(1-muij**2)*cos(phiij) , rij*sqrt(1-muij**2)*sin(phiij) ])
	rklv = array([ rkl*mukl , rkl*sqrt(1-mukl**2)*cos(phikl) , rkl*sqrt(1-mukl**2)*sin(phikl) ])

	if rdswitch == 1:
		rdelv = rijv
	elif rdswitch == 2:
		rdelv = rklv
	else:
		print "bad rdswitch -- what have you been up to?"
		return -3

	rijv = transpose(rijv)
	rklv = transpose(rklv)
	rdelv = transpose(rdelv)
	rd = anorm(rdelv)
		
	mudraw = zeros(n)
	r1draw = zeros(n)
	
	q = abs(reshape(rd,(n,1)) - reshape(rdv,(1,len(rdv))))
	rdi = where(q == reshape(min(q,axis=1),(n,1)))[1]
	
	for ri in range(min(rdi),max(rdi)+1):
		m = where( rdi == ri )
		if len(m[0])==0: continue
		mudraw_t = interp(rand(len(m[0])),mucdf[ri],muv)
		r1draw_t = r1inv[ri].ev(mudraw_t,rand(len(m[0])))
		r2 = sqrt(rdv[ri]**2 + r1draw_t**2 - 2*rdv[ri]*r1draw_t*mudraw_t)
		p[m] *= (r1draw_t**2) * rpdf(r1draw_t) * rpdf(r2) / pdfnorm[ri]
		mudraw[m] = mudraw_t
		r1draw[m] = r1draw_t

	xhat,yhat = aortho(rdelv)
	p *= 1/(2*pi)

	rjkv  = reshape( r1draw * mudraw , (n,1) )*xhat
	rjkv += reshape( r1draw * sqrt(1-mudraw**2) , (n,1) )*yhat
	rjkv = -rjkv
	
	rliv = -(rijv + rjkv + rklv)
	
	return rijv,rjkv,rklv,rliv,p

def drawtriple(n,bn,rpdf):
	#We'll use these grids in determining the pdf for delta phi
	drv = array([ 0 , 0.1 * bn['dr'] , 0.5 * bn['dr'] , 0.9 * bn['dr'] , bn['dr'] ])
	dmuv = array([ 0 , 0.1 * bn['dmu'] , 0.5 * bn['dmu'] , 0.9 * bn['dmu'] , bn['dmu'] ])
	r1v = bn['r1min'] + drv
	r2v = bn['r2min'] + drv
	mu1v = bn['mu1min'] + dmuv
	mu2v = bn['mu2min'] + dmuv
	tv = linspace(-1,1,11)
	phiv_cdf = (pi/2)*(1 + abs(tv)**(1./3) * sign(tv) )
	phiv_pdf = make_pdfv(phiv_cdf)
	unit = linspace(0,1,5)
	p = ones(n)
	
	#Evaluate r_delta on the grid, compare to the target pdf for r_delta to get a pdf on the grid
	r1g,r2g,mu1g,mu2g,phig = meshgrid(r1v,r2v,mu1v,mu2v,phiv_pdf,indexing='ij')
	rdg = sqrt( abs(r1g**2 + r2g**2 + 2*r1g*r2g*( mu1g*mu2g + sqrt(1-mu1g**2)*sqrt(1-mu2g**2)*cos(phig) )))
	
	f = where(rdg==0)
	g = where(rdg!=0)
	rdg[f] = 0.001 * min(rdg[g])
	
	pdfg = rpdf(rdg)
	
	#Integrate to get a pdf for delta phi, generate the delta phi draws
 	phipdf = trapz( trapz( trapz( trapz( pdfg , mu2v,axis=-2) , mu1v,axis=-2) , r2v,axis=-2) , r1v,axis=-2)
	phi_inv,phi_p = pinv_pair(phipdf,phiv_cdf)	
	deltaphi = phi_inv(rand(n))
	p *= phi_p(deltaphi)
	deltaphi *= sign( 2*(rand(n)-0.5) )
	p /= 2

	#Draw the remaining "binned" variables
	dr   = bn['dr']
	dmu  = bn['dmu']
	rij  = bn['r1min']  + rand(n)*dr
	muij = bn['mu1min'] + rand(n)*dmu
	phiij = 2*pi*rand(n)
	rjk  = bn['r2min']  + rand(n)*dr
	mujk = bn['mu2min'] + rand(n)*dmu
	phijk = phiij + deltaphi
	p /= 2*pi* (dr*dmu)**2

	#With the binned values, compute r_ij, r_kl, r_delta
	rijv = array([ rij*muij , rij*sqrt(1-muij**2)*cos(phiij) , rij*sqrt(1-muij**2)*sin(phiij) ])
	rjkv = array([ rjk*mujk , rjk*sqrt(1-mujk**2)*cos(phijk) , rjk*sqrt(1-mujk**2)*sin(phijk) ])
	rijv = transpose(rijv)
	rjkv = transpose(rjkv)
	rikv = rijv + rjkv
	
	return rijv,rjkv,rikv,p


def drawtriple_z1(rijv,sampler):
	#Unpack the sampler
	rd_gridder,mu_gridder = sampler[0]
	mu_inv,mu_p = sampler[1]
	r1_inv,r1_p = sampler[2]
	rdv_pdf,pdfnorm = sampler[3]
	
	n = len(rijv)
	p = ones(n)
	rdelv = array(rijv)
	rd = anorm(rdelv)

	#Start work on r_jk	
	mudraw = zeros(n)
	r1draw = zeros(n)

	rdi = rd_gridder(rd).astype(int)
	for ri in range(min(rdi),max(rdi)+1):
		f = where(rdi == ri)
		mudraw[f] = mu_inv[ri](rand(len(f[0])))
		p[f] *= mu_p[ri](mudraw[f])

	mui = mu_gridder(mudraw).astype(int)
	for ri in range(min(rdi),max(rdi)+1):
		for mi in range(min(mui),max(mui)+1):
			f = where( (rdi==ri) & (mui==mi) )
			r1draw[f] = r1_inv[ri][mi](rand(len(f[0])))
			p[f] *= r1_p[ri][mi](r1draw[f])
	
	xhat,yhat = aortho(rdelv)
	p *= 1/(2*pi)
	
	rjkv  = reshape( r1draw * mudraw , (n,1) )*xhat
	rjkv += reshape( r1draw * sqrt(1-mudraw**2) , (n,1) )*yhat
	rjkv = -rjkv
	
	return rjkv,p

def pkl_fixer(rdswitch,rijv,rklv,p_kl,bn,sampler):
	#Unpack the sampler
	rd_gridder,mu_gridder = sampler[0]
	mu_inv,mu_p = sampler[1]
	r1_inv,r1_p = sampler[2]
	rdv_pdf,pdfnorm = sampler[3]
	
	n = len(rijv)
	
	#We'll use these grids in determining the pdf for delta phi
	drv = array([ 0 , 0.05 , 0.1 , 0.5 , 0.9 , 0.95 ]) * bn['dr']
	dmuv = array([ 0 , 0.05 , 0.1 , 0.5 , 0.9 , 0.95 ]) * bn['dmu']
	r1v_cdf = bn['r1min'] + drv
	r2v_cdf = bn['r2min'] + drv
	mu1v_cdf = bn['mu1min'] + dmuv
	mu2v_cdf = bn['mu2min'] + dmuv
	r1v_pdf = make_pdfv(r1v_cdf)
	r2v_pdf = make_pdfv(r2v_cdf)
	mu1v_pdf = make_pdfv(mu1v_cdf)
	mu2v_pdf = make_pdfv(mu2v_cdf)
	tv = linspace(-1,1,11)
	phiv_cdf = (pi/2)*(1 + abs(tv)**(1./3) * sign(tv) )
	phiv_pdf = make_pdfv(phiv_cdf)
	unit = linspace(0,1,5)
	
	#Evaluate r_delta on the grid, compare to the target pdf for r_delta to get a pdf on the grid
	r1g,r2g,mu1g,mu2g,phig = meshgrid(r1v_pdf,r2v_pdf,mu1v_pdf,mu2v_pdf,phiv_pdf,indexing='ij')
	if rdswitch == 1:
		rdg = sqrt( abs(r1g**2 + r2g**2 - 2*r1g*r2g*( mu1g*mu2g + sqrt(1-mu1g**2)*sqrt(1-mu2g**2)*cos(phig) )))
	elif rdswitch == 2:
		rdg = sqrt( abs(r1g**2 + r2g**2 + 2*r1g*r2g*( mu1g*mu2g + sqrt(1-mu1g**2)*sqrt(1-mu2g**2)*cos(phig) )))
	else:
		print "bad rdswitch"
		return -1
	pdfg = interp(rdg,rdv_pdf,pdfnorm)
	
	#Integrate to get a pdf for delta phi, generate the delta phi draws
	phipdf = trapz( trapz( trapz( trapz( pdfg , mu2v_pdf,axis=-2) , mu1v_pdf,axis=-2) , r2v_pdf,axis=-2) , r1v_pdf,axis=-2)
	phi_inv,phi_p = pinv_pair(phipdf,phiv_cdf)
	phiij = arctan2(rijv[:,2],rijv[:,1])
	phikl = arctan2(rklv[:,2],rklv[:,1])
	deltaphi = abs(phikl - phiij) #Between 0 and 2pi
	f = where(deltaphi>pi)
	deltaphi[f] = 2*pi - deltaphi[f] #Now between 0 and pi

	dp_kl = phi_p(deltaphi) / 2 #Dividing by two accounts for the sign, i.e. (0,pi) -> (-pi,pi)
	return 0.5 * (p_kl + dp_kl)

def pjk_fixer(rijv,rjkv,p_jk,bn,sampler,rdmax = 499):
	#Unpack the sampler
	rd_gridder,mu_gridder = sampler[0]
	mu_inv,mu_p = sampler[1]
	r1_inv,r1_p = sampler[2]
	rdv_pdf,pdfnorm = sampler[3]
	
	n = len(rijv)
	dp_jk_all = zeros(n)
		
	rd = anorm(rijv)
	rv = anorm(rjkv)
	muv = adot(rijv,rjkv)/(rd*rv)
	muv *= -1
	
	ok = where( rd <= rdmax)
	rd = rd[ok]
	rv = rv[ok]
	muv = muv[ok]
	dp_jk_ok = ones(len(ok[0]))/(2*pi)
	
	rdi = rd_gridder(rd).astype(int)
	for ri in range(min(rdi),max(rdi)+1):
		f = where(rdi == ri)
		dp_jk_ok[f] *= mu_p[ri](muv[f])

	mui = mu_gridder(muv).astype(int)
	for ri in range(min(rdi),max(rdi)+1):
		for mi in range(min(mui),max(mui)+1):
			f = where( (rdi==ri) & (mui==mi) )
			dp_jk_ok[f] *= r1_p[ri][mi](rv[f])
	
	dp_jk_all[ok] = dp_jk_ok

	return 0.5 * (p_jk + dp_jk_all)

def drawquad_0(n,bn,cdfv,pdfv):
	dr   = bn['dr']
	dmu  = bn['dmu']
	rij  = bn['r1min']  + rand(n)*dr
	muij = bn['mu1min'] + rand(n)*dmu
	phiij = 2*pi*rand(n)
	rkl  = bn['r2min']  + rand(n)*dr
	mukl = bn['mu2min'] + rand(n)*dmu
	phikl = 2*pi*rand(n)
	p = ones(n)/( 2*pi*dr*dmu )**2
	
	rr = pdfv[0]
	r2pdf = rr**2 * pdfv[1]
	r2pdf /= trapz(r2pdf,rr)
	r2cdf = zeros(shape(r2pdf))
	r2cdf[1:] = cumtrapz(r2pdf,rr)
	
	rjk = interp(rand(n),r2cdf,rr)
	mujk = 2*(rand(n)-0.5)
	phijk = 2*pi*rand(n)
	p *= 1/(2*2*pi)
	p *= interp(rjk,rr,r2pdf)

	rijv = array([ rij*muij , rij*sqrt(1-muij**2)*cos(phiij) , rij*sqrt(1-muij**2)*sin(phiij) ])
	rklv = array([ rkl*mukl , rkl*sqrt(1-mukl**2)*cos(phikl) , rkl*sqrt(1-mukl**2)*sin(phikl) ])
	rjkv = array([ rjk*mujk , rjk*sqrt(1-mujk**2)*cos(phijk) , rjk*sqrt(1-mujk**2)*sin(phijk) ])

	rijv = transpose(rijv)
	rklv = transpose(rklv)
	rjkv = transpose(rjkv)

	rliv = -(rijv + rjkv + rklv)
	
	return rijv,rjkv,rklv,rliv,p

def phisample(n,bn,(rdv,pdfnorm)):
	drv = array([ 0 , 0.1 * bn['dr'] , 0.5 * bn['dr'] , 0.9 * bn['dr'] , bn['dr'] ])
	dmuv = array([ 0 , 0.1 * bn['dmu'] , 0.5 * bn['dmu'] , 0.9 * bn['dmu'] , bn['dmu'] ])
	r1v = bn['r1min'] + drv
	r2v = bn['r2min'] + drv
	mu1v = bn['mu1min'] + dmuv
	mu2v = bn['mu2min'] + dmuv
	tv = linspace(-1,1,11)
	phiv = (pi/2)*(1 + abs(tv)**(1./3) * sign(tv) )
	unit = linspace(0,1,5)
	p = ones(n)
	
	r1g,r2g,mu1g,mu2g,phig = meshgrid(r1v,r2v,mu1v,mu2v,phiv,indexing='ij')
	rdg = sqrt( abs( r1g**2 + r2g**2 - 2*r1g*r2g*( mu1g*mu2g + sqrt(1-mu1g**2)*sqrt(1-mu2g**2)*cos(phig) ) ))
	pdfg = interp(rdg,rdv,pdfnorm)
	
	phipdf = trapz( trapz( trapz( trapz( pdfg , mu2v,axis=-2) , mu1v,axis=-2) , r2v,axis=-2) , r1v,axis=-2)
	phipdf /= trapz(phipdf,phiv)
	phicdf = zeros(shape(phipdf))
	phicdf[1:] = cumtrapz(phipdf,phiv)
	deltaphi = interp(rand(n),phicdf,phiv)
	
	dr   = bn['dr']
	dmu  = bn['dmu']
	rij  = bn['r1min']  + rand(n)*dr
	muij = bn['mu1min'] + rand(n)*dmu
	phiij = 2*pi*rand(n)
	rkl  = bn['r2min']  + rand(n)*dr
	mukl = bn['mu2min'] + rand(n)*dmu
	phikl = phiij + deltaphi

	rijv = array([ rij*muij , rij*sqrt(1-muij**2)*cos(phiij) , rij*sqrt(1-muij**2)*sin(phiij) ])
	rklv = array([ rkl*mukl , rkl*sqrt(1-mukl**2)*cos(phikl) , rkl*sqrt(1-mukl**2)*sin(phikl) ])
	rijv = transpose(rijv)
	rklv = transpose(rklv)
	rdelv = rklv - rijv
	rd = anorm(rdelv)
	
	return (phiv,phicdf,phipdf)

