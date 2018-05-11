#!/home/wtluo/anaconda/bin/python2.7

import numpy as np 
import galsim 
import lsstetc
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pyfits as pf
import scipy.signal as signal

pi = np.pi
#------------------------------------------------------------------------
def centroids(psfimage):
  nx,ny  = psfimage.shape[0],psfimage.shape[1] 
  x,y    = np.mgrid[:nx,:ny]
  m00    = np.sum(psfimage)
  x0     = np.sum(x*psfimage)/m00
  y0     = np.sum(y*psfimage)/m00
  mxy  =np.sum((y-y0)*(x-x0)*psfimage)/m00
  mxx  =np.sum((x-x0)*(x-x0)*psfimage)/m00
  myy  =np.sum((y-y0)*(y-y0)*psfimage)/m00
  return [x0,y0]

#-------------------------------------------------------------------------
def gaussfilter(psfimage):
  x,y   = np.mgrid[0:3,0:3]
  xc,yc = 1.0,1.0
  w     = 1.2
  ker   = (1.0/2.0/pi/w/w)*np.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
  imcov = signal.convolve(psfimage,ker,mode='same')
  return imcov
#-----------------------------------------------------------------------
def mfpoly(psfimage):
  nx,ny  = psfimage.shape[0],psfimage.shape[1]
  xi,yi  = np.mgrid[0:nx,0:ny]
  idx    = psfimage==np.max(psfimage)
  xr,yr  = xi[idx][0],yi[idx][0]
  sli    = slice(xr-1,xr+2)
  slj    = slice(yr-1,yr+2)
  xp,yp  = np.mgrid[xr-1:xr+2,yr-1:yr+2]
  patchim= psfimage[sli,slj]

  zz     = patchim.reshape(9)
  xx     = xp.reshape(9)
  yy     = yp.reshape(9)
  xy     = xx*yy
  x2     = xx*xx
  y2     = yy*yy
  mA     = np.array([np.ones_like(zz),xx,yy,x2,xy,y2]).T
  cov    = np.linalg.inv(np.dot(mA.T,mA))
  a,b,c,d,e,f = np.dot(cov,np.dot(mA.T,zz))
  coeffs = np.array([[2.0*d,e],[e,2.0*f]])
  mult   = np.array([-b,-c])
  xc,yc  = np.dot(np.linalg.inv(coeffs),mult)
  res    = np.array([xc,yc])
  return res


#-------------------------------------------------------------------------
def moff(xc,yc,fwhm,beta,flx,nx,ny):
  alpha = fwhm/(2.0*np.sqrt(2.0**(1/beta)-1.0))/0.2
  x,y   = np.mgrid[0:nx,0:ny]
  rr    = np.sqrt((x-xc)**2+(y-yc)**2) 
  image = flx*(beta-1.0)*(1+(rr/alpha)**2)**(-beta)/pi/alpha/alpha
    
  return image

#----------------------------------------------------------------------
def lnlike(theta,psfimage,sigsky,flux,beta,fwhm,nx,ny):
    xc,yc  = theta 
    model  = moff(xc,yc,fwhm,beta,flux,nx,ny)
    invsig = 1.0/(sigsky*sigsky)
    
    return -0.5*(np.sum(model-psfimage)**2*invsig-np.log(invsig))
#---------------------------------------------------------------------

opt_defocus=0.       # wavelengths
opt_a1=-0.           # wavelengths
opt_a2=0.            # wavelengths
opt_c1=0.64            # wavelengths
opt_c2=-0.33           # wavelengths
opt_obscuration=0.    # linear scale size of secondary mirror obscuration
lam = 800.             # nm    NB: don't use lambda - that's a reserved word.
tel_diam = 10.         # meters
lam_over_diam = lam*1.e-9/tel_diam
#lam_over_diam = 206265.

gsparams = galsim.GSParams(folding_threshold=1.e-2, maxk_threshold=2.e-3,
    xvalue_accuracy=1.e-4,
    kvalue_accuracy=1.e-4,
    shoot_accuracy=1.e-4,
    minimum_fft_size=64)

optics = galsim.OpticalPSF(lam_over_diam,coma1 = -0., coma2 = -0.14,gsparams=gsparams)

coma = optics.drawImage()
plt.imshow(coma.array,interpolation='nearest')
plt.colorbar()
plt.savefig('anisotropy_2.eps')
plt.show()


narr     = 2000
pixscale = 0.2
etc = lsstetc.ETC('r',pixel_scale=pixscale)
scales = 2.0
opt_defocus=np.random.uniform(low=0.0,high=0.53/scales,size=narr)
opt_a1     =np.random.uniform(low=0.0,high=-0.29/scales,size=narr)
opt_a2     =np.random.uniform(low=0.0,high=0.12/scales,size=narr)
opt_c1     =np.random.uniform(low=0.0,high=0.64/scales,size=narr)
opt_c2     =np.random.uniform(low=0.0,high=-0.33/scales,size=narr)
opt_obscuration=np.random.uniform(low=0.0,high=0.3/scales,size=narr)
psf_fwhm = np.random.uniform(low=0.5,high=0.7,size=narr)
beta     = np.random.uniform(low=2.2,high=2.5,size=narr)
mag      = np.random.uniform(low=22.5,high=23.0,size=narr)

nx,ny    = 48,48
xin,yin  = 23.5,23.5
ic       = 0
ep1      = -0.03
ep2      = 0.001
while ic< narr:
    i =ic
    xc,yc    = np.random.uniform(low=xin-1.,high=xin+1,size=2)
    noise    = galsim.GaussianNoise(sigma=etc.sigma_sky)
    sigsky   = etc.sigma_sky
    flux     = etc.flux(mag[i])  
    pfwhm    = psf_fwhm[i]
    bet      = beta[i]
    dx,dy    = pixscale*(xc-xin),pixscale*(yc-yin)
    psfmock  = galsim.Moffat(beta=2.5,fwhm=1.5).withFlux(flux).shear(e1=ep1,e2=ep2)
    optics = galsim.OpticalPSF(lam_over_diam,
                               defocus = opt_defocus[i],
                               coma1 = -0.07, coma2 = opt_c2[i],
                               astig1 = 0.0, astig2 = 0.0,
                               obscuration = 0.0)
    psfopt   = galsim.Convolve([psfmock,optics])
    #comma = optics.drawImage(nx=48,ny=48,scale=0.2)
    psfim    = psfopt.drawImage(nx=nx,ny=ny,scale=pixscale)
    #psfim.shift(dy,dx)
    #xcen,ycen= centroids(psfim.array)
    #xcen,ycen= centroids(psfshift.array)
    #imconv   = gaussfilter(psfim.array)
    #xpol,ypol= mfpoly(imconv)
    #fnname    = 'psf_nonoise_'+str(i)+'.fits'
    #hdu1      = pf.PrimaryHDU(psfim.array)
    #hdu1.writeto(fnname)

    #print dx,dy,xcen,ycen
    psfim.addNoise(noise)
    plt.imshow(psfim.array,interpolation='nearest')
    plt.show()
    #fname    = 'psf_noise_'+str(i)+'.fits'
    #hdu      = pf.PrimaryHDU(psfim.array)
    #hdu.writeto(fname)
    
    ic = ic +1 
    
    



