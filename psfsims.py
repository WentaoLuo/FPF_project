#!/home/wtluo/anaconda/bin/python2.7

import numpy as np 
import galsim 
import lsstetc
import matplotlib.pyplot as plt 
import scipy.optimize as opt 
import pyfits as pf

pi = np.pi
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
etc = lsstetc.ETC('r',pixel_scale=0.2)
narr     = 2000
psf_fwhm = np.random.uniform(low=0.6,high=0.8,size=narr)
beta     = np.random.uniform(low=2.4,high=2.5,size=narr)
#psf_fwhm = 0.7
#beta     = 2.5
mag      = np.random.uniform(low=22.5,high=23.0,size=narr)

nx,ny    = 48,48
xin,yin  = 23.5,23.5
ic       = 0
while ic< narr:
    i =ic
    xc,yc    = np.random.uniform(low=xin-1.,high=xin+1,size=2)
    imnull   = galsim.Sersic(3.0,half_light_radius=1.9).withFlux(10000000.0).shear(e1=0.5,e2=-0.1)
    imm      = imnull.drawImage(nx=nx,ny=ny,scale=0.2)
    noise    = galsim.GaussianNoise(sigma=etc.sigma_sky)
    imm.addNoise(noise)
    sigsky   = etc.sigma_sky
    #print sigsky
    flux     = etc.flux(mag[i])  
    fwhm     = psf_fwhm[i]
    bet      = beta[i]
    psfmock  = moff(xc,yc,fwhm,bet,flux,nx,ny)
    #chi2     = lambda *args: -lnlike(*args)
    #results  = opt.minimize(chi2,[xin,yin],args=(psfmock,sigsky,flux,bet,fwhm,nx,ny)) 
    #xm,ym    = results["x"]
    #fname    = 'psf_'+str(i)+'.fits'
    #hdu      = pf.PrimaryHDU(imm.array+psfmock)
    #hdu.writeto(fname)
    plt.imshow(imm.array,interpolation='nearest')
    plt.show()
    ic = ic +1 
    
    print i,xc,yc
    



