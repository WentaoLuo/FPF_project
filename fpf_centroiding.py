#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import galsim as gs
import lsstetc
import scipy.ndimage as ndimg
import pyfits as pf

pi = np.pi

#########################################################################
def polyfit1(fftimage):
   xc,yc =32.0,32.0
   return 0
#-------------------------------------------------------------------------
def getphase(fftimage):
   freal = np.real(fftimage)
   fimag = np.imag(fftimage)
   phase = np.arctan2(fimag,freal)  

   return phase
#-------------------------------------------------------------------------
def moff(xc,yc,fwhm,beta,flx,nx,ny):
  alpha = fwhm/(2.0*np.sqrt(2.0**(1/beta)-1.0))/0.2
  x,y   = np.mgrid[0:nx,0:ny]
  rr    = np.sqrt((x-xc)**2+(y-yc)**2)
  image = flx*(beta-1.0)*(1+(rr/alpha)**2)**(-beta)/pi/alpha/alpha

  return image
#-------------------------------------------------------------------------
def main():
  
  xc = 32.
  yc = 32.
  dx = 2.5
  dy = 3.5
  
  beta = 2.4
  fwhm = 0.7
  etc  = lsstetc.ETC('r',pixel_scale=0.2)
  flux = etc.flux(20.0)
  nax  = 64 
  imnull   = gs.Sersic(2.0,half_light_radius=0.1).withFlux(0.0)
  imm      = imnull.drawImage(nx=nax,ny=nax,scale=0.2)
  noise    = gs.GaussianNoise(sigma=etc.sigma_sky)
  #imm.addNoise(noise)
  noiseim  = imm.array

  psfmock  = moff(xc,yc,fwhm,beta,flux,nax,nax)+noiseim
  shiftvec = [dy,dx]
  psfshift = ndimg.shift(psfmock,shiftvec)+noiseim

  #fftimgcen= np.fft.fftshift(np.abs(np.fft.fft2(psfmock)))
  fftimgcen= np.fft.fftshift(np.fft.fft2(psfmock)) 
  #fftimgcen= (np.fft.fft2(psfmock))
  #fftimgoff= np.fft.fftshift(np.abs(np.fft.fft2(psfshift)))
  #fftimgoff= np.fft.fft2(psfshift)
  fftimgoff= np.fft.fftshift((np.fft.fft2(psfshift)))
  fftimgceni= np.log(np.imag(np.fft.fftshift(np.fft.fft2(psfmock))))
  fftimgcenr= np.log(np.real(np.fft.fftshift(np.fft.fft2(psfmock))))
  fftimgoffi= np.log(np.imag(np.fft.fftshift(np.fft.fft2(psfshift))))
  fftimgoffr= np.log(np.real(np.fft.fftshift(np.fft.fft2(psfshift))))
  phase1    = getphase(fftimgcen)
  phase2    = getphase(fftimgoff)
  
  #test=(np.mod(phase1,pi))
  test=np.floor(phase1/pi)
  ixa = (phase1)>=3.0
  ixb = (phase1)<=-3.0
  #phase1[ixa]=phase1[ixa]-pi
  #phase1[ixb]=phase1[ixb]+pi
  
  #plt.hist(phase2.reshape(64*64),50,facecolor='blue') 
  #plt.hist(phase1.reshape(64*64),50,facecolor='red') 
  #plt.show()
  ftsize =18
  fig,axes=plt.subplots(2,2)
  print np.max(test),np.min(test)
  #plt.subplot(2,2,1)
  axes[0,0].imshow(np.log10(psfmock),interpolation='nearest')
  axes[0,0].set_title('perfectly centered',fontsize=ftsize)
  axes[0,0].hlines(32,xmin=0,xmax=64)
  axes[0,0].vlines(32,0,64)
  #plt.subplot(2,2,2)
  #im=axes[0,1].imshow((phase1%pi)%pi,interpolation='nearest')
  im=axes[0,1].imshow(phase1,interpolation='nearest')
  #plt.hlines(32,xmin=0,xmax=64)
  #plt.vlines(32,0,64)
  #plt.imshow((phase1%pi)%pi,interpolation='nearest')
  plt.colorbar(im,ax=axes[0,1])
  axes[0,1].set_title('phase pattern',fontsize=ftsize)
  #plt.subplot(2,2,3)
  axes[1,0].imshow(np.log10(psfshift),interpolation='nearest')
  #plt.hlines(32,xmin=0,xmax=48)
  #axes[1,0].set_vlines(24,ymin=0,ymax=48,ls='--')
  axes[1,0].set_title('off centered',fontsize=ftsize) 
  im2=axes[1,1].imshow((phase2%pi)%pi,interpolation='nearest')
  axes[1,0].hlines(32,xmin=0,xmax=64)
  axes[1,0].vlines(32,0,64)
  plt.colorbar(im2,ax=axes[1,1])
  axes[1,1].set_title('phase pattern',fontsize=ftsize)
  fig.tight_layout()
  plt.subplots_adjust(wspace=-0.5, hspace=0.3) 
  plt.savefig('phase_pattern.eps')
  plt.show()

if __name__ =='__main__':

   main()
