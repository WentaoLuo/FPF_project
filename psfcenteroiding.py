#!/home/wtluo/anaconda/bin/python2.7

import warnings
import numpy as np 
import galsim 
import lsstetc
import matplotlib.pyplot as plt
import scipy.signal as signal
import pyfits as pf

pi = np.pi
#----------------------------------------------------------------------
def centroids(psfimage):
  nx,ny  = psfimage.shape[0],psfimage.shape[1]
  x,y    = np.mgrid[:nx,:ny]
  m00    = np.sum(psfimage)
  x0     = np.sum(x*psfimage)/m00
  y0     = np.sum(y*psfimage)/m00
  mxy  =np.sum((y-y0)*(x-x0)*psfimage)/m00
  mxx  =np.sum((x-x0)*(x-x0)*psfimage)/m00
  myy  =np.sum((y-y0)*(y-y0)*psfimage)/m00

  itr  =10
  for i in range(itr):
       detM =mxx*myy-mxy*mxy
       mxx  =mxx/detM
       myy  =myy/detM
       mxy  =-mxy/detM

       w    =(x-x0)*(x-x0)*mxx+2.0*(x-x0)*(y-y0)*mxy+(y-y0)*(y-y0)*myy
       w    =np.exp(-0.5*w)
       m00  =np.sum(w*psfimage)
       x0   =np.sum(x*w*psfimage)/m00
       y0   =np.sum(y*w*psfimage)/m00
       mxy  =np.sum((y-y0)*(x-x0)*w*psfimage)
       mxx  =np.sum((x-x0)*(x-x0)*w*psfimage)
       myy  =np.sum((y-y0)*(y-y0)*w*psfimage)


  return [x0,y0]
#----------------------------------------------------------------------
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
#---------------------------------------------------------------------

def main():
  narr      = 3000
  #finput    = 'psf_images/'+'snr100_inputs'
  #finput    = 'psf_images/'+'snr50_inputs'
  #finput    = 'psf_images/'+'snr20_inputs'
  #finput    = 'psf_images/'+'snr20_optics_e_inputs'
  #finput    = 'psf_images/'+'snr50_optics_e_inputs'
  #finput    = 'psf_images/'+'snr100_optics_e_inputs'
  #dat       = np.loadtxt(finput,unpack=True)
  #xin       = dat[1][:]
  #yin       = dat[2][:]

  warnings.simplefilter('ignore')
  
  for i in range(narr):
     fname1    = 'test_optics/snr100_03_23/psf_nonoise'+str(i)+'.fits'
     fname2    = 'test_optics/snr100_03_23/psf_noise'+str(i)+'.fits'
     image1   = pf.getdata(fname1)
     image2   = pf.getdata(fname2)
     imconv   = gaussfilter(image2)

     xp,yp    = mfpoly(imconv) 
     xc,yc    = centroids(image1) 
     print i,xp,yp,xc,yc
 
  

if __name__=='__main__':
  main()

