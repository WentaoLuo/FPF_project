#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import pyfits as pf
import scipy.misc as msc
from scipy import signal
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt
import galsim  as gs
import emcee
import scipy.optimize as opt

pi = np.pi
#modeling of each star using PCA------------------------------------------
#return coefficients for each star using either chi-square or machine learning
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

#calculating the centroid using an adaptive Gaussian weight----------------
def moments(image):
  m00 = np.sum(image)
  x,y = np.mgrid[:image.shape[0],:image.shape[1]]
  x0  = np.sum(x*image)/m00
  y0  = np.sum(y*image)/m00
  mxy = np.sum((x-x0)*(y-y0)*image)/m00
  mxx = np.sum((x-x0)*(x-x0)*image)/m00
  myy = np.sum((y-y0)*(y-y0)*image)/m00
  #print x0,y0
  itr = 0
  for i in range(itr):
       detM =mxx*myy-mxy*mxy
       mxx  =mxx/detM
       myy  =myy/detM
       mxy  =-mxy/detM

       w    =(x-x0)*(x-x0)*mxx+2.0*(x-x0)*(y-y0)*mxy+(y-y0)*(y-y0)*myy
       w    =np.exp(-0.5*w)
       m00  =np.sum(w*image)
       x0   =np.sum(x*w*image)/m00
       y0   =np.sum(y*w*image)/m00
       mxy  =np.sum((y-y0)*(x-x0)*w*image)
       mxx  =np.sum((x-x0)*(x-x0)*w*image)
       myy  =np.sum((y-y0)*(y-y0)*w*image)

  cen = [x0,y0]
  mnstr = {'centroid':cen,'moments':[mxx,mxy,myy]}
  return mnstr

#shift the centroid of the original star image-----------------------------
def shiftcen(image,offset):
  xoff,yoff = np.abs(offset)
  nx,ny     = image.shape
  shift_im  = np.zeros([nx,ny])
  shift_im[0:nx-xoff,0:ny-yoff]  = image[xoff:,yoff:]

  return shift_im

#PCA images 1--------------------------------------------------------------------
def pcaimages(X):
  #get dimensions
  num_data,dim = X.shape
  nn           = np.sqrt(dim)
  #center data
  mean_X    = X.mean(axis=0)
  for i in range(num_data):
      X[i] -= mean_X

  if dim>100:
      #print 'PCA - compact trick used'
      M    = np.dot(X,X.T)     #covariance matrix
      e,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
      tmp  = np.dot(X.T,EV).T  #this is the compact trick
      V    = tmp[::-1]         #reverse since last eigenvectors are the ones we want
      #S    = np.sqrt(e)[::-1]  #reverse since eigenvalues are in increasing order
  else:
      #print 'PCA - SVD used'
      U,S,V = np.linalg.svd(X)
      V     = V[:num_data] #only makes sense to return the first num_data
     
  comp_0  = V[0]
  comp_1  = V[1]
  comp_2  = V[2]
  comp_3  = V[3]

  res={'comps':[comp_0,comp_1,comp_2,comp_3]}
  return res

# psf model-------------------------------------------------------
def psfmodel(coeffs,pcstr):
  a1,a2,a3,a4 = coeffs
  comp1,comp2,comp3,comp4 = pcstr['comps']
  psf = a1*comp1+a2*comp2+a3*comp3+a4*comp4
  
  return psf

# likelyhood or chi2 function to maximize or minimize---------------
def lnlike(coeffs,imstars,pcstr):
  psf   = psfmodel(coeffs,pcstr)
  res = -0.5*(np.sum((imstars-psf)**2))
  return res

#reading cfhtlens files -------------------

def main():
   prefix   = 'star_'
   secd     = np.array([831549,831550,831551,831552,831553,831554,831555])
   thid     = np.linspace(1,36,36)
#---test image extraction---
    
   #nx = int(xmax_str/48.0)
   #ny = int(ymax_str/48.0)
   nx = 100
   ny = 100
   nstar = 100
   imstar_3d = np.zeros([nstar,48*48])
   imstars   = np.zeros([nstar,48*48])
   ellipstar = np.zeros([nstar,2])
   Rstar     = np.zeros(nstar)
   num       = 0   
#   for ix in range(1):
   for ix in range(len(secd)):
      #for iy in range(1):
      for iy in range(len(thid)):
          if thid[iy]<=9:
             starf = 'w2_m0m0_stars/star_'+str(secd[ix])+'_0'+str(int(thid[iy]))+'.fits'
          else:
             starf = 'w2_m0m0_stars/star_'+str(secd[ix])+'_'+str(int(thid[iy]))+'.fits'
          #print starf
          fcen1 = 'fourier_center/fourier1-'+str(secd[ix])+'-'+str(int(thid[iy]))+'.txt'
          fcen3 = 'fourier_center/fourier3-'+str(secd[ix])+'-'+str(int(thid[iy]))+'.txt'
          dcen1 = np.loadtxt(fcen1,unpack=True)
          xf1   = dcen1[0][:]
          yf1   = dcen1[1][:]
          dcen3 = np.loadtxt(fcen3,unpack=True)
          xf3   = dcen3[0][:]
          yf3   = dcen3[1][:]
          plt.plot(xf1,xf3,'k.')
          plt.plot(yf1,yf3,'r.')
          plt.xlim(-1.,1.)
          plt.ylim(-1.,1.)
          plt.show()
          starim = pf.getdata(starf) 
          xmax   = starim.shape[0]  
          ymax   = starim.shape[1]  
          nx = int(xmax/48.0)
          ny = int(ymax/48.0)
          for i in range(nx):
             for j in range(ny):
                i_min = i*48
                i_max = (i+1)*48
                j_min = j*48
                j_max = (j+1)*48
                slc_i = slice(i_min,i_max)
                slc_j = slice(j_min,j_max)
                if j+i*nx<nstar:
                   starsub      = starim[slc_i,slc_j]
                   if np.max(starsub)!=0.0:
                      imconv   = gaussfilter(starsub)
                      xp,yp    = mfpoly(imconv)
                      print secd[ix],int(thid[iy]),xp,yp
                      shiftvec= [23.5-xp,23.5-yp]
                      shiftim = ndimg.shift(starsub,shiftvec)
                      num=num+1
                      fileim = 'starim_shifted_'+str(num)+'.fits'
                      origim = 'starim_original_'+str(num)+'.fits'
                      hdu1   = pf.PrimaryHDU(shiftim)
                      hdu2   = pf.PrimaryHDU(starsub)
                      hdu1.writeto(fileim)
                      hdu2.writeto(origim)
   


  
  
if __name__=='__main__':
   main()



