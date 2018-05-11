#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import pyfits as pf
import scipy.misc as msc
import scipy.signal as signal
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt
import galsim  as gs
import emcee
import scipy.optimize as opt
import os

pi = np.pi
#----------------------------------------------------------------------
def gaussfilter(psfimage):
  x,y   = np.mgrid[0:3,0:3]
  xc,yc = 1.0,1.0
  w     = 1.2
  ker   = (1.0/2.0/pi/w/w)*np.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
  imcov = signal.convolve(psfimage,ker,mode='same')
  return imcov
 
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

def centroids(image):
   m00   =np.sum(image)
   x, y =np.mgrid[:image.shape[0],:image.shape[1]]
   x0   =np.sum(x*image)/m00
   y0   =np.sum(y*image)/m00
   mxy  =np.sum((y-y0)*(x-x0)*image)/m00
   mxx  =np.sum((x-x0)*(x-x0)*image)/m00
   myy  =np.sum((y-y0)*(y-y0)*image)/m00

   itr  =10
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
       mxy  =np.sum((y-y0)*(x-x0)*w*image)/m00
       mxx  =np.sum((x-x0)*(x-x0)*w*image)/m00
       myy  =np.sum((y-y0)*(y-y0)*w*image)/m00

   return [x0,y0]         

#--------------------------------------------------------------------
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

#reading cfhtlens files -------------------

def main():
   pointing   = 'w2m0m0/step1/'
   prefix     = '../../../'
   pointing   = 'step1/'
   exposure   = '831549'
   chipid     = '_01'
   allgalfits = prefix+pointing+'gal_'+exposure+chipid+'.fits'    
   allstarfits= prefix+pointing+'star_'+exposure+chipid+'.fits'    
   galcat     = prefix+pointing+'gal_info'+exposure+chipid+'.dat'    
   starcat    = prefix+pointing+'star_info'+exposure+chipid+'.dat'    
   
   mgpix   = 0.08   
   sky     = 2653.10250279776
   starim  = pf.getdata(allstarfits)
   fcenters= 'fourier_centers'
   stardat = np.loadtxt(starcat,unpack=True,skiprows=1)
   fcen    = np.loadtxt(fcenters,unpack=True,skiprows=1)
   #fcen    = np.l
   #ra_gal  = galdat[0][:]
   #dec_gal = galdat[1][:]
   #e1_lf   = galdat[4][:]
   #e2_lf   = galdat[5][:]
   #w_lf    = galdat[6][:]
   #m       = galdat[10][:]
   #c       = galdat[11][:]
   #xgal    = galdat[17][:]
   #ygal    = galdat[18][:]
   #snrgal  = galdat[19][:]
   #ngal    = np.size(ra_gal)

   xstar   = stardat[1][:]
   ystar   = stardat[2][:]
   ra_st   = stardat[3][:] 
   dec_st  = stardat[4][:] 
   snrst   = stardat[5][:]
   nstar   = np.size(ra_st)
 
   #xmax_gal = np.size(galim[:][0])
   #ymax_gal = np.size(galim[0][:])
   xmax_str = np.size(starim[:][0])
   ymax_str = np.size(starim[0][:])
   #print xmax_str/48.,ymax_str/48.
#---test image extraction---
   
   nx = int(xmax_str/48.0)
   ny = int(ymax_str/48.0)

   imstar_3d = np.zeros([nstar,48*48])
   imstars   = np.zeros([nstar,48*48])
   ellipstar = np.zeros([nstar,2])
   Rstar     = np.zeros(nstar)
   #imgalsim  = gs.ImageF(48,48) 
   #imtmp     = imgalsim.drawImage(nx=48,ny=48) 
   #print imgalsim.array
   for i in range(nx):
      for j in range(ny):
          i_min = i*48
          i_max = (i+1)*48
          j_min = j*48
          j_max = (j+1)*48
          slc_i = slice(i_min,i_max)
          slc_j = slice(j_min,j_max)
          #galsub= galim[slc_i,slc_j]
          if j+i*nx<nstar:
             starsub      = starim[slc_i,slc_j]
             imconv       = gaussfilter(starsub)  
             xp,yp        = mfpoly(imconv)
             xc,yc        = centroids(starsub)
             if np.isnan(xc)==False:
               ftmp         = 'psf_'+str(j+i*nx)+'.fits'
               #hdu          = pf.PrimaryHDU(starsub)
               #hdu.writeto(ftmp)
               print j+i*nx,xp,yp,xc,yc
               shiftvec= [23.5-xp,23.5-yp]
               shiftim = ndimg.shift(starsub,shiftvec)
               imconv1 = gaussfilter(shiftim)  
               xp1,yp1 = mfpoly(imconv1)
               print xp1,yp1
               plt.imshow(shiftim,interpolation='nearest')
               plt.show()





   #pcstr      = pcaimages(imstar_3d)



if __name__=='__main__':
   main()



