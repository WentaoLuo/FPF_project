#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
import galsim as gs
import lsstetc as lsst
from scipy.fftpack import fftshift

def pcaimages(X):
  num_data,dim = X.shape
  nn           = np.sqrt(dim) 
  mean_X       = X.mean(axis=0)
  for i in range(num_data):
      X[i] -= mean_X

  if dim>100:
      M    = np.dot(X,X.T)    
      e,EV = np.linalg.eigh(M) 
      tmp  = np.dot(X.T,EV).T 
      V    = tmp[::-1]        
      #S    = np.sqrt(e)[::-1]  
  else:
      U,S,V = np.linalg.svd(X)
      V     = V[:num_data] 

  comp_0  = V[0]
  comp_1  = V[1]
  comp_2  = V[2]
  comp_3  = V[3]
  comp_4  = V[4]
  comp_5  = V[5]
  comp_6  = V[6]
  comp_7  = V[7]
  comp_8  = V[8]
  comp_9  = V[8]
  comp_10  = V[10]
  comp_11  = V[11]
  comp_12  = V[12]
  comp_13  = V[13]
  comp_14  = V[14]
  comp_15  = V[15]

  res={'comps':[comp_0,comp_1,comp_2,comp_3,comp_4,comp_5,comp_6,comp_7,comp_8,comp_9,comp_10,comp_11,comp_12,comp_13,comp_14,comp_15]}
  return res

def main():
  narr      = 8000
  imstar_3d = np.zeros((narr,48*48))
  for i in range(narr):
     #fname    = 'shiftims/shifted_'+str(i+1)+'.fits'
     fname    = 'test_04_13/starim_shifted_'+str(i+1)+'.fits'
     #fname    = 'test_04_13/starim_original_'+str(i+1)+'.fits'
     #fname    = 'cfht_stars_for_pca/starim_'+str(i+1)+'.fits'
     image    = pf.getdata(fname)
     imstar_3d[i][:]=image.reshape(48*48)/np.sum(image)

  etc   = lsst.ETC('r', pixel_scale=0.2)
  flux  = etc.flux(23.0)
  psim  = 0.0
  xfc,yfc  = np.mgrid[:48,:48]
  fac   = 1.+0.01+0.5*xfc+0.5*yfc+0.1*xfc**2+0.2*yfc**2+0.1*xfc*yfc
  #for i in range(1000):
  #   noise = gs.GaussianNoise(sigma=etc.sigma_sky)
  #   gal   = gs.Sersic(2.5,half_light_radius=0.6).withFlux(flux).shear(e1=0.3,e2=0.1)
  #   img   = gal.drawImage(nx=48,ny=48,scale=0.2)
  #   img.addNoise(noise)
  #   galimg= fac*img
  #   psim  = psim+np.real(np.fft.fftshift(galimg)*np.fft.fftshift(galimg))
  #print np.max(img.array),np.min(img.array)

  noise = gs.GaussianNoise(sigma=etc.sigma_sky)
  gal   = gs.Sersic(2.5,half_light_radius=0.6).withFlux(flux).shear(e1=0.3,e2=0.1)
  img   = gal.drawImage(nx=48,ny=48,scale=0.2)
  galim = fac*img.array
  img.addNoise(noise)
  #psim  = psim+np.real(np.fft.fftshift(img.array)*np.conj(np.fft.fftshift(img.array)))
  #galim = img.array
  #psim  = psim+np.real(np.fft.fft2(galim)*np.conj(np.fft.fft2(galim)))
  #psim  = psim+np.real(fft2d(galim)*np.conj(fft2d(galim)))
  psfc=np.fft.fftshift(np.abs(np.fft.fft2(galim))**2)
  #plt.subplot(2,1,1)
  #plt.imshow(galim,interpolation='nearest')
  #plt.subplot(2,1,2)
  #plt.imshow(psim,interpolation='nearest')
  #plt.show()
  
  pcstr = pcaimages(imstar_3d)
  com0  = pcstr['comps'][0]
  com1  = pcstr['comps'][1]
  com2  = pcstr['comps'][2]
  com3  = pcstr['comps'][3]
  com4  = pcstr['comps'][4]
  com5  = pcstr['comps'][5]
  com6  = pcstr['comps'][6]
  com7  = pcstr['comps'][7]
  com8  = pcstr['comps'][8]
  com9  = pcstr['comps'][9]
  com10  = pcstr['comps'][10]
  com11  = pcstr['comps'][11]
  com12  = pcstr['comps'][12]
  com13  = pcstr['comps'][13]
  com14  = pcstr['comps'][14]
  com15  = pcstr['comps'][15]
  #print com0
  psim1  =np.fft.fftshift(np.abs(np.fft.fft2(com3.reshape((48,48)))))
  psim2  =np.fft.fftshift(np.abs(np.fft.fft2(com4.reshape((48,48)))))
  psim3  =np.fft.fftshift(np.abs(np.fft.fft2(com5.reshape((48,48)))))
  #psfc=np.fft.fftshift(np.abs(np.fft.fft2(fac))**2)
  #print np.max(psim),np.max(psfc)
  #psim=np.real(np.fft.fftshift(img.array)*np.fft.fftshift(img.array))
  #psim=np.real(np.fft.fft2(img.array)*np.fft.fft2(img.array))
  plt.subplot(4,4,1)
  plt.imshow(com0.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,2)
  plt.imshow(com1.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,3)
  plt.imshow(com2.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,4)
  plt.imshow(com3.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,5)
  plt.imshow(com4.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,6)
  plt.imshow(com5.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,7)
  plt.imshow(com6.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,8)
  plt.imshow(com7.reshape((48,48)),interpolation='nearest')
  #plt.imshow(psim2,interpolation='nearest')
  plt.subplot(4,4,9)
  plt.imshow(com8.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,10)
  plt.imshow(com9.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,11)
  plt.imshow(com10.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,12)
  plt.imshow(com11.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,13)
  plt.imshow(psfc.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,14)
  plt.imshow(psim3.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,15)
  plt.imshow(psim2.reshape((48,48)),interpolation='nearest')
  plt.subplot(4,4,16)
  plt.imshow(psim3.reshape((48,48))/psim3.sum()+psfc/psfc.sum(),interpolation='nearest')
  #plt.imshow(psim3,interpolation='nearest')
  plt.show()

if __name__=='__main__':
  main()

                                      
