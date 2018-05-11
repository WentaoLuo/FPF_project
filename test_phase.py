#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt

def getphase(fftimage):
  freal = np.real(fftimage)
  fimag = np.imag(fftimage)
  phase = np.arctan2(fimag,freal)

  return phase 

def testphase(center,fftimage):
  nx,ny =fftimage.shape
  xc,yc = center 
  kx,ky = np.mgrid[0:nx,0:ny] 
  phase = np.zeros((nx,ny))
  for i in range(nx):
     for j in range(ny):
	phase[i,j]= np.sum(2.0*np.pi*(float(i)*(kx.astype(float)-xc))/float(nx)+\
	            float(j)*(ky.astype(float)-yc)/float(ny))
  return phase

def moff(xc,yc,fwhm,beta,flx,nx,ny):
   alpha = fwhm/(2.0*np.sqrt(2.0**(1/beta)-1.0))/0.2
   x,y   = np.mgrid[0:nx,0:ny]
   rr    = np.sqrt((x-xc)**2+(y-yc)**2)
   image = flx*(beta-1.0)*(1+(rr/alpha)**2)**(-beta)/pi/alpha/alpha

   return image

#------------------------------------------

arr1 = np.array([[1,1,1],[1,4,1],[1,1,1]])
arr2 = np.array([[1,2,1],[2,4,1],[3,2,1]])
arr3 = np.array([[1,2,1],[1,4,2],[1,2,3]])

kx,ky = np.mgrid[0:3,0:3] 
#print kx
#print ky
#far1 = np.fft.fftshift(np.fft.fft2(arr1))
#far2 = np.fft.fftshift(np.fft.fft2(arr2))
#far3 = np.fft.fftshift(np.fft.fft2(arr3))
far1 = (np.fft.fft2(arr1))
far2 = (np.fft.fft2(arr2))
far3 = (np.fft.fft2(arr3))

pha1 = getphase(far1)
pha2 = getphase(far2)
pha3 = getphase(far3)
center= [1.1,1.1]
pha11= testphase(center,far1)*np.pi/180.0
pha21= testphase(center,far2)*np.pi/180.0
pha31= testphase(center,far3)*np.pi/180.0
print pha1
print pha11
print pha2
print pha21
print pha3
print pha31


