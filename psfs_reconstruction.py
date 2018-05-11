#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import pyfits as pf
import scipy.misc as misc 
import matplotlib.pyplot as plt 
import galsim  as gs
import emcee
import scipy.optimize as opt


#modeling of each star using PCA------------------------------------------
#return coefficients for each star using either chi-square or machine learning
"""def polyfit(coeffs,x,y):
  f01,f02,f11,f12,f21,f22,l20,l11,l02,r20,r11,r02,w20,w11,w02 = coeffs
  a1  = l20*x*x+l11*x*y+l02*y*y+f01*x+f02*y          
  a2  = r20*x*x+r11*x*y+r02*y*y+f11*x+f12*y          
  a3  = w20*x*x+w11*x*y+w02*y*y+f21*x+f22*y      
  res = [a1,a2,a3] 
  return res"""
# TOO MUCH PARAMETERS TO FIT!!!!!!!!!!!  NEVER DO THAT!!!!!!!!!

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
   #prefix     = '/media/wtluo/8A6C28646C284CEF/w2/'
   #pointing   = 'w2m0m0/step1/'
   prefix     = '../'
   pointing   = 'step1/'
   exposure   = '831549'
   chipid     = '_01'
   allgalfits = prefix+pointing+'gal_'+exposure+chipid+'.fits'    
   allstarfits= prefix+pointing+'star_'+exposure+chipid+'.fits'    
   galcat     = prefix+pointing+'gal_info'+exposure+chipid+'.dat'    
   starcat    = prefix+pointing+'star_info'+exposure+chipid+'.dat'    
   
   #hdr     = pf.open(allgalfits)  
   sky     = 2653.10250279776
   galim   = pf.getdata(allgalfits)
   starim  = pf.getdata(allstarfits)
   galdat  = np.loadtxt(galcat,unpack=True,skiprows=1)
   stardat = np.loadtxt(starcat,unpack=True,skiprows=1)

   ra_gal  = galdat[0][:]
   dec_gal = galdat[1][:]
   e1_lf   = galdat[4][:]
   e2_lf   = galdat[5][:]
   w_lf    = galdat[6][:]
   m       = galdat[10][:]
   c       = galdat[11][:]
   xgal    = galdat[17][:]
   ygal    = galdat[18][:]
   snrgal  = galdat[19][:]
   ngal    = np.size(ra_gal)

   xstar   = stardat[1][:]
   ystar   = stardat[2][:]
   ra_st   = stardat[3][:] 
   dec_st  = stardat[4][:] 
   snrst   = stardat[5][:]
   nstar   = np.size(ra_st)
 
   xmax_gal = np.size(galim[:][0])
   ymax_gal = np.size(galim[0][:])
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
  	     mnstr        = moments(starsub)
             cen_star     = mnstr['centroid']
             mxx,mxy,myy  = mnstr['moments']
             ellipstar[j+i*nx][0] = (mxx-myy)/(mxx+myy)
             ellipstar[j+i*nx][1] = (2.*mxy)/(mxx+myy)
             Rstar[j+i*nx]        = mxx+myy
             cen_ini      = np.array([24,24])
             offset       = np.array([int(round(cen_star[0]-cen_ini[0])),\
                                      int(round(cen_star[1]-cen_ini[1]))])
             shift_im     = shiftcen(starsub,offset)
             #imstar_3d[j+i*nx][:]    = shift_im.reshape(48*48)/np.sum(starsub)
             imstar_3d[j+i*nx][:]    = starsub.reshape(48*48)/np.sum(starsub)
             imstars[j+i*nx][:]      = starsub.reshape(48*48)

   pcstr      = pcaimages(imstar_3d)
   coeffs_ini = [1.,1.,1.,1.]

   ind        = np.random.randint(low=0,high=100,size=100) 
   idx        = np.unique(ind)
   ntrain     = len(idx)
   comatrix   = np.zeros((ntrain,4))
   trainsample= np.zeros((ntrain,4))
  
   for ix in range(len(idx)): 
      i          = idx[ix]
      chi2       = lambda *args:-lnlike(*args)
      constraints= opt.minimize(chi2,coeffs_ini,\
                args=(imstars[i][:]/np.sum(imstars[i][:]),pcstr))
      comatrix[ix][:] = constraints["x"]
      trainsample[ix][0] = xstar[i]-np.mean(xstar)
      trainsample[ix][1] = ystar[i]-np.mean(ystar)
      trainsample[ix][2] = ellipstar[i][0]
      trainsample[ix][3] = ellipstar[i][1]
  

   from mpl_toolkits.mplot3d import Axes3D    

   """fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(xstar[idx]-np.mean(xstar),ystar[idx]-np.mean(ystar),comatrix[:,3], c='r', marker='o')

   ax.set_xlabel('xccd-center')
   ax.set_ylabel('yccd-center')
   ax.set_zlabel('Coeff_4')
   plt.show()"""
   # ANN learning as interpolation----------------------
   from pybrain.tools.shortcuts import buildNetwork
   from pybrain.structure import SigmoidLayer,LinearLayer
   from pybrain.datasets import SupervisedDataSet 
   from pybrain.supervised.trainers import BackpropTrainer

   net = buildNetwork(2,3,3,4,bias=True,hiddenclass=SigmoidLayer,\
         outclass=LinearLayer)
   ds  = SupervisedDataSet(2,4)

   for iy in range(len(idx)):
       i = idx[iy]
       ds.addSample((xstar[i]-np.mean(xstar),ystar[i]-np.mean(ystar)),\
              (comatrix[iy][0],comatrix[iy][1],comatrix[iy][2],comatrix[iy][3]))

   trainer = BackpropTrainer(net,ds)
   trainer.trainUntilConvergence()

   nout  = nstar-len(idx)
   pos   = np.zeros([nout,2])
   for j in range(nout):
      if j != idx[j]:
         pos[j][0] = xstar[j]-np.mean(xstar)
         pos[j][1] = ystar[j]-np.mean(ystar)
         outmatrix = net.activate([xstar[j]-np.mean(xstar),ystar[j]-np.mean(ystar)]) 
         psfim     = psfmodel(outmatrix,pcstr)
         #print psfim.shape
         mnpsf     = moments(psfim.reshape(48,48))
         mxx,mxy,myy = mnpsf['moments']
         Rpsf      = mxx+myy
         e1psf     = (mxx-myy)/(mxx+myy)
         e2psf     = (2.*mxy)/(mxx+myy)
         print Rstar[j],ellipstar[j][0],ellipstar[j][1],\
               Rpsf,e1psf,e2psf

   """fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(ra_st[idx],dec_st[idx],comatrix[:,1], c='r', marker='o')
   ax.scatter(pos[:,0],pos[:,0],outmatrix[:,1], c='g', marker='o')

   ax.set_xlabel('ra')
   ax.set_ylabel('dec')
   ax.set_zlabel('Coeff_2')
   plt.show()"""
   # ANN learning as interpolation----------------------
   # Polynomial Interpolation----------------------------

   #plt.imshow((imstar_3d[0][:]).reshape(48,48),interpolation='nearest')
   #plt.imshow((psf-imstars[0][:]).reshape(48,48),interpolation='nearest')
   #plt.colorbar() 
   #plt.show()

if __name__=='__main__':
   main()



