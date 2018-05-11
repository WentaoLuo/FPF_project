#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt

dat1 = np.loadtxt('before_shift_poly.dat',unpack=True)
xp1  = dat1[1][:]
yp1  = dat1[2][:]
dat2 = np.loadtxt('after_shift_poly.dat',unpack=True)
xp2  = dat2[1][:]
yp2  = dat2[2][:]
dat3 = np.loadtxt('cfht_results.txt',unpack=True)
xt   = dat3[0][:]
yt   = dat3[1][:]

xx   = [-1,1]
yy   = [-1,1]
#plt.plot(-xp1+0.5,xt,'r+')
#plt.plot(xx,yy,'k-',linewidth=3)
plt.plot(yp2,yp1,'g*')
#plt.xlim(-1,1)
#plt.ylim(-1,1)
plt.xlabel('polynomial')
plt.ylabel('fourier')
plt.show()

