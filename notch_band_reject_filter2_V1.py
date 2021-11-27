# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 00:04:18 2021

@author: ghezi661@yahoo.com
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.fftpack as fp
from skimage import data
from skimage import img_as_float

from skimage.color import rgb2gray
from scipy import ndimage

im_noisy1 = cv2.imread('data/CN2.png')
im_noisy = rgb2gray(im_noisy1)
F_noisy = np.fft.fft2(im_noisy)
im=255*rgb2gray(im_noisy1)

im_noisy_2 = cv2.imread('data/CN2.png')
im_noisy2 = rgb2gray(im_noisy_2)
F_noisy2 = np.fft.fft2(im_noisy2)
im2=255*rgb2gray(im_noisy_2)


print(F_noisy.shape)
(380,400)

def plot_image(im , title):
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title(title, size=20)
    
def plot_freq_spectrum(F, title, cmap='gray'):
    plt.imshow((20*np.log10(0.1+ fp.fftshift(F))).real.astype(int), cmap=cmap)
    plt.xticks(np.arange(0, im.shape[1],25))
    plt.yticks(np.arange(0, im.shape[0],25))
    plt.title(title, size=20)


def plot_band_reject(img1):
    
    img2= rgb2gray(img1)
    F=fp.fftshift(np.fft.fft2(img2))
    
    F1=(np.log10(np.abs(F)+1)) 
    maxf=max((map(max , F1)))
    F1[112,112]=0
    maxf=max((map(max , F1)))

    print(maxf)
    maxw=np.where(F1==maxf)
    print(maxw)
    
    M1=maxw[0]
    M2=maxw[1]
    
    M1x=M1[0]
    M1y=M1[1]
   
    
    M2x=M2[0]
    M2y=M2[1]
    
    print(M1x, M1y, M2x, M2y)
    
    
    plt.figure(1)

   
    plt.imshow((np.log10(np.abs(F)+1)), cmap=plt.cm.gray), plt.title('Frequency Domain Image')
 
    [M,N]=F.shape
   
    print('*1111111*')
    
    # plt.subplot(333),plot(np.abs(F(:,round(N/2)))), plt.title('Spectrum on the center vertical line')
    # u=0..M-1
    #v=0..N-1
    
    u=np.arange(0,M)
    v=np.arange(0,N)

    #print(u)
    print('*222222*')

    [V,U]=np.meshgrid(v,u)
    D0=4
    n=3
    u0=M1x
    u1=M1y
    D1=np.abs(U-u0)
    D2=np.abs(U-u1)
    H=(1/(1+((D0**2)/(D1*D2))**n))

    #print([V,U])
    print('333333333')

    v0=M2x
    v1=M2y
    D_1=np.abs(V-v0)
    D_2=np.abs(V-v1)
    H1=(1/(1+((D0**2)/(D_1*D_2))**n))

    plt.figure(2)
    plt.imshow(np.abs(H),cmap=plt.cm.gray), plt.title('Filter image representation')

    plt.figure(3)
    plt.imshow(np.abs(H1),cmap=plt.cm.gray), plt.title('Filter image representation ** 2')

    G=F*H*H1
   
    print('44444444')

    plt.figure(4)
    plt.imshow(np.log10(np.abs(G)+1),cmap=plt.cm.gray), plt.title('Filtered frequency domain image')

    g=(fp.ifft2(fp.ifftshift(G))).real
   
    plt.figure(5)
    plt.imshow(g ,cmap=plt.cm.gray), plt.title('Filter image')
    
    
    plt.imsave('data/t.png',g , cmap='gray')
    #cv2.imwrite('data/test.png',g )
##--------------------------------------------------------------------------

##***********___Bond reject ___*********************************

plot_band_reject(im_noisy) # for image  movarab

##*****************************************************************
 
   
plt.figure(figsize=(20, 10))
plt.figure(6)

plt.figure(7)
#plot_freq_spectrum(F_noisy2 , 'noisy image spectrom')

plt.tight_layout()
plt.show()

F_noisy_shifted = fp.fftshift(F_noisy2)
F_noisy_shifted[112,125] = F_noisy_shifted[133,141] = 0

im_out = fp.ifft2(fp.ifftshift(F_noisy_shifted)).real
plt.figure(figsize=(10,8))

plt.figure(8)
plt.show()

