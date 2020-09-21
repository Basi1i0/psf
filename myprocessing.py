# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 02:36:45 2020

@author: WFS
"""

import os
import numpy as np
import pandas as pd
import json
from skimage.io import imread
from scipy.io import loadmat

import sys
sys.path.append('C:/Users/WFS/repos/psf/psf/')
from main import *

import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
sns.set_context('paper', font_scale=2.0)
sns.set_style('ticks')

from ipywidgets import interactive
from ipywidgets import IntSlider
from IPython.display import display

#%%

basepath = 'J:/Vasily/20200827_correction_newlens/2ppsf/Nikon40x/';
dirname = '/meas02/'
metadataname = '/metadata/basicMetaData.mat'
filename = '/file_00001.tif'


xunits = 'pix';#'um';
zunits = 'um';

metadata = loadmat(basepath + dirname + metadataname)['metadata'];
zoom = metadata['zoom'][0][0][0][0]
zstep = metadata['stackZStepSize'][0][0][0][0];
zrange = metadata['stackZEndPos'][0][0][0][0] -  metadata['stackZStartPos'][0][0][0][0];

FOVumLat = 512.0#512.0/(0.32*zoom)
FOVpxLat = 512.0 # 512 
pxPerUmLat = FOVpxLat/FOVumLat  
pxPerUmAx = 1/zstep

windowUm = [zrange/2*0.85, 15, 15] #bead will be discarded if it falls outside fov in x,y or Z
options = {'FOVumLat':FOVumLat, 'FOVpxLat':FOVpxLat, 'pxPerUmLat':FOVpxLat/FOVumLat, 
           'pxPerUmAx':pxPerUmAx, 'windowUm':windowUm} 
options['countsvar'] = .5
options['thresh'] = .1
options


#%% 
#'direct_.2umbeads_1/zoom15_step0.2um_-5005.5_-4985.5_pos2/'
# 'upsidedown_.2umbeads/zoom15_step0.2um_-5018_-4998_pos3/'

im = imread(basepath +dirname + filename, plugin='tifffile')

print(im.shape)
for i in range(0, im.shape[0]):
       im[i] = im[i]-np.median(im[i][im[i] < options['thresh']*np.max(im) ])
im = im - np.min(im);

#im = np.empty((2048, 2048,0));
#
#dirname = "J:/Vasily/20200807/2p_excitation_1p_collection_psf/"
#f = [];
#for filename in os.listdir(dirname):
#    print(filename);
#    image = np.array(imread(dirname + filename));
#    im = np.dstack((im, image));
#    
#droppixs = int((1024 + 512 + 256 + 128)/2);    
#im = im[droppixs:(-droppixs),droppixs:(-droppixs),:]
#options['FOVpxLat'] = options['FOVpxLat'] - droppixs*2;
#options['FOVumLat'] = options['FOVpxLat']/options['pxPerUmLat'];
#im = np.append(im, im, axis = 0);
#im = np.append(im, im, axis = 1);
#options['FOVpxLat'] = options['FOVpxLat']*2;       
#im = np.transpose(im, (2,0,1))    
#

       
#for i in range(0, im.shape[0]):
#       im[i] = im[i] - np.min(im)


#%%
fig = plt.figure(figsize=(20,20));
beads, maxima, centers, smoothed = getCenters(im, options)
fig.savefig(basepath +dirname + 'spots_selection.png')
#%%

data = [getPSF(x, options) for x in beads]
max_vals = [d[4][4] for d in data]

counts_backgrounds = [np.mean([d[1][5], d[2][5], d[3][5]]) for d in data]

allbeads_data = pd.concat([x[0] for x in data])
allbeads_data['Counts'] = np.round((beads - np.mean(counts_backgrounds)).sum(axis = 3).sum(axis = 2).max(axis = 1))


allbeads_data = allbeads_data.reset_index().drop(['index'],axis=1)

summary = pd.concat([allbeads_data.median(), allbeads_data.std(), allbeads_data.std()/(allbeads_data.shape[0])**0.5], axis = 1)
summary.columns = ['median', 'std', 'err']

summary.to_csv(basepath +dirname+'summary.txt', sep = '\t')
summary

#%%
average = np.mean(beads, 0)#

data_show = getPSF(average, options)
PSF = data_show[0]
PSF = PSF.reset_index().drop(['index'],axis=1)

counts_background = np.mean([data_show[1][5], data_show[2][5], data_show[3][5]]);
PSF['Counts'] = np.max((average - counts_background).sum(axis=2).sum(axis=1))

latProfile = data_show[1]
axProfile = data_show[2]
centProfile = data_show[3]
maxProfile = data_show[4]


#plane = IntSlider(min=0, max=average.shape[0]-1, step=1, value=average.shape[0]/2)
#interactive(lambda i: plt.imshow(average[i]), i=plane)

#%%


ncols = 7;
fig = plt.figure(figsize=(3*ncols, 3.5*average.shape[0]/ncols))

for i in range(1, average.shape[0]):
    plt.subplot( np.ceil(average.shape[0]/ncols), ncols, i)
    plt.imshow(average[i,:,:], extent=[-options['windowUm'][1], options['windowUm'][1], 
                                         -options['windowUm'][2], options['windowUm'][2]])
    plt.title('z='+str( round((i-np.round(average.shape[0]/2))/options['pxPerUmAx'], 2) ) + zunits)
    
    

fig.savefig(basepath +dirname + 'spots_zscan.png')
#%%
fig = plt.figure(figsize=(12,10));
plt.subplot(2, 3, 1)
plt.imshow(average.mean(axis=0), extent=[-options['windowUm'][1], options['windowUm'][1], 
                                         -options['windowUm'][2], options['windowUm'][2]])
plt.xlabel(xunits)
plt.ylabel(xunits)
plt.title('z average')

plt.subplot(2, 3, 4)
plt.imshow(average[int(average.shape[0]/2)], extent=[-options['windowUm'][1], options['windowUm'][1], 
                                         -options['windowUm'][2], options['windowUm'][2]])
plt.xlabel(xunits)
plt.ylabel(xunits)
plt.title('z max section')


plt.subplot(1, 3, 2)

plt.imshow(average.mean(axis=1), aspect = 0.5, #pxPerUmLat/pxPerUmAx
           extent=[-options['windowUm'][1], options['windowUm'][1], 
                   -options['windowUm'][0], options['windowUm'][0]]);
plt.xlabel(xunits)
plt.ylabel(zunits)
plt.title('x average')


plt.subplot(1, 3, 3)
plt.imshow(average.mean(axis=2), aspect = 0.5,
           extent=[-options['windowUm'][2], options['windowUm'][2], 
                   -options['windowUm'][0], options['windowUm'][0]])
plt.xlabel(xunits)
plt.ylabel(zunits)
plt.title('y average')


fig.savefig(basepath +dirname + 'mean_profile.png')
#%% 
fig = plt.figure(figsize=(28,15));

plt.subplot(3, 4, (1,5))
plotPSF(latProfile[0],latProfile[1], latProfile[2], latProfile[3], pxPerUmLat,PSF.Counts, xunits)
plt.title("z average")
plt.subplot(3, 4, 9)
plt.hist(allbeads_data['FWHMlat'])

plt.subplot(3, 4, (2,6))
plotPSF(axProfile[0], axProfile[1],axProfile[2],axProfile[3],pxPerUmAx,PSF.Counts, zunits)
plt.title("xy average")
plt.subplot(3, 4, 10)
plt.hist(allbeads_data['FWHMax'])

plt.subplot(3, 4, (3,7))
plotPSF(centProfile[0], centProfile[1],centProfile[2],centProfile[3],pxPerUmLat,PSF.Counts, xunits)
plt.title("z max section")
plt.subplot(3, 4, 11)
plt.hist(allbeads_data['FWHMcent'])

plt.subplot(3, 4, (4,8))
plotPSF(maxProfile[0], maxProfile[1],maxProfile[2],maxProfile[3],pxPerUmAx,PSF.Counts, zunits)
plt.title("xy max")
plt.subplot(3, 4, 12)
plt.hist(allbeads_data['FWHMax'])

fig.savefig(basepath +dirname + 'fits.png')
