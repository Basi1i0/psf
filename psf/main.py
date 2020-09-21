from numpy import all, asarray, array, where, exp, inf, mean
import numpy as np
from pandas import DataFrame
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def compute(im, options):
    beads, maxima, centers, smoothed = getCenters(im, options)
    return [getPSF(x, options) for x in beads], beads, maxima, centers, smoothed

def inside(shape, center, window):
    """
    Returns boolean if a center and its window is fully contained
    within the shape of the image on all three axes
    """
    return all([(center[i]-window[i] >= 0) & (center[i]+window[i] <= shape[i]) for i in range(0,3)])

def volume(im, center, window):
    if inside(im.shape, center, window):
        volume = im[(center[0]-window[0]):(center[0]+window[0]), (center[1]-window[1]):(center[1]+window[1]), (center[2]-window[2]):(center[2]+window[2])]
        volume = volume.astype('float64')
        #baseline = volume[[0,-1],[0,-1],[0,-1]].mean()
        #volume = volume - baseline
        #volume = volume/volume.max()
        return volume

def findBeads(im, window, thresh):
    smoothpoints = [np.ceil(x/10) for x in window];
    
    smoothed = gaussian(im, smoothpoints, output=None, mode='nearest', cval=0, multichannel=None)#window
    centers = peak_local_max(smoothed, min_distance=3, threshold_rel=thresh, exclude_border=True)
    return centers, smoothed.max(axis=0)

def keepBeads(im, window, centers, options):
    #centersM = asarray([[x[0]/options['pxPerUmAx'], x[1]/options['pxPerUmLat'], x[2]/options['pxPerUmLat']] for x in centers])
    #centerDists = [nearest(x,centersM) for x in centersM]
    #
    centers = asarray([[x[0], x[1], x[2] ] for x in centers])
    centerDists = [nearest(x,centers) for x in centers]
    keep = where([x>((window[1]**2 + window[2]**2)/2)**0.5 for x in centerDists])
    
    centers = centers[keep[0],:]
    keep = where([inside(im.shape, x, window) for x in centers])
    return centers[keep[0],:]

def getCenters(im, options):
    
    im_mean = mean(im, 0)
    plt.imshow(im_mean);
    plt.xlim([0, im_mean.shape[0]])
    plt.ylim([im_mean.shape[1], 0])
    plt.axis('off');
    
    
    window = [options['windowUm'][0]*options['pxPerUmAx'], options['windowUm'][1]*options['pxPerUmLat'], options['windowUm'][2]*options['pxPerUmLat']]
    window = [int(round(x)) for x in window]
    centers, smoothed = findBeads(im, window, options['thresh'])
    
    print(centers.shape[0], 'beads above threshold')
    
    keep = np.repeat(True, centers.shape[0])
    for i in range(0, centers.shape[0]):
        inearest = argnearest(centers[i], centers);
        if(dist(centers[i], centers[inearest]) < 0.2*((window[1]**2 + window[2]**2)/2)**0.5):
            if(np.abs(centers[i, 0] - centers[inearest, 0]) > 3):
                keep[i] = im[centers[i,0], centers[i,1], centers[i,2]] > im[centers[inearest,0], centers[inearest,1], centers[inearest,2]]
                keep[inearest] = im[centers[i,0], centers[i,1], centers[i,2]] <= im[centers[inearest,0], centers[inearest,1], centers[inearest,2]]
    centers = centers[keep];
    
    
    print(centers.shape[0], 'beads above threshold, z filtered')

    plt.plot(centers[:, 2], centers[:, 1], 'w+', ms=5, label = 'threshold filter ' + str(options['thresh']) +'*max');    
    
    centers = keepBeads(im, window, centers, options)
    print(centers.shape[0], 'beads isoleted in window and fit in volume')

    plt.plot(centers[:, 2], centers[:, 1], 's', ms=window[1]+window[2], mfc='none', label = 'size filter ' + str(window[1]+window[2]) + ' pix')     
    
    beads = [volume(im, x, window ) for x in centers]
    
    m = np.array([np.mean(b[b > np.max(b)*options['thresh'] ]) for b in beads])
    keep = where((m < np.median(m)*(1 + options['countsvar'])) * (m > (1-options['countsvar'])*np.median(m)))
        
    centers = centers[keep[0],:]
    beads = [volume(im, x, window ) for x in centers]
    maxima = [im[x[0], x[1], x[2]] for x in centers]

    print(centers.shape[0], 'beads pass intensity criterion' )
        
    plt.plot(centers[:, 2], centers[:, 1], 'o', ms=(window[1]+window[2])*0.95, mfc='none', label = 'intensity filter m*(1±'  + str(options['countsvar'])  +')'); 
    
    plt.legend()    
    
    return beads, maxima, centers, smoothed

def getPSF(bead, options):
    latProfile, axProfile, centProfile, maxProfile = getSlices(bead)
    latFit = fit(latProfile,options['pxPerUmLat'])
    centFit = fit(centProfile,options['pxPerUmLat'])
    axFit = fit(axProfile,options['pxPerUmAx'])
    maxFit = fit(maxProfile,options['pxPerUmAx'])
    data = DataFrame([latFit[3], axFit[3], centFit[3], maxFit[3]],index = ['FWHMlat', 'FWHMax', 'FWHMcent', 'FWHMmax']).T
    return data, latFit, axFit, centFit, maxFit

def getSlices(average):
    latProfile = (average.mean(axis=0).mean(axis=1) + average.mean(axis=0).mean(axis=1))/2
    axProfile = (average.mean(axis=1).mean(axis=1) + average.mean(axis=2).mean(axis=1))/2
    centProfile = (average[int(average.shape[0]/2),:,:].mean(axis=0) + average[int(average.shape[0]/2),:,:].mean(axis=1))/2
    maxProfile = (average.max(axis=1).max(axis=1) + average.max(axis=2).max(axis=1))/2
    return latProfile, axProfile, centProfile, maxProfile

def fit(yRaw,scale):
    y = yRaw; # - (yRaw[0]+yRaw[-1])/2
    x = (array(range(y.shape[0])) - y.shape[0]/2)
    x = (array(range(y.shape[0])) - y.shape[0]/2)
    popt, pcov = curve_fit(gauss, x, y, p0 = [ np.max(y)- np.median(y), 0, np.max(x)/10, np.median(y) ], #[a, mu, sigma, b]: a*exp(-(x-mu)**2/(2*sigma**2))+b
                           bounds = ([0,np.min(x), 0, 0], [np.max(y), np.max(x), 2*np.max(x), np.max(y)]), maxfev = 1e4)
    FWHM = 2.355*popt[2]/scale
    a = popt[0]
    b = popt[3]
    yFit = gauss(x, *popt)
    return x, y-popt[3], yFit-popt[3], FWHM, a, b

def plotPSF(x,y,yFit,FWHM,scale,Max, unit):
    plt.plot(x.astype(float)/scale,yFit/yFit.max(), lw=2);
    plt.plot(x.astype(float)/scale,y/yFit.max(),'ok');
    plt.xlim([-x.shape[0]/2/scale, x.shape[0]/2/scale])
    plt.ylim([0, 1.1])
    plt.xlabel('Distance (' + unit +')')
    plt.ylabel('Norm. intensity')
    plt.annotate(('FWHM %.2f ' + unit) % FWHM,xy=(x.shape[0]/5/scale, .95), size=14)
    plt.annotate('Counts %2.0f' % Max,xy=(x.shape[0]/5/scale, .9), size=14)


def plotAvg(i):
    plt.figure(figsize=(5,5));
    plt.imshow(average[i], vmin=0, vmax=.9);
    if i==average.shape[0]/2:
        plt.plot(average.shape[1]/2, average.shape[2]/2, 'r.', ms=10);
    plt.xlim([0, average.shape[1]])
    plt.ylim([average.shape[2], 0])
    plt.axis('off');

def plotAvg(i):
    plt.figure(figsize=(5,5));
    plt.imshow(average[i], vmin=0, vmax=.9);
    if i==average.shape[0]/2:
        plt.plot(average.shape[1]/2, average.shape[2]/2, 'r.', ms=10);
    plt.xlim([0, average.shape[1]])
    plt.ylim([average.shape[2], 0])
    plt.axis('off');

def dist(x,y):
    return ((x - y)**2)[1:].sum()**(.5)

def nearest(x,centers):
    z = [dist(x,y) for y in centers if not (x == y).all()]
    return abs(array(z)).min(axis=0)

def argnearest(x,centers):
    z = [dist(x,y) for y in centers if not (x == y).all()]
    return abs(array(z)).argmin(axis=0)

def gauss(x, a, mu, sigma, b):
    return a*exp(-(x-mu)**2/(2*sigma**2))+b
