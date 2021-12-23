#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 10:33:27 2021

@author: marcelo
"""
import numpy as np
#import matplotlib.pyplot as plt
#from skimage.io import imread, imshow
#import cv2
from skimage.morphology import area_opening, thin, area_closing
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.color import label2rgb
from my_canny import my_canny
from skimage.exposure import rescale_intensity
from scipy.ndimage.filters import convolve

class isec():
    
    def __init__(self, f_order=11, step=0.05, nit=6):
        
        self.f_order = f_order
        self.step = step
        self.nit = nit
        
    def segment(self, img):
        
        # Get input image dimensions
        M, N, C = img.shape
        segments = np.zeros((M,N), dtype=np.float32)
        
        # Convert to gray
        if C >=2:
            gray_img = np.asarray(rgb2gray(img), dtype=np.float32)
        else:
            gray_img = np.asarray(img, dtype=np.float32)
        
        # Create a Canny edge detector
        edge_detector = my_canny(gray_img)        
        
        # Iteractive segmentation
        for it in range(self.nit):
            
            # Update treshold
            k = 0.1 + it*self.step
            
            # Select edges
            ed = edge_detector.get_edges(high_threshold=k, it=it)
            
            # Create filter kernel
            S = max(3, self.f_order - it*2)
            d_filter = np.ones([S, S], dtype=np.float32)
            
            # Edge density filtering
            ed_den = convolve(ed.astype(np.float32), d_filter)
            
            # Normalize
            ed_den =  rescale_intensity(ed_den, out_range=(0.0, 1.0))
            #cv2.normalize(ed_den, ed_den, 1.0 , 0.0 ,cv2.NORM_MINMAX)
            
            # replace the edges expanded by the filter
            ed_den = replace_edges(ed_den, S)
            
            # Threshold edge density progressively
            ed_den = (ed_den > 0.05*it) 
            
            # Clean up some noise
            ed_den = area_closing(ed_den, area_threshold=10+(it*5),connectivity=1)
            
            # Separate segment's shells and kernels
            im_shell, im_kernel = shell_kernel(ed_den)
            
            # Store the segments
            segments = (segments + im_shell)>0            
            
        # post-processing
        segments = area_closing(segments, area_threshold=80, connectivity=1)
        segments = replace_edges(segments, self.f_order)
        n_shell, n_kernel = shell_kernel(segments)
        segments = segments * (1-n_kernel)
        segments = area_opening(segments, area_threshold=80, connectivity=2)
        
        
        display_segs = np.asarray(np.dstack((segments, np.zeros_like(segments), gray_img)), dtype=np.float32)
        
        # assign labels to segments
        labels, num = label((1-segments).astype(int), connectivity=1, return_num=True)  
        labels = remove_borders(labels)
        
        display_labels = label2rgb(labels, gray_img, bg_label=0)
        
        return display_segs, display_labels, num
              

def shell_kernel(img):
    img = img.astype(np.float32)
    d_filter = np.ones([3,3], dtype=np.float32)
    im_kernel = (convolve(img, d_filter) == 9.0).astype(np.float32)
    im_shell = abs(img - im_kernel)
    
    return im_shell, im_kernel

def replace_edges(ed_den, S):
    P = int((S-1)/2)    
    ed = ed_den > 0
    ed = np.pad(ed, pad_width=S, mode='reflect')
    ed = thin(ed, P).astype(np.float32)    
    ed_den = ed_den * ed[S:-S, S:-S]
    
    return ed_den

def remove_borders(labels):
    # Remove the borders (zero-valued pixels) that separate the segments
    labels3 = np.pad(labels, pad_width=1, mode='reflect')
    M, N = labels3.shape
    neighs = np.zeros((M-2,N-2,8), dtype=int)
    
    neighs[:,:,0] = labels3[0:M-2,0:N-2]
    neighs[:,:,1] = labels3[0:M-2,1:N-1]
    neighs[:,:,2] = labels3[0:M-2,2:N]
    neighs[:,:,3] = labels3[1:M-1,2:N]
    neighs[:,:,4] = labels3[2:M,2:N]
    neighs[:,:,5] = labels3[2:M,1:N-1]    
    neighs[:,:,6] = labels3[2:M,0:N-2]
    neighs[:,:,7] = labels3[1:M-1,0:N-2]
    
    newlabels = np.amax(neighs, axis=2).astype(np.float32)
    newlabels = newlabels * (labels==0).astype(np.float32)
    labels = labels + newlabels.astype(int)
    
    return labels
