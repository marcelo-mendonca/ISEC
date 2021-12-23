#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 12:56:07 2021

@author: marcelo
"""

import numpy as np
import scipy.ndimage as ndi
from skimage.filters import gaussian


class my_canny():
    
    def __init__(self, image, sigma=1., mask=None, mode='constant', cval=0.0, nit=6):
        
        self.magnitudes = []
        self.local_maximas = []
        
        for i in range(nit):            
        
            # Image filtering
            smoothed, eroded_mask = _preprocess(image, mask, sigma+i, mode, cval)
        
            # Gradient magnitude estimation
            jsobel = ndi.sobel(smoothed, axis=1)
            isobel = ndi.sobel(smoothed, axis=0)
            magnitude = np.hypot(isobel, jsobel)
            
            # Non-maximum suppression
            local_maxima = _get_local_maxima(isobel, jsobel, magnitude, eroded_mask)
            self.magnitudes.append(magnitude)
            self.local_maximas.append(local_maxima)
        
    def get_edges(self, high_threshold, low_threshold=None, it=1):
        
        if low_threshold is None:
            low_threshold = high_threshold * .4
        
        magnitude = self.magnitudes[it]
        local_maxima = self.local_maximas[it]
        
        # Double thresholding and edge traking
        low_mask = local_maxima & (magnitude >= low_threshold)
    
        #
        # Segment the low-mask, then only keep low-segments that have
        # some high_mask component in them
        #
        strel = np.ones((3, 3), bool)
        labels, count = ndi.label(low_mask, strel)
        if count == 0:
            return low_mask
    
        high_mask = local_maxima & (magnitude >= high_threshold)
        nonzero_sums = np.unique(labels[high_mask])
        good_label = np.zeros((count + 1,), bool)
        good_label[nonzero_sums] = True
        output_mask = good_label[labels]
        return output_mask



def _preprocess(image, mask, sigma, mode, cval):
    """Generate a smoothed image and an eroded mask.
    The image is smoothed using a gaussian filter ignoring masked
    pixels and the mask is eroded.
    Parameters
    ----------
    image : array
        Image to be smoothed.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    mode : str, {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'.
    Returns
    -------
    smoothed_image : ndarray
        The smoothed array
    eroded_mask : ndarray
        The eroded mask.
    Notes
    -----
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """

    gaussian_kwargs = dict(sigma=sigma, mode=mode, cval=cval,
                           preserve_range=False)
    if mask is None:
        # Smooth the masked image
        smoothed_image = gaussian(image, **gaussian_kwargs)
        eroded_mask = np.ones(image.shape, dtype=bool)
        eroded_mask[:1, :] = 0
        eroded_mask[-1:, :] = 0
        eroded_mask[:, :1] = 0
        eroded_mask[:, -1:] = 0
        return smoothed_image, eroded_mask

    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    # Compute the fractional contribution of masked pixels by applying
    # the function to the mask (which gets you the fraction of the
    # pixel data that's due to significant points)
    bleed_over = (
        gaussian(mask.astype(float), **gaussian_kwargs) + np.finfo(float).eps
    )

    # Smooth the masked image
    smoothed_image = gaussian(masked_image, **gaussian_kwargs)

    # Lower the result by the bleed-over fraction, so you can
    # recalibrate by dividing by the function on the mask to recover
    # the effect of smoothing from just the significant pixels.
    smoothed_image /= bleed_over

    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    s = ndi.generate_binary_structure(2, 2)
    eroded_mask = ndi.binary_erosion(mask, s, border_value=0)

    return smoothed_image, eroded_mask


def _set_local_maxima(magnitude, pts, w_num, w_denum, row_slices,
                      col_slices, out):
    """Get the magnitudes shifted left to make a matrix of the points to
    the right of pts. Similarly, shift left and down to get the points
    to the top right of pts.
    """
    r_0, r_1, r_2, r_3 = row_slices
    c_0, c_1, c_2, c_3 = col_slices
    c1 = magnitude[r_0, c_0][pts[r_1, c_1]]
    c2 = magnitude[r_2, c_2][pts[r_3, c_3]]
    m = magnitude[pts]
    w = w_num[pts] / w_denum[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[r_1, c_1][pts[r_0, c_0]]
    c2 = magnitude[r_3, c_3][pts[r_2, c_2]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    out[pts] = c_plus & c_minus

    return out


def _get_local_maxima(isobel, jsobel, magnitude, eroded_mask):
    """Edge thinning by non-maximum suppression.
    Finds the normal to the edge at each point using the arctangent of the
    ratio of the Y sobel over the X sobel - pragmatically, we can
    look at the signs of X and Y and the relative magnitude of X vs Y
    to sort the points into 4 categories: horizontal, vertical,
    diagonal and antidiagonal.
    Look in the normal and reverse directions to see if the values
    in either of those directions are greater than the point in question.
    Use interpolation (via _set_local_maxima) to get a mix of points
    instead of picking the one that's the closest to the normal.
    """
    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)

    eroded_mask = eroded_mask & (magnitude > 0)

    # Normals' orientations
    is_horizontal = eroded_mask & (abs_isobel >= abs_jsobel)
    is_vertical = eroded_mask & (abs_isobel <= abs_jsobel)
    is_up = (isobel >= 0)
    is_down = (isobel <= 0)
    is_right = (jsobel >= 0)
    is_left = (jsobel <= 0)
    #
    # --------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(magnitude.shape, bool)
    # ----- 0 to 45 degrees ------
    # Mix diagonal and horizontal
    pts_plus = is_up & is_right
    pts_minus = is_down & is_left
    pts = ((pts_plus | pts_minus) & is_horizontal)
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    local_maxima = _set_local_maxima(
        magnitude, pts, abs_jsobel, abs_isobel,
        [slice(1, None), slice(-1), slice(1, None), slice(-1)],
        [slice(None), slice(None), slice(1, None), slice(-1)],
        local_maxima)
    # ----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts = ((pts_plus | pts_minus) & is_vertical)
    local_maxima = _set_local_maxima(
        magnitude, pts, abs_isobel, abs_jsobel,
        [slice(None), slice(None), slice(1, None), slice(-1)],
        [slice(1, None), slice(-1), slice(1, None), slice(-1)],
        local_maxima)
    # ----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = is_down & is_right
    pts_minus = is_up & is_left
    pts = ((pts_plus | pts_minus) & is_vertical)
    local_maxima = _set_local_maxima(
        magnitude, pts, abs_isobel, abs_jsobel,
        [slice(None), slice(None), slice(-1), slice(1, None)],
        [slice(1, None), slice(-1), slice(1, None), slice(-1)],
        local_maxima)
    # ----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts = ((pts_plus | pts_minus) & is_horizontal)
    local_maxima = _set_local_maxima(
        magnitude, pts, abs_jsobel, abs_isobel,
        [slice(-1), slice(1, None), slice(-1), slice(1, None)],
        [slice(None), slice(None), slice(1, None), slice(-1)],
        local_maxima)

    return local_maxima


        
        