import numpy as np
import torch
import os
import os.path
from os import listdir
from PIL import Image as PImage
import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt


#We apply the laplacian operator to detect edges (LoG), second derivative filter. 
#First, we use gaussian filter to smooth the picture.
#Second, we apply the Laplacian operator to detect the edges.

def LoG(image, gaussian_ksize, laplacian_ksize):
    """
    Filter, performs gaussian smoothness and laplacian operator for edge detection

    #Arguments:
        image (type:`numpy.array`)
        gaussian_ksize: to smooth the picure
        laplacian_ksize: to detect edges

    #Returns:
        `np.array`: input image for the clustering

    """

    blurred = cv2.GaussianBlur(image, (gaussian_ksize, gaussian_ksize), sigmaX = 0) #the larger the sigma the less noisy the img
    laplacian = cv2.Laplacian(blurred, laplacian_ksize)

    return laplacian

#Input of AE for clustering. Black 'n' white images detecting white lines
def get_lines(image, gaussian_ksize = 3, laplacian_ksize = 5, min_width = 25, min_height = 20, output_shape_hw = (28, 28)):
    """
    Given a  image, returns a gray (float0-255) image with its lines.
    #Arguments:
        image
        gaussian_ksize
        laplacian_ksize
        min_width
        min_height
        output_shape_hw: 'same' or tuple of ints (h, w)
    #Returns:
        np.array: same dimension as input
    """    
    # smooth and find edges
    laplacian = LoG(image, gaussian_ksize = gaussian_ksize, laplacian_ksize = laplacian_ksize)
    
    # preserve horizontal lines
    h_dilate = cv2.dilate(laplacian, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(min_width), 1))
    h_lines = cv2.morphologyEx(h_dilate, cv2.MORPH_OPEN, h_kernel, iterations = 1)
    _, h_lines_bis = cv2.threshold(h_lines, 1, 255, cv2.THRESH_BINARY)

    # preserve vetical lines
    v_dilate = cv2.dilate(laplacian, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(min_height)))
    v_lines = cv2.morphologyEx(v_dilate, cv2.MORPH_OPEN, v_kernel, iterations = 1)
    _, v_lines_bis = cv2.threshold(v_lines, 1, 255, cv2.THRESH_BINARY)
    
    # join all
    all_lines = cv2.addWeighted(h_lines_bis, 1, v_lines_bis, 1, 0)

    # to black'n'white picture
    thr, all_lines_bw = cv2.threshold(all_lines, 5, 255, cv2.THRESH_BINARY)
    lines = cv2.dilate(all_lines_bw, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations = 1)
    
    # resize
    if output_shape_hw == 'same':
        lines_output = lines
    else:
        lines_output = cv2.resize(lines, (output_shape_hw[1], output_shape_hw[0]))
    
    return lines_output.astype('float32') #lines_output.astype('unit.8')