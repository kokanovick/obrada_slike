# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:49:48 2023

@author: SW6
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import h5py
import cv2
from PIL import Image, ImageChops
from skimage import measure, img_as_ubyte, io
from skimage.filters import threshold_otsu
from skimage.color import label2rgb, rgb2gray
from collections import namedtuple

study_df = pd.read_csv(os.path.join('study_list.csv'))

def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path
with h5py.File('lab_petct_vox_5.00mm.h5', 'r') as f:
    for dset in traverse_datasets(f):
        print('Path:', dset)
        print('Shape:', f[dset].shape)
        print('Data type:', f[dset].dtype)


with h5py.File(os.path.join('lab_petct_vox_5.00mm.h5'), 'r') as p_data:
    ct_images = p_data['ct_data'].items()
    pet_images = p_data['pet_data'].values()
    lab_images = p_data['label_data'].values()
    fig, sb_mat = plt.subplots(7, 4, figsize=(10, 25))
    (ax1s, ax2s, ax3s, ax4s) = sb_mat.T
    for c_ax1, c_ax2, c_ax3, c_ax4, (p_id, ct_img), pet_img, lab_img in zip(ax1s, ax2s, ax3s, ax4s, ct_images, pet_images, lab_images):
        
        ct_image = np.mean(ct_img, 1)[::-1]
        c_ax1.imshow(ct_image, cmap = 'bone')
        c_ax1.axis('off')
        
        pet_proj = np.max(pet_img, 1)[::-1]
        pet_image = np.sqrt(np.max(pet_img, 1).squeeze()[::-1,:])
        c_ax2.imshow(pet_image, cmap = 'bone')
        c_ax2.axis('off')
         
        label = np.mean(lab_img, 1)[::-1]
        c_ax3.imshow(label, cmap='bone')
        c_ax3.axis('off')
        
        result = cv2.multiply(pet_image, label)
        c_ax4.imshow(result, cmap='bone')
        c_ax4.axis('off')

for x in range(sb_mat.shape[0]):
        for y in range(sb_mat.shape[1]):
            extent = sb_mat[x, y].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(str(x) + str(y) + '.png', bbox_inches=extent.expanded(1.1, 1.2))
            
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else: 
        # Failed to find the borders, convert to "RGB"        
        return trim(im.convert('RGB'))

im = Image.open("03.png")
im = trim(im)
im = im.save("First patient final.png")

im1 = Image.open("13.png")
im1 = trim(im1)
im1 = im1.save("Second patient final.png")

im2 = Image.open("23.png")
im2 = trim(im2)
im2= im2.save("Third patient final.png")

im3 = Image.open("33.png")
im3 = trim(im3)
im3 = im3.save("Fourth patient final.png")

im4 = Image.open("43.png")
im4 = trim(im4)
im4 = im4.save("Fifth patient final.png")

im5 = Image.open("53.png")
im5 = trim(im5)
im5 = im5.save("Sixth patient final.png")

im6 = Image.open("63.png")
im6 = trim(im6)
im6 = im6.save("Seventh patient final.png")

test = img_as_ubyte(rgb2gray(io.imread("First patient final.png")))
threshold = threshold_otsu(test)
label_image = measure.label(test < threshold, connectivity = test.ndim)
props = measure.regionprops_table(label_image, test, properties=['area', 'eccentricity'])

test2 = img_as_ubyte(rgb2gray(io.imread("Second patient final.png")))
threshold2 = threshold_otsu(test2)
label_image2 = measure.label(test2 < threshold2, connectivity = test2.ndim)
props2 = measure.regionprops_table(label_image2, test2, properties=['area', 'eccentricity'])

test3 = img_as_ubyte(rgb2gray(io.imread("Fifth patient final.png")))
threshold3 = threshold_otsu(test3)
label_image3 = measure.label(test3 < threshold3, connectivity = test3.ndim)
props3 = measure.regionprops_table(label_image3, test3, properties=['area', 'eccentricity'])