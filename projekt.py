# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:49:48 2023

@author: SW6
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import h5py
import cv2
from PIL import Image, ImageChops
from skimage import measure, img_as_ubyte, io
import seaborn as sns
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from scipy import stats
import statsmodels.formula.api as smf

study_df = pd.read_csv(os.path.join('study_list.csv'))

def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): 
                yield (path, item)
            elif isinstance(item, h5py.Group): 
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
    fig, sb_mat = plt.subplots(7, 4, figsize=(10,25))
    (ax1s, ax2s, ax3s, ax4s) = sb_mat.T
    for c_ax1, c_ax2, c_ax3, c_ax4, (p_id, ct_img), pet_img, lab_img in zip(ax1s, ax2s, ax3s, ax4s, ct_images, pet_images, lab_images):
        
        ct_image = np.mean(ct_img, 1)[::-1]
        c_ax1.imshow(ct_image, cmap = 'bone')
        c_ax1.axis('off')
        
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
            fig.savefig('images/' + str(x) + str(y) + '.png', bbox_inches=extent.expanded(1.1, 1.2))
            
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:        
        return trim(im.convert('RGB'))
for i in range(0,7):
    im = Image.open("images/" + str(i) + "3.png")
    im = trim(im)
    im = im.save("images/Final" + str(i) + ".png")

areas = []
eccentricity = []
for j in range(0,7):
    test = img_as_ubyte(rgb2gray(io.imread("images/Final" + str(j) + ".png")))
    threshold = threshold_otsu(test)
    label_image = measure.label(test > threshold, connectivity = test.ndim)
    props = measure.regionprops_table(label_image, test, properties=['area', 'eccentricity'])
    areas.append(props['area'][0]) 
    eccentricity.append(props['eccentricity'][0])

with h5py.File('patient_images_lowres.h5', 'r') as f:
    for dset in traverse_datasets(f):
        print('Path:', dset)
        print('Shape:', f[dset].shape)
        print('Data type:', f[dset].dtype)

with h5py.File(os.path.join('patient_images_lowres.h5'), 'r') as p_data2:
    ct_images2 = p_data2['ct_data'].items()
    fig, sb_mat2 = plt.subplots(18,1, figsize=(10,55))
    (ax1s2) = sb_mat2.T
    for c_ax12, (p_id2, ct_img2) in zip(ax1s2, ct_images2):      
        ct_image2 = np.mean(ct_img2, 1)[::-1]
        c_ax12.imshow(ct_image2, cmap = 'bone')
        c_ax12.axis('off')
        

for a in range(sb_mat2.shape[0]):
    extent2 = sb_mat2[a].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('images/PET' + str(a) + '.png', bbox_inches=extent2.expanded(1.1, 1.2))
    
for b in range(0,18):
    im = Image.open("images/PET" + str(b) + ".png")
    im = trim(im)
    im = im.save("images/Trim" + str(b) + ".png")

mean = []
for c in range(0,18):
    image = img_as_ubyte(rgb2gray(io.imread("images/Trim" + str(c) + ".png")))
    threshold2 = threshold_otsu(image)
    label_image2 = measure.label(image > threshold2, connectivity = image.ndim)
    props2 = measure.regionprops_table(label_image2, image, properties=['area', 'intensity_mean'])
    plt.imshow(label_image2)
    mean.append(np.mean(label_image2))

all_patients = ['STS_002', 'STS_003', 'STS_005',  'STS_011','STS_012', 'STS_015', 'STS_020', 'STS_021', 'STS_022', 'STS_023',  'STS_024',  'STS_029', 'STS_031', 'STS_034', 'STS_037', 'STS_039', 'STS_041', 'STS_048']
df = pd.DataFrame(mean, index=all_patients, columns=['Average PET scan value'])
df['Area Volume'] = [areas[0], areas[1], areas[2], 0, areas[3], 0, 0, areas[4], 0, areas[5], 0, 0, areas[6], 0, 0, 0, 0, 0]
df['Eccentricity'] = [eccentricity[0], eccentricity[1], eccentricity[2], 0, eccentricity[3], 0, 0, eccentricity[4], 0, eccentricity[5], 0, 0, eccentricity[6], 0, 0, 0, 0, 0]
#'No', 'No','No','No','Yes','No','Yes','Yes','Yes', 'Yes','No','No','Yes','Yes','Yes','Yes','No','No'
# 0 - no, 1 - yes
df['Recurrence'] = [0,0,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,0]
df.to_csv('Patient_analysis.csv')
# df['Area Volume'].hist(by = df['Recurrence'], sharex = True)
y = df.Recurrence                          # M or B 
x = df.drop('Recurrence', axis=1)
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:18]],axis=1)
data = pd.melt(data,id_vars="Recurrence",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="Recurrence", data=data,split=True, inner="quart")
plt.xticks(rotation=90)
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="Recurrence", data=data)
plt.xticks(rotation=90)
df['Area Volume'].hist(by = df['Recurrence'], sharex = True)

corr_x = df['Area Volume'].values
corr_x1 = df['Eccentricity']
corr_x2 = df['Average PET scan value']
corr_y = df['Recurrence'].values
corr = stats.pointbiserialr(corr_x, corr_y)
print('Pearsons correlation: %.3f' % corr[0] + ' with p-value: %.3f' % corr[1])
corr1 = stats.pointbiserialr(corr_x1, corr_y)
print('Pearsons correlation: %.3f' % corr1[0] + ' with p-value: %.3f' % corr1[1] )
corr2 = stats.pointbiserialr(corr_x2, corr_y)
print('Pearsons correlation: %.3f' % corr2[0] + ' with p-value: %.3f' % corr2[1] )
newdf = df.copy()
newdf.drop(columns= ['Average PET scan value', 'Area Volume'])
model = smf.ols(formula='Eccentricity ~ Recurrence', data=newdf).fit()
print(model.summary())