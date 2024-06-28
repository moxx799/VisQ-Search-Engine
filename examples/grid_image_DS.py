#libraries
import os
import time
import h5py
import argparse
import numpy as np
import pandas as pd
import multiprocessing
import skimage.io as io
from functools import partial
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from skimage.io import imsave, imread
import gc
from statistics import mean
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from skimage import exposure

dst='/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/GB_M0.1' #save path

window_size=75
window_step=50

#load in the first image 
# img_names=['DAPI.tif', 'Olig2.tif', 'Nestin.tif', 'CD45.tif'] #names of images to make the composite
img_names=['DAPI.tif', 'Sall1.tif', 'Tmem119.tif', 'P2ry12.tif'] #names of images to make the composite

#tumor set: DAPI, Olig2, Nestin, CD45
#microglia set: DAPI, Sall1, Tmem119, P2ry12



img_dst='/project/roysam/rwmills/data/glioblastoma/' #pull path 
#--------------------------------------------------------------------
#--------------------------------------------------------------------

img_list=[os.path.join(img_dst, i) for i in img_names] #make it the full path

img = imread(img_list[0] ) #read in a sample image 
image_shape = img.shape #get the size
height, width = image_shape[0], image_shape[1] 

composite = np.zeros((height, width, len(img_list))) # , dtype=np.uint16) #make a composite
for i, img in enumerate(img_list):
    print('Loading... ')
    channel_image = io.imread(img,as_gray=True)
    enhanced_image= exposure.adjust_log(channel_image, 1)
    print('...enchanced')
    composite[:, :, i] = enhanced_image #fill it in
# plt.imshow(composite[:,:,0:3])
# plt.show()
#
print('Created composite Image')
#------------crop the images----------------------------------------
cents=[] 
for y in range(0, height - window_size +1, window_step): 
    for x in range(0, width-window_size + 1, window_step):
        #window = composite[y:y+window_size, x:x+window_size] #gets the crop
        cx=int(x + window_size/2)
        cy=int(y + window_size/2)
        cents.append((cx,cy)) #might need to flip this
print('Extracted Centroids: ', len(cents))

def get_crop_no_pad(image, cen, crop_size):
    x_marg = int(crop_size / 2 )
    y_marg = int(crop_size / 2)
    try:
        # return image[cen[0] -x_marg : cen[0] + x_marg, cen[1] - y_marg : cen[1] + y_marg, :]
        return image[cen[1] -x_marg : cen[1] + x_marg, cen[0] - y_marg : cen[0] + y_marg, :]
    except:
        # return image[cen[0] -x_marg : cen[0] + x_marg, cen[1] - y_marg : cen[1] + y_marg]
        return image[cen[1] -x_marg : cen[1] + x_marg, cen[0] - y_marg : cen[0] + y_marg]
        
         
intensities = np.array([np.mean(get_crop_no_pad(composite, centroid, window_size), axis=(0, 1)) for centroid in cents])
    
intensities = intensities[:, 1:]  # we don't need DAPI for classification

labels = (intensities == intensities.max(axis=1)[:, None]).astype(int)
print('Created Pseudo-labels')

cells = [get_crop_no_pad(composite, cen, window_size ) for cen in cents]
print('Cropped Patches')
    

#now lets do the stuff for reid-------------------------------------------------------------------------
#make the folders
if os.path.exists(dst) is False:
    os.mkdir(dst)
if os.path.exists(os.path.join(dst, 'bounding_box_train')) is False:
    os.mkdir(os.path.join(dst, 'bounding_box_train'))
if os.path.exists(os.path.join(dst, 'bounding_box_test')) is False:
    os.mkdir(os.path.join(dst, 'bounding_box_test'))
if os.path.exists(os.path.join(dst, 'query')) is False:
    os.mkdir(os.path.join(dst, 'query'))
print('Created Folders')
#create naming schematic and save:
for i, x in enumerate(cells):
    #c_id = cell_ids[i][0]
    y=labels[i] #get the class ID
    c_id= np.where(y==1)[0][0]
    cent = str(cents[i][0]) + '_' + str(cents[i][1])  #get the atlas location
    loc =y #  e_labels[i][0]
    pid = str(c_id).zfill(3) # str(y) + str(loc).zfill(3) #get the cell identifier
    # img_name = pid + '_c' + camera + 's1_'
    #Define the folder
    if i % 2 ==0:
        dst_set = dst + '/bounding_box_test/'
    elif i % 11 == 0:
        dst_set = dst + '/query/'
    else:
        dst_set = dst + '/bounding_box_train/'
    #Define the name
    if dst_set== dst + '/bounding_box_test/':
        new_name = dst_set + pid + '_c' + cent + 's1_' + str(c_id).zfill(6) +'.tif'
    elif dst_set== dst + '/query/':
        new_name = dst_set + pid + '_c' + cent + 's1_' + str(c_id).zfill(6) +'.tif'
    else:
        new_name = dst_set + pid + '_c' + cent + 's1_' + str(c_id).zfill(6) + '.tif'
    #Extract rgb image:
    # if x.shape == (128, 128, 7):
        # imsave(new_name, x) #save
    if np.amax(x) > 0:
        imsave(new_name, x)
    else:
        print("Image is just black: ", x.shape, ', continuing... ')

# define custom transform function
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    '''for the brain data loader '''
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            # x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x =self.data[index].astype(np.uint8)
            x = self.transform(x)
        return x
    def __len__(self):
        return len(self.data)

preprocess = transforms.Compose([ transforms.ToTensor(),])
dataset = MyDataset(cells, transform=preprocess)
del cells
gc.collect() #free memory

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1, pin_memory=False,)


nimages = 0
mean = 0.0
var = 0.0
for i_batch, batch_target in enumerate(data_loader):
    batch = batch_target
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0)
    var += batch.var(2).sum(0)

mean /= nimages
var /= nimages
std = torch.sqrt(var)

print('mean: ', mean)
print('std: ', std)

    