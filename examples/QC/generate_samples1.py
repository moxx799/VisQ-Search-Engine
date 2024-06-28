import torch 
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib 
matplotlib.use('Agg')

import os
import os.path as osp 
from torch.utils.data import DataLoader
import sys 
os.chdir('/project/roysam/rwmills/repos/cluster-contrast-reid/') 
print(os.getcwd())
# print(os.listdir())
import os #RWM
print('We are here: ', os.getcwd())
sys.path.append('/project/roysam/rwmills/repos/cluster-contrast-reid/')
sys.path.append('/project/roysam/rwmills/repos/cluster-contrast-reid/clusterconstrast')

from clustercontrast import datasets
import h5py 
import numpy as np 
from skimage.io import imread 
import matplotlib.pyplot as plt 
import seaborn as sn
from clustercontrast.utils.data import transforms as T
# import transforms as T
# import Preprocessor 
from clustercontrast.utils.data.preprocessor import Preprocessor
# import matplotlib.pyplot as plt % matplotlib inline
import random 
import pandas as pd
from skimage.transform import rescale, resize, downscale_local_mean
import scipy
import seaborn as sns



#In[]: set infp

dst_path= '/project/roysam/rwmills/repos/cluster-contrast-reid/results/neuron_p+s'

if os.path.exists(dst_path) is False: 
    os.mkdir(dst_path) #create if doesn't exist 
    print('Creating folder... ')


manual=True
#In[] Set 1: 

dn= 'neuron_processes' # 'Sham_G2_BR22_HC_13L' # internal1.0_class' #'cortex1.3'#'Lin0'#  'single_v_double_new'# 'Lin0' 'internal1.8_neuronal' #'internal1.0_class'#'internal1.6_onlyRegions'
dn2= 'neuron_processes_S10'

data_size = 75 # 175 #128 
try: 
    feats = h5py.File('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/' + dn + '/feature_results.h5', 'r') 
except: 
    feats = h5py.File('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/' + dn2 + '/feature_results.h5', 'r') 
print(feats.keys() )
h5filename = '/project/roysam/rwmills/data/brain/h5_files/Select_RegionMasked_50.h5'
#26982 in gallery 
#5418 in query 
brain_img_orig  = imread('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/atlas_borders.tif')
brain_img_orig = (brain_img_orig > 0)*256
#imread('/project/roysam/rwmills/data/brain/copy_dapi.tif') # /project/roysam/rwmills/data/brain/S1_R1C1.tif')
classids = ['Parvalbumin', 'CNPase','GFAP', 'NeuN', 'Olig2','Tyrosine','S100']
cell_types = ['DAPI', 'HISTONE','NeuN', 'S100', 'Olig2','Iba1','RECA1']
print(brain_img_orig.shape) 
#----------------------------------------------------------------------------------------------------------------
dist = feats['distmat']
gal = feats['gallery_features'] 
q = feats['query_features']

#7 channel data 

my_file = open(os.path.join('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data', dn, "mean_std.txt"), "r")
# reading the file
data = my_file.read()
# replacing end splitting the text 
# when newline ('\n') is seen.
data_into_list = data.split("\n")
my_file.close()
mean_array = data_into_list[0]
mean_array = mean_array.split()
# print(mean_array)
mean_array = [float(i) for i in mean_array]
std_array = data_into_list[1]
std_array =std_array.split()
std_array = [float(i) for i in std_array]
print('Mean and Std: ', mean_array, std_array)
    
def get_data(name, dataset_name , data_dir, height, width, mean_array, std_array, batch_size, workers):
    # root = osp.join(data_dir, name)
    # dataset = datasets.create(name, root)
    if 'brain' in name: 
        root = osp.join(data_dir, dataset_name) #RWM 
        dataset = datasets.create(name, root)
    else: 
        root = osp.join(data_dir, name)
        dataset = datasets.create(name, root)
    normalizer = T.Normalize(mean=mean_array, std=std_array) 
    # normalizer = T.Normalize(mean=[0.1012, 0.1008, 0.0997, 0.0637, 0.0404, 0.0986, 0.0982],
                             # std=[0.2360, 0.2351, 0.2339, 0.1883, 0.1514, 0.2325, 0.2315])
    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return dataset, test_loader
try: 
    dataset, test_loader = get_data('brain',dn  , '/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data', data_size,data_size, 
                                mean_array, std_array, 1, 1)
except: 
    dataset, test_loader = get_data('brain',dn2  , '/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data', data_size,data_size, 
                                mean_array, std_array, 1, 1)
# mask = np.load('/project/roysam/rwmills/data/brain/smallAllMasks.npy') 

#for 7 channel - good bbx location 
centroid_Q=[] 
for i, (fpath, fpid, _) in enumerate(dataset.query) : 
    # i_id = fpath.split('s1_')[1].split('.tif')[0]
    # actual = TCells.iloc[int(i_id)].tolist()[0] 
    # numbers_list = infos.iloc[actual].tolist() 
    # centroid_Q.append(numbers_list)
    cy, cx = fpath.split('_c')[1].split('s')[0].split('_')
    centroid_Q.append([int(cx), int(cy)])

centroid_G=[] 
for i, (fpath, fpid, _) in enumerate(dataset.gallery) : 
    # i_id = fpath.split('s1_')[1].split('.tif')[0]
    # actual = TCells.iloc[int(i_id)].tolist()[0] 
    # numbers_list = infos.iloc[actual].tolist() 
    # centroid_Q.append(numbers_list)
    cy, cx = fpath.split('_c')[1].split('s')[0].split('_')
    centroid_G.append([int(cx), int(cy)])
print('max g: ', max(centroid_G)) 

#In[] Set 2 

dn = 'neuron_soma' # neuronal2'
dn2= 'neuron_soma_S10'

data_size = 75 #128 
feats = h5py.File('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/' + dn2 + '/feature_results.h5', 'r') 
print(feats.keys() )
h5filename = '/project/roysam/rwmills/data/brain/h5_files/Select_RegionMasked_50.h5'
#26982 in gallery 
#5418 in query 
brain_img_orig  = imread('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/atlas_borders.tif')
brain_img_orig = (brain_img_orig > 0)*256
#imread('/project/roysam/rwmills/data/brain/copy_dapi.tif') # /project/roysam/rwmills/data/brain/S1_R1C1.tif')
classids = ['Parvalbumin', 'CNPase','GFAP', 'NeuN', 'Olig2','Tyrosine','S100']
cell_types = ['DAPI', 'HISTONE','NeuN', 'S100', 'Olig2','Iba1','RECA1']
print(brain_img_orig.shape) 
#----------------------------------------------------------------------------------------------------------------
dist2 = feats['distmat']
gal2 = feats['gallery_features'] 
q2 = feats['query_features']

#7 channel data 

my_file = open(os.path.join('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data', dn, "mean_std.txt"), "r")
# reading the file
data = my_file.read()
# replacing end splitting the text 
# when newline ('\n') is seen.
data_into_list = data.split("\n")
my_file.close()
mean_array = data_into_list[0]
mean_array = mean_array.split()
# print(mean_array)
mean_array = [float(i) for i in mean_array]
std_array = data_into_list[1]
std_array =std_array.split()
std_array = [float(i) for i in std_array]
print('Mean and Std: ', mean_array, std_array)
    
dataset2, test_loader2 = get_data('brain',dn  , '/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data', data_size,data_size, 
                                mean_array, std_array, 1, 1)
# mask = np.load('/project/roysam/rwmills/data/brain/smallAllMasks.npy') 

#for 7 channel - good bbx location 
centroid_Q2=[] 
for i, (fpath, fpid, _) in enumerate(dataset2.query) : 
    # i_id = fpath.split('s1_')[1].split('.tif')[0]
    # actual = TCells.iloc[int(i_id)].tolist()[0] 
    # numbers_list = infos.iloc[actual].tolist() 
    # centroid_Q.append(numbers_list)
    cy, cx = fpath.split('_c')[1].split('s')[0].split('_')
    centroid_Q2.append([int(cx), int(cy)])

centroid_G2=[] 
for i, (fpath, fpid, _) in enumerate(dataset2.gallery) : 
    # i_id = fpath.split('s1_')[1].split('.tif')[0]
    # actual = TCells.iloc[int(i_id)].tolist()[0] 
    # numbers_list = infos.iloc[actual].tolist() 
    # centroid_Q.append(numbers_list)
    cy, cx = fpath.split('_c')[1].split('s')[0].split('_')
    centroid_G2.append([int(cx), int(cy)])
print('max g: ', max(centroid_G2)) 

#In[] Stuff 


#In[] Centroids 
if manual is True: 
    internal_centroids = centroid_Q # np.load('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/clicks/clicks0.npy')
else: 
    internal_centroids = centroid_Q #just do it automatically 

#In[]  plotting functions 



#plot images 
def plot_vids_comparison(indx, images, indx2, images2, frames, class_ids): 
    print('plotting ids: ', indx, indx2) 
    data=images[indx]
    data2=images2[indx2]

    fig = plt.figure(figsize=(20, 10))  # width, height in inches
    for i in range(frames):
        if frames <10: 
            #row, col, position 
            sub = fig.add_subplot(2, frames, i + 1)
            sub2 = fig.add_subplot(2, frames, frames+i + 1)
        else: 
            sub = fig.add_subplot(int(np.ceil(np.sqrt(frames))), int(np.ceil(np.sqrt(frames))), i + 1)
        plt.setp(sub.get_xticklabels(), visible=False)
        plt.setp(sub.get_yticklabels(), visible=False)
        sub.title.set_text('Orig '+ class_ids[i])
        if frames <10:
            sub.imshow(data[:,:,i], interpolation='nearest', cmap='gray', vmin=0, vmax=np.max(data))
        else: 
            sub.imshow(data[:,:,i], interpolation='nearest') 
        
        plt.setp(sub2.get_xticklabels(), visible=False)
        plt.setp(sub2.get_yticklabels(), visible=False)
        sub2.title.set_text('Sug '+ class_ids[i])
        if frames <10:
            sub2.imshow(data2[:,:,i] , interpolation='nearest',  cmap='gray', vmin=0, vmax=np.max(data2))
        else: 
            sub2.imshow(data2[:,:,i], interpolation='nearest') 
        
    plt.show()
    
#plot images 
def plot_vids(indx, images, frames, class_ids): 
    data=images[indx]
#     data = next(itertools.islice(images, 0, None))[0][0]
#     data.shape
    print(data.shape)
    fig = plt.figure(figsize=(12, 12))  # width, height in inches
    for i in range(frames):
        if frames <10: 
            sub = fig.add_subplot(frames, 1, i + 1)
        else: 
            sub = fig.add_subplot(int(np.ceil(np.sqrt(frames))), int(np.ceil(np.sqrt(frames))), i + 1)
        plt.setp(sub.get_xticklabels(), visible=False)
        plt.setp(sub.get_yticklabels(), visible=False)
        sub.title.set_text(class_ids[i])
        sub.imshow(data[:,:,i], interpolation='nearest')
        
    plt.show()

    
def plot7ch(data, frames, class_ids): 
    # print(data.shape)
    fig = plt.figure(figsize=(12, 12))  # width, height in inches
    for i in range(frames):
        # if frames <10: 
        sub = fig.add_subplot( frames, 1, i + 1)
        # else: 
            # sub = fig.add_subplot(int(np.ceil(np.sqrt(frames))), int(np.ceil(np.sqrt(frames))), i + 1)
        plt.setp(sub.get_xticklabels(), visible=False)
        plt.setp(sub.get_yticklabels(), visible=False)
        sub.title.set_text(class_ids[i])
        sub.imshow(data[:,:,i], interpolation='nearest', cmap='gray')
        
    plt.show()

    
def plotSeveral_7ch(data_list, samples, class_ids): 
    frames = 7
    # fig = plt.figure(figsize=(30, 30)) 
    # fig = plt.figure(figsize=(12, 12))  # width, height in inches
    fig, axs = plt.subplots(nrows=frames, ncols=samples, figsize=(30, 20)) 
    
    # plt.subplots_adjust(hspace=0.2)
    for j in range(samples): # how many samples (rows) 
        data = data_list[j] #get the jth sample 
        for i in range(frames): #How many channels (columns) 
            # if frames <10: 
            # ax = plt.subplot(samples, frames, j+i + 1)
            # sub = fig.add_subplot(frames, samples, i + j+ 1) #7 columns, sample rows
            # else: 
                # sub = fig.add_subplot(int(np.ceil(np.sqrt(frames))), int(np.ceil(np.sqrt(frames))), i + 1)
            plt.setp(axs[i,j].get_xticklabels(), visible=False)
            plt.setp(axs[i,j].get_yticklabels(), visible=False)
            axs[i,j].title.set_text(class_ids[i])
            # ax.imshow(data[:,:,i], interpolation='nearest')
            axs[i,j].imshow(data[:,:,i], interpolation='nearest', cmap='gray')
    plt.subplots_adjust(wspace=.2, hspace=.2) 
        
    plt.show()
    
    
def plot_overlay_brain_cents(cent, cent2, bigimg): 
    fig = plt.figure(figsize=(15, 15)) 
    plt.plot(int(cent[0]), int(cent[1]), marker='*', color='red', markersize=15)
    plt.plot(int(cent2[0]), int(cent2[1]), marker='*', color='green', markersize=15)
    plt.imshow(bigimg, cmap='gray')
    plt.show()
    
def plot_overlay_brain(cent, cent_list, bigimg): 
    fig = plt.figure(figsize=(15, 15)) 
    for i in cent_list : 
        plt.plot(i[0], i[1], marker='*', color='green', markersize=15)
    plt.plot(cent[0], cent[1], marker='*', color='red', markersize=15)
    plt.imshow(bigimg, cmap='gray')
    plt.show()
    
def plot_overlay_brainBbx(centL, cent_listL, bigimg): 
    #image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :]
    #doesn't work 
    # cent = [int((centL[3] - centL[1])/2), int((centL[2] - centL[1])/2) ] 
    # cent_list = [ [int((centi[3] - centi[1])/2), int((centi[2] - centi[1])/2) ] for centi in cent_listL]  
    #works but is flipped? 
    cent = [int(centL[1] + (centL[3] - centL[1])/2) , int(centL[0] + (centL[2] - centL[0]))/2] 
    cent_list = [ [int(centi[1] + (centi[3] - centi[1])/2) , int(centi[0] + (centi[2] - centi[0]))/2 ] for centi in cent_listL]  
    fig = plt.figure(figsize=(15, 15)) 
    for i in cent_list : 
        # plt.plot(i[0], i[1], marker='*', color='green', markersize=15)
        # plt.plot(i[1] + 10763, i[0] + 10763, marker='*', color='green', markersize=15)
        plt.plot(i[1] , i[0], marker='*', color='green', markersize=15)
    # plt.plot(cent[0], cent[1], marker='*', color='red', markersize=15)
    # plt.plot(cent[1] + 10763, cent[0] + 10763, marker='*', color='red', markersize=15)
    plt.plot(cent[1] , cent[0], marker='*', color='red', markersize=15)
    plt.imshow(bigimg, cmap='gray')
    plt.show()

def plot_overlay_brainBbx_cent(cent, cent_list, bigimg): 
    #image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :] 
    fig = plt.figure(figsize=(15, 15)) 
    for i in cent_list : 
        # plt.plot(i[0], i[1], marker='*', color='green', markersize=15)
        # plt.plot(i[1] + 10763, i[0] + 10763, marker='*', color='green', markersize=15)
        plt.plot(i[1] , i[0], marker='.', color='green', markersize=15)
    # plt.plot(cent[0], cent[1], marker='*', color='red', markersize=15)
    # plt.plot(cent[1] + 10763, cent[0] + 10763, marker='*', color='red', markersize=15)
    plt.plot(cent[1] , cent[0], marker='.', color='red', markersize=15)
    plt.imshow(bigimg, cmap='gray')
    plt.show()

def plot_4_brain_overlay(cent, cent_list, dist_list, xy_mask, cutoff2, cutoff3, cutoff4, magic_number2, magic_number3, magic_number4,bigimg, fname): 
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20)) 
    # plt.setp(axs[0,0].get_xticklabels(), visible=False)
    # plt.setp(axs[0,0].get_yticklabels(), visible=False)
    plt.setp(axs[0,1].get_xticklabels(), visible=False)
    plt.setp(axs[0,1].get_yticklabels(), visible=False)
    plt.setp(axs[1,0].get_xticklabels(), visible=False)
    plt.setp(axs[1,0].get_yticklabels(), visible=False)
    plt.setp(axs[1,1].get_xticklabels(), visible=False)
    plt.setp(axs[1,1].get_yticklabels(), visible=False)
    #plot the histogram: 
    sns.histplot(ax= axs[0,0], data=dist_list, binwidth = 0.05)
    #plot the near centroids based off of cutoffs 
    for count, i in enumerate(cent_list) :
        # print(dist_list[count], cutoff1) 
        # if dist_list[count] <= cutoff1:
        #     # plt.plot(i[1] , i[0], marker='.', color='green', markersize=15)     
        #     axs[0,0].plot(i[1] , i[0], marker='.', color='green', markersize=6)
        #     axs[0,1].plot(i[1] , i[0], marker='.', color='green', markersize=6)
        #     axs[1,0].plot(i[1] , i[0], marker='.', color='green', markersize=6)
        #     axs[1,1].plot(i[1] , i[0], marker='.', color='green', markersize=6)
        if dist_list[count] <= cutoff2: #dist_list[count] > cutoff1 and dist_list[count]<= cutoff2: 
            axs[0,1].plot(i[1] , i[0], marker='.', color='green', markersize=6)
            axs[1,0].plot(i[1] , i[0], marker='.', color='green', markersize=6)
            axs[1,1].plot(i[1] , i[0], marker='.', color='green', markersize=6)
        elif dist_list[count] > cutoff2 and dist_list[count] <= cutoff3: 
            axs[1,0].plot(i[1] , i[0], marker='.', color='green', markersize=6)
            axs[1,1].plot(i[1] , i[0], marker='.', color='green', markersize=6)
        elif dist_list[count] > cutoff3 and dist_list[count] <= cutoff4: 
            axs[1,1].plot(i[1] , i[0], marker='.', color='green', markersize=6)
        else: 
            print('...centroids complete') 
            break 
    # #plot the mask on top  
    # for ii in xy_mask: 
    #     i=ii[0]
    #     j=ii[1]
    #     axs[0,0].plot(i,j, marker='.', color='white', markersize=1)
    #     axs[0,1].plot(i,j, marker='.', color='white', markersize=1)
    #     axs[1,0].plot(i,j, marker='.', color='white', markersize=1)
    #     axs[1,1].plot(i,j, marker='.', color='white', markersize=1)
            
    #red dot 
    # print('plotting click points')
    # axs[0,0].plot(cent[1] , cent[0], marker='.', color='red', markersize=6) #plot the click point
    axs[0,1].plot(cent[1] , cent[0], marker='.', color='red', markersize=6) #plot the click point
    axs[1,0].plot(cent[1] , cent[0], marker='.', color='red', markersize=6) #plot the click point
    axs[1,1].plot(cent[1] , cent[0], marker='.', color='red', markersize=6) #plot the click point 
    #plot underlying image
    # print('plotting atlas') 
    # axs[0,0].imshow(bigimg, cmap='gray')
    axs[1,0].imshow(bigimg, cmap='gray')
    axs[0,1].imshow(bigimg, cmap='gray')
    axs[1,1].imshow(bigimg, cmap='gray')
    #title each 
    # print('setting titles')
    # axs[0,0].title.set_text('Distance Cutoff: ' + str(cutoff1) )
    axs[0,0].title.set_text('Histogram' )
    axs[0,1].title.set_text('Distance Cutoff ' + str(cutoff2) + ' count: ' +  str(magic_number2) )
    axs[1,0].title.set_text('Distance Cutoff ' + str(cutoff3)+ ' count: '  +  str(magic_number3) )
    axs[1,1].title.set_text('Distance Cutoff ' + str(cutoff4) + ' count: ' +  str(magic_number4) )
    plt.subplots_adjust(wspace=.2, hspace=.2) 
    # plt.show()        
    print('...saving')
    fig.savefig(fname)
    plt.close('all') #RWM close just to save memory? 
    return 'done!'

def plot_4_brain_overlay_IU(cent, cent_list1, cent_list2, cent_list3, cent_list4,dist1, dist2 , bigimg, fname): 
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20)) 
    # plt.setp(axs[0,0].get_xticklabels(), visible=False)
    # plt.setp(axs[0,0].get_yticklabels(), visible=False)
    plt.setp(axs[0,1].get_xticklabels(), visible=False)
    plt.setp(axs[0,1].get_yticklabels(), visible=False)
    plt.setp(axs[1,0].get_xticklabels(), visible=False)
    plt.setp(axs[1,0].get_yticklabels(), visible=False)
    plt.setp(axs[1,1].get_xticklabels(), visible=False)
    plt.setp(axs[1,1].get_yticklabels(), visible=False)
    
    #plot the near centroids based off of cutoffs 
    for count, i in enumerate(cent_list1) :
        axs[0,0].plot(i[1] , i[0], marker='.', color='green', markersize=6)
            

    for count, i in enumerate(cent_list2) :
        axs[0,1].plot(i[1] , i[0], marker='.', color='green', markersize=6)

    for count, i in enumerate(cent_list3) :
        axs[1,0].plot(i[1] , i[0], marker='.', color='green', markersize=6)

    for count, i in enumerate(cent_list4) :
        axs[1,1].plot(i[1] , i[0], marker='.', color='green', markersize=6)

            
    #red dot 
    # print('plotting click points')
    # axs[0,0].plot(cent[1] , cent[0], marker='.', color='red', markersize=6) #plot the click point
    axs[0,0].plot(cent[1] , cent[0], marker='.', color='red', markersize=6) #plot the click point
    axs[0,1].plot(cent[1] , cent[0], marker='.', color='red', markersize=6) #plot the click point
    axs[1,0].plot(cent[1] , cent[0], marker='.', color='red', markersize=6) #plot the click point
    axs[1,1].plot(cent[1] , cent[0], marker='.', color='red', markersize=6) #plot the click point 
    #plot underlying image
    # print('plotting atlas') 
    # axs[0,0].imshow(bigimg, cmap='gray')
    axs[0,0].imshow(bigimg, cmap='gray')
    axs[1,0].imshow(bigimg, cmap='gray')
    axs[0,1].imshow(bigimg, cmap='gray')
    axs[1,1].imshow(bigimg, cmap='gray')
    #title each 
    # print('setting titles')
    # axs[0,0].title.set_text('Distance Cutoff: ' + str(cutoff1) )
    axs[0,0].title.set_text('Set 1 ' +  ' count: ' +  str( len(cent_list1)) )
    axs[0,1].title.set_text('Set 2' + ' count: '  +  str( len(cent_list2)) )
    axs[1,0].title.set_text('Soft Intersection ' + str(dist1) + ' count: ' +  str( len(cent_list3)) )
    axs[1,1].title.set_text('Soft Intersection ' + str(dist2) + ' count: ' +  str(len(cent_list4) ) )

    plt.subplots_adjust(wspace=.2, hspace=.2) 
    # plt.show()        
    print('...saving')
    fig.savefig(fname)
    plt.close('all') #RWM close just to save memory? 
    return 'done!'
     
    
def plot_overlay_brain_heat(cent, cent_list, distances, bigimg): 
    heatmap, xedges = np.histogram(distances, bins=50)
    # print(heatmap, xedges) 
    heatmap=heatmap/np.amax(heatmap) 
    extent = [xedges[0], xedges[-1]]

    fig = plt.figure(figsize=(15, 15)) 
    for ind, i in enumerate(cent_list ): 
        # print(distances[i] ) 
        col_indx = np.where(distances[ind] <= xedges)[0][0] 
        plt.plot(i[0], i[1], marker='*', color=str(heatmap[col_indx-1] ), markersize=15)
    plt.plot(cent[0], cent[1], marker='*', color='red', markersize=15)
    plt.imshow(bigimg, cmap='gray')
    plt.show()
    
def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes

def plot_little_brain_cent(centL, bigimg): 
    cent = [int(centL[1] + (centL[3] - centL[1])/2) , int(centL[0] + (centL[2] - centL[0]))/2] 
    fig = plt.figure(figsize=(15, 15)) 
    plt.plot(cent[1] , cent[0], marker='*', color='red', markersize=15)
    plt.imshow(bigimg, cmap='gray')
    plt.show()
    
def get_top(indexList): 
    
    top_mask0 = sorted(indexList.values())[-1]
    top1Mask = sorted(indexList.values())[-2]
    if top_mask0 == top1Mask: 
        top1Mask = top1Mask - 1

    # result= dict((new_val,new_k) for new_k,new_val in sim_list.items()).get(top_mask)
    max_keys0 = [key for key, value in indexList.items() if value == top_mask0]
    max_keys01 = [key for key, value in indexList.items() if value == top1Mask]

    return max_keys0, max_keys01
def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes
# print(result)

def pairwise_distance(mini_Set, gal):
#     m, n = x.size(0), y.size(0)
#     print('m and n: ', m, n) 
    
#     b= torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     print('b works.', b.size() ) 
#     a=  torch.pow(x, 2).sum(dim=0, keepdim=True).expand(m, n)
#     print('a words', a.size() ) 
#     dist_m = a +  b
#     print('got dist', dist_m.size() ) 
#     # dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#            # torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist_m.addmm_(1, -2, x, y.t())
#     return dist_m
    allDist=[] 
    pdist = torch.nn.PairwiseDistance(p=2)
    v2 = torch.unsqueeze(torch.from_numpy(mini_Set) , 0)
    for i in gal: 
        v1 = torch.unsqueeze(torch.from_numpy(i) , 0)
        # compute the distance
        output = float(pdist(v1, v2 )) 
        allDist.append(output) 
    sorted_distances= sorted(allDist)
    sorted_indicies= np.argsort(allDist)
    return allDist, sorted_distances, sorted_indicies

def find_several_cents(target_num, num_top, centroid_Q,centroid_G, q): 
    ''' target_num is the number in the q of the cell to select 
        num_top is the number of similar centroids to find 
        centroi_Q has the centroid information of the target_num 
        centroid_G has the centroid infromation for the similar cells to search 
        q is for generating the mini feature matrix 
    ''' 
    A=np.array(centroid_G) 
    leftbottom = np.array(centroid_Q[target_num])
    distances = np.linalg.norm(A-leftbottom, axis=1)
    min_index = np.argmin(distances)
    # print(f"the closest point is {A[min_index]}, at a distance of {distances[min_index]}")
    sorted_distances= sorted(distances)
    sorted_indicies= np.argsort(distances)
    mini_dist = sorted_distances[0:num_top] 
    mini_ind = sorted_indicies[0:num_top] 
    mini_Set = q[mini_ind[0] ]
    for i in mini_ind[1:]:
        mini_Set+=i
    mini_Set =mini_Set/num_top
    
    return sorted_indicies, mini_Set

def find_several_cents_to_pt(find_cent, centroid_list): 
    ''' target_num is the number in the q of the cell to select 
        num_top is the number of similar centroids to find 
        centroi_Q has the centroid information of the target_num 
        centroid_G has the centroid infromation for the similar cells to search 
        q is for generating the mini feature matrix 
    ''' 
    A=np.array(centroid_list) 
    leftbottom = np.array(find_cent)
    distances = np.linalg.norm(A-leftbottom, axis=1)
    min_index = np.argmin(distances)
    # print(f"the closest point is {A[min_index]}, at a distance of {distances[min_index]}")
    sorted_distances= sorted(distances)
    sorted_indicies= np.argsort(distances)

    
    return sorted_distances, sorted_indicies

#In[] Do the plotting 

#In[] Generate Images 
count=0 #in case the code gets stopped 
internal_centroids = internal_centroids[count::] 

for indx, int_cent in enumerate(internal_centroids): 
    print('Working on sample: ', count)
    print('goal centroid is: ', int_cent) 
    sD, sI = find_several_cents_to_pt((int_cent ), centroid_Q)

    target_num =sI[0] 
    s = dist[target_num]
    sort_index = np.argsort(s)

    # Generate histogram/distribution plot
    # sns.displot(sorted(s), binwidth = 0.05)
    magic_number1 = (np.sum((s < 1)*1) ) 
    magic_number2 =  (np.sum((s < 1.3)*1) ) 
    
    if manual is True: 
        int_cent1 = int_cent
        
        sD1, sI1 = find_several_cents_to_pt((int_cent1 ), centroid_Q)
        sD2, sI2 = find_several_cents_to_pt((int_cent1 ), centroid_Q2)
        target_num1 =sI1[0] 
        s1 = dist[target_num1]
        sort_index1 = np.argsort(s1)
        target_num2 =sI2[0] 
        s2 = dist2[target_num2]
        sort_index2 = np.argsort(s2)
        
        #cutoff: 
        magic_number11 = (np.sum((s1 < 1)*1) ) 
        magic_number12 =  (np.sum((s1 < 1.25)*1) ) 
        magic_number21 = (np.sum((s2 < 1)*1) ) 
        magic_number22 =  (np.sum((s2 < 1.25)*1) ) 
        magic_number = int((magic_number11 + magic_number21) / 2 )
        magic_number2 = int((magic_number12 + magic_number22) / 2 )
        
        special_num = magic_number2 #more samples 

        radius_check1, radius_check2=[], []  
        dist_thresh1, dist_thresh2= 100, 200 


        top10_cent_indx1 = [ centroid_G[i] for i in sort_index1[0:magic_number12] ]#top 100
        top10_cent_indx2 = [ centroid_G2[i] for i in sort_index2[0:magic_number22] ]#top 100

        for i in top10_cent_indx2: 
            distpts, ids = find_several_cents_to_pt( (i ), top10_cent_indx1)
            count=1 
            for idx, d in enumerate(distpts): 
                if d <= dist_thresh1: 
                    if count== 1: 
                        radius_check1.append(i) 
                        radius_check1.append(top10_cent_indx2[idx]) 
                        count+=1
                    else: 
                        radius_check1.append(top10_cent_indx2[idx]) 
        for i in top10_cent_indx2: 
            distpts, ids = find_several_cents_to_pt( (i ), top10_cent_indx1)
            count=1 
            for idx, d in enumerate(distpts): 
                if d <= dist_thresh2: 
                    if count== 1: 
                        radius_check2.append(i) 
                        radius_check2.append(top10_cent_indx2[idx]) 
                        count+=1
                    else: 
                        radius_check2.append(top10_cent_indx2[idx]) 

        fname = os.path.join(dst_path, str(len(radius_check1) ) + '_' + str(indx) + '.png') 

        plot_4_brain_overlay_IU(centroid_Q[target_num1], top10_cent_indx1, top10_cent_indx2, radius_check1, radius_check2, dist_thresh1, dist_thresh2 , brain_img_orig, fname)

        print('Point complete')
        plt.close('all') #RWM so memory isn't too much ? 
    else: 
        distpts, ids = find_several_cents_to_pt( (i ), top10_cent_indx1)
        count=1 
        for idx, d in enumerate(distpts): 
            if d <= dist_thresh1: 
                if count== 1: 
                    radius_check1.append(i) 
                    radius_check1.append(top10_cent_indx2[idx]) 
                    count+=1
                else: 
                    radius_check1.append(top10_cent_indx2[idx]) 
        for i in top10_cent_indx2: 
            distpts, ids = find_several_cents_to_pt( (i ), top10_cent_indx1)
            count=1 
            for idx, d in enumerate(distpts): 
                if d <= dist_thresh2: 
                    if count== 1: 
                        radius_check2.append(i) 
                        radius_check2.append(top10_cent_indx2[idx]) 
                        count+=1
                    else: 
                        radius_check2.append(top10_cent_indx2[idx]) 

        fname = os.path.join(dst_path, special_num + '_' + str(indx) + '.png') 

        plot_4_brain_overlay_IU(centroid_Q[target_num1], top10_cent_indx1, top10_cent_indx2, radius_check1, radius_check2, dist_thresh1, dist_thresh2 , brain_img_orig, fname)

        

        print('Point complete')
        plt.close('all') #RWM so memory isn't too much ? 
    count+=1 

