#Loading Data Functions 

import torch 
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import os.path as osp 
from torch.utils.data import DataLoader
import sys 
os.chdir('/project/ece/roysam/lhuang37/cluster-contrast-reid/') 
print(os.getcwd())
sys.path.append('/project/ece/roysam/lhuang37/cluster-contrast-reid/clustercontrast/') 
sys.path.append('/project/ece/roysam/lhuang37/cluster-contrast-reid/clustercontrast/datasets/') 
import clustercontrast 
from clustercontrast import datasets
import h5py 
import numpy as np 
from skimage.io import imread 
import matplotlib.pyplot as plt 
import seaborn as sn
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
# import matplotlib.pyplot as plt % matplotlib inline
import random 
import pandas as pd
from skimage.transform import rescale, resize, downscale_local_mean
import scipy
import seaborn as sns
from sklearn.preprocessing import normalize


##################################################################################################################
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

def set_up_data(dn, dn2, data_size, data_type ): 
    try: 
        feats = h5py.File('/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/logs/' + dn + '/feature_results.h5', 'r') 
    except: 
        feats = h5py.File('/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/logs/' + dn2 + '/feature_results.h5', 'r') 
    try: 
        distance_rerank = np.load('/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/logs/' + dn2 + 'rerankingDist.npy') 
    except: 
        distance_rerank=[] 
        print('No reranking accessible') 
    try: 
        cosineFeat = h5py.File('/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/logs/' + dn + '/feature_results_Cosine.h5', 'r' ) 
    except: 
        print('No Cosine distance features avalaible') 
    
    h5filename = '/project/ece/roysam/lhuang37/rachel_data/Select_RegionMasked_50.h5'

    
    #----------------------------------------------------------------------------------------------------------------
    dist = feats['distmat']
    gal = feats['gallery_features'] 
    q = feats['query_features']

    my_file = open(os.path.join('/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/data/', dn, "mean_std.txt"), "r")
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

    try: 
        dataset, test_loader = get_data(data_type,dn  , '/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/data/', data_size,data_size, 
                                    mean_array, std_array, 1, 1)
    except: 
        dataset, test_loader = get_data(data_type,dn2  , '/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/data/', data_size,data_size, 
                                    mean_array, std_array, 1, 1)

    if 'class' in dn: 
        j = 2
    elif 'cortex' in dn: 
        j=2
    else: 
        j=1
    centroid_Q=[] 
    for i, (fpath, fpid, _) in enumerate(dataset.query) : 

        cy, cx = fpath.split('_c')[j].split('s')[0].split('_')
        if 'single_v_double' in dn:
            centroid_Q.append([int(cy), int(cx)])    
        else: 
            centroid_Q.append([int(cx), int(cy)])

    centroid_G=[] 
    for i, (fpath, fpid, _) in enumerate(dataset.gallery) : 
        cy, cx = fpath.split('_c')[j].split('s')[0].split('_')
        if 'single_v_double' in dn: 
            centroid_G.append([int(cy), int(cx)])
        else: 
            centroid_G.append([int(cx), int(cy)])

    return dist, distance_rerank, centroid_Q, centroid_G, dataset


def set_up_dataCosine(dn, dn2, data_size, data_type ): 
    try: 
        feats = h5py.File('/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/logs/' + dn + '/feature_results_Cosine.h5', 'r') 
    except: 
        feats = h5py.File('/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/logs/' + dn2 + '/feature_results_Cosine.h5', 'r') 

    
    h5filename = '/project/ece/roysam/lhuang37/rachel_data/Select_RegionMasked_50.h5'

    
    #----------------------------------------------------------------------------------------------------------------
    dist = feats['distmat']
    gal = feats['gallery_features'] 
    q = feats['query_features']

    my_file = open(os.path.join('/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/data/', dn, "mean_std.txt"), "r")
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

    try: 
        dataset, test_loader = get_data(data_type,dn  , '/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/data/', data_size,data_size, 
                                    mean_array, std_array, 1, 1)
    except: 
        dataset, test_loader = get_data(data_type,dn2  , '/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/data/', data_size,data_size, 
                                    mean_array, std_array, 1, 1)

    if 'class' in dn: 
        j = 2
    elif 'cortex' in dn: 
        j=2
    else: 
        j=1
    centroid_Q=[] 
    for i, (fpath, fpid, _) in enumerate(dataset.query) : 

        cy, cx = fpath.split('_c')[j].split('s')[0].split('_')
        if 'single_v_double' in dn:
            centroid_Q.append([int(cy), int(cx)])    
        else: 
            centroid_Q.append([int(cx), int(cy)])

    centroid_G=[] 
    for i, (fpath, fpid, _) in enumerate(dataset.gallery) : 
        cy, cx = fpath.split('_c')[j].split('s')[0].split('_')
        if 'single_v_double' in dn: 
            centroid_G.append([int(cy), int(cx)])
        else: 
            centroid_G.append([int(cx), int(cy)])
            
    #normalize rows of matrix
    normalize(dist, axis=1, norm='l1')
    
    return dist, centroid_Q, centroid_G, dataset



#-----------------------------------------------------------Plotting------------------------------------------------
def plot7ch_single(data, frames, minn, maxx):      
    fig = plt.figure(figsize=(12, 12))  # width, height in inches
    for i in range(frames):
        # if frames <10: 
        sub = fig.add_subplot( 1, frames, i+ 1) #horizontal 
        # sub = fig.add_subplot( frames, 1, i+1 ) #vertical
        # else: 
            # sub = fig.add_subplot(int(np.ceil(np.sqrt(frames))), int(np.ceil(np.sqrt(frames))), i + 1)
        plt.setp(sub.get_xticklabels(), visible=False)
        plt.setp(sub.get_yticklabels(), visible=False)
        # sub.title.set_text(class_ids[i])
        sub.imshow(data[:,:,i], interpolation='spline16', cmap='gray', vmin=minn, vmax = maxx)
        
    plt.show()


def plot7ch_cmap(data, frames, cmap_list): 
    # print(data.shape)
    mins=[] 
    maxes=[] 
    for i in range(frames):  
        mins.append(np.amin(data[:,:,i]))
        maxes.append(np.amax(data[:,:,i]))
    minn = 2000 # np.amin(np.array(mins))  #      
    maxx= np.amax(np.array(maxes)) #* 5 #np.amax(np.array(maxes))  
    print('min and max: ', minn, maxx) 
    fig = plt.figure(figsize=(12, 12))  # width, height in inches
    for i in range(frames):
        # if frames <10: 
        sub = fig.add_subplot( 1, frames, i+ 1)
        # sub = fig.add_subplot( frames, 1, i+1 )
        # else: 
            # sub = fig.add_subplot(int(np.ceil(np.sqrt(frames))), int(np.ceil(np.sqrt(frames))), i + 1)
        plt.setp(sub.get_xticklabels(), visible=False)
        plt.setp(sub.get_yticklabels(), visible=False)
        # sub.title.set_text(class_ids[i])
        sub.imshow(data[:,:,i], interpolation='spline16', cmap=cmap_list[i], vmin=minn, vmax = maxx)
        
    plt.show()


def plot7ch(data, frames, class_ids): 
    # print(data.shape)
    mins=[] 
    maxes=[] 
    for i in range(frames):  
        mins.append(np.amin(data[:,:,i]))
        maxes.append(np.amax(data[:,:,i]))
    minn = np.amin(np.array(mins))        
    maxx= np.amax(np.array(maxes)) - 2000 # +200  
        
    fig = plt.figure(figsize=(12, 12))  # width, height in inches
    for i in range(frames):
        # if frames <10: 
        sub = fig.add_subplot( 1, frames, i+ 1)
        # sub = fig.add_subplot( frames, 1, i+1 )
        # else: 
            # sub = fig.add_subplot(int(np.ceil(np.sqrt(frames))), int(np.ceil(np.sqrt(frames))), i + 1)
        plt.setp(sub.get_xticklabels(), visible=False)
        plt.setp(sub.get_yticklabels(), visible=False)
        # sub.title.set_text(class_ids[i])
        sub.imshow(data[:,:,i], interpolation='spline16', cmap='gray', vmin=minn, vmax = maxx)
        
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
    plt.imshow(bigimg, cmap='gray') # .invert_yaxis()
    plt.show()

def plot_overlay_brainBbx_multicent(cent, cent_list, bigimg): 
    #image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :] 
    fig = plt.figure(figsize=(15, 15)) 
    for i in cent_list : 
        plt.plot(i[1] , i[0], marker='.', color='green', markersize=15)
    for i in cent: 
        plt.plot(i[1] , i[0], marker='.', color='red', markersize=15)
    plt.imshow(bigimg, cmap='gray') # .invert_yaxis()
    plt.show()    
    
def quick_overlay_brainBbx_multicent(cent, cent_list): 
    #image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :] 
    fig = plt.figure(figsize=(15, 15)) 
    for i in cent_list : 
        plt.plot(i[1] , i[0], marker='.', color='green', markersize=15)
    for i in cent: 
        plt.plot(i[1] , i[0], marker='.', color='red', markersize=15)
    plt.show().invert_yaxis()    

def plot_cents_only(cent, cent_list, bigimg): 
    #image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :] 
    fig = plt.figure(figsize=(bigimg.shape)) 
    # fig = plt.figure(figsize=(15, 15)) 
    for i in cent_list : 
        # plt.plot(i[0], i[1], marker='*', color='green', markersize=15)
        # plt.plot(i[1] + 10763, i[0] + 10763, marker='*', color='green', markersize=15)
        plt.plot(i[1] , i[0], marker='.', color='green', markersize=3)
    # plt.plot(cent[0], cent[1], marker='*', color='red', markersize=15)
    # plt.plot(cent[1] + 10763, cent[0] + 10763, marker='*', color='red', markersize=15)
    plt.plot(cent[1] , cent[0], marker='.', color='red', markersize=3)
    plt.imshow(np.zeros(bigimg.shape), cmap='gray')
    
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # plt.show()
    return data 

def plot_4_brain_overlay(cent, cent_list, dist_list, xy_mask, cutoff1, cutoff2, cutoff3, cutoff4, bigimg): 
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20)) 
    plt.setp(axs[0,0].get_xticklabels(), visible=False)
    plt.setp(axs[0,0].get_yticklabels(), visible=False)
    plt.setp(axs[0,1].get_xticklabels(), visible=False)
    plt.setp(axs[0,1].get_yticklabels(), visible=False)
    plt.setp(axs[1,0].get_xticklabels(), visible=False)
    plt.setp(axs[1,0].get_yticklabels(), visible=False)
    plt.setp(axs[1,1].get_xticklabels(), visible=False)
    plt.setp(axs[1,1].get_yticklabels(), visible=False)
    #plot the near centroids based off of cutoffs 
    for count, i in enumerate(cent_list) :
        # print(dist_list[count], cutoff1) 
        if dist_list[count] <= cutoff1:
            # plt.plot(i[1] , i[0], marker='.', color='green', markersize=15)     
            axs[0,0].plot(i[1] , i[0], marker='.', color='green', markersize=9)
            axs[0,1].plot(i[1] , i[0], marker='.', color='green', markersize=9)
            axs[1,0].plot(i[1] , i[0], marker='.', color='green', markersize=9)
            axs[1,1].plot(i[1] , i[0], marker='.', color='green', markersize=9)
        elif dist_list[count] > cutoff1 and dist_list[count]<= cutoff2: 
            axs[0,1].plot(i[1] , i[0], marker='.', color='green', markersize=9)
            axs[1,0].plot(i[1] , i[0], marker='.', color='green', markersize=9)
            axs[1,1].plot(i[1] , i[0], marker='.', color='green', markersize=9)
        elif dist_list[count] > cutoff2 and dist_list[count] <= cutoff3: 
            axs[1,0].plot(i[1] , i[0], marker='.', color='green', markersize=9)
            axs[1,1].plot(i[1] , i[0], marker='.', color='green', markersize=9)
        elif dist_list[count] > cutoff3 and dist_list[count] <= cutoff4: 
            axs[1,1].plot(i[1] , i[0], marker='.', color='green', markersize=9)
        else: 
            print('Centroids complete') 
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
    axs[0,0].plot(cent[1] , cent[0], marker='.', color='red', markersize=9) #plot the click point
    axs[0,1].plot(cent[1] , cent[0], marker='.', color='red', markersize=9) #plot the click point
    axs[1,0].plot(cent[1] , cent[0], marker='.', color='red', markersize=9) #plot the click point
    axs[1,1].plot(cent[1] , cent[0], marker='.', color='red', markersize=9) #plot the click point 
    #plot underlying image 
    axs[0,0].imshow(bigimg, cmap='gray')
    axs[1,0].imshow(bigimg, cmap='gray')
    axs[0,1].imshow(bigimg, cmap='gray')
    axs[1,1].imshow(bigimg, cmap='gray')
    #title each 
    axs[0,0].title.set_text('Distance Cutoff: ' + str(cutoff1) )
    axs[0,1].title.set_text('Distance Cutoff: ' + str(cutoff2) )
    axs[1,0].title.set_text('Distance Cutoff: ' + str(cutoff3) )
    axs[0,0].title.set_text('Distance Cutoff: ' + str(cutoff4) )
    plt.subplots_adjust(wspace=.2, hspace=.2) 
    plt.show()        
    
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
    print(f"the closest point is {A[min_index]}, at a distance of {distances[min_index]}")
    sorted_distances= sorted(distances)
    sorted_indicies= np.argsort(distances)

    
    return sorted_distances, sorted_indicies

def quickplot_overlay_brainBbx_cent(cent, cent_list): 
    #image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :] 
    fig = plt.figure(figsize=(15, 15)) 
    for i in cent_list : 
        # plt.plot(i[0], i[1], marker='*', color='green', markersize=15)
        # plt.plot(i[1] + 10763, i[0] + 10763, marker='*', color='green', markersize=15)
        plt.plot(i[1] , i[0], marker='.', color='green', markersize=15)
    # plt.plot(cent[0], cent[1], marker='*', color='red', markersize=15)
    # plt.plot(cent[1] + 10763, cent[0] + 10763, marker='*', color='red', markersize=15)
    plt.plot(cent[1] , cent[0], marker='.', color='red', markersize=15)
    plt.show()
    
def quickplot_overlay_brainBbx_centHEAT(cent, cent_list, distances): 
    #distnaces are sorted!!
    #image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :] 
    
    heatmap, xedges = np.histogram(distances, bins=50)
    # print(heatmap, xedges) 
    heatmap=heatmap/np.amax(heatmap) 
    extent = [xedges[0], xedges[-1]]

    fig = plt.figure(figsize=(15, 15)) 
    for ind, i in enumerate(cent_list ): 
        col_indx = np.where(distances[ind] <= xedges)[0][0] 
        plt.plot(i[1] , i[0], marker='.', color=str(heatmap[col_indx-1] ), markersize=15)
    # plt.plot(cent[0], cent[1], marker='*', color='red', markersize=15)
    # plt.plot(cent[1] + 10763, cent[0] + 10763, marker='*', color='red', markersize=15)
    plt.plot(cent[1] , cent[0], marker='.', color='red', markersize=15)
    plt.show()
    
plotting_samples=0.75 

box_size= 125

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

def generate_points(int_cent1, centroid_Q, centroid_G, dist, plotting_samples): 
    '''int_cent1 is the target centroid to find
        centroid_Q is the centroid list from the query set 
        centroid_G is the centroid list from the gallery set
        dist is the distance matrix to manipulate 
        plotting_samples is the number of samples to plot
    ''' 
    sD1a, sI1a = find_several_cents_to_pt((int_cent1 ), centroid_Q)
    #query Cell
    target_num1a =sI1a[0] 
    s1a = dist[target_num1a]
    magic_number = (np.sum((s1a < plotting_samples)*1) ) 
    sort_index1a = np.argsort(s1a)
    top10_cent_indx1a = [ centroid_G[i] for i in sort_index1a[0:magic_number] ]#top 100
    return top10_cent_indx1a


def generate_points_multiQuery(query_list, centroid_Q, centroid_G, dist, plotting_samples=1, dist_thresh=100): 
    '''query_list: a list of click point centroids from user 
        centroid_Q is the centroid list from the query set 
        centroid_G is the centroid list from the gallery set
        dist is the distance matrix to manipulate 
        plotting_samples is the number of samples to plot
        dist_thresh is the distance for the thresholded intra-distance measure intersection 
        '''
    cent_list=[] #store list of centroids for each click here 
    for int_cent1 in query_list: 
        sD1a, sI1a = find_several_cents_to_pt((int_cent1 ), centroid_Q)
        #query Cell
        target_num1a =sI1a[0] 
        s1a = dist[target_num1a]
        magic_number = (np.sum((s1a < plotting_samples)*1) ) 
        sort_index1a = np.argsort(s1a)
        # print(magic_number) 
        top10_cent_indx1a = [ centroid_G[i] for i in sort_index1a[0:magic_number] ]
        cent_list.append(top10_cent_indx1a) #add to overall list of centroids 
    radius_check=[] #store spatially relevant centroids here 
    for Alist in cent_list[0:-1] : 
        for Blist in cent_list[1::] : 
            for i in Alist: 
                distpts, ids = find_several_cents_to_pt( (i ), Blist)
                count=1 
                for idx, d in enumerate(distpts): 
                    if d <= dist_thresh: 
                        # print('Blist: ',list(Blist[idx]) , type(Blist[idx]),  ' radius: ', radius_check) 
                        if Blist[idx] not in radius_check: # [Blist[idx][0], Blist[idx][1]] not in radius_check: 
                            radius_check.append(Blist[idx])
    return radius_check


def generate_contrast_multiQuery(query_list_y, query_list_n, centroid_Q, centroid_G, dist, plotting_samples=1, dist_thresh=100): 
    '''query_list: a list of click point centroids from user 
        centroid_Q is the centroid list from the query set 
        centroid_G is the centroid list from the gallery set
        dist is the distance matrix to manipulate 
        plotting_samples is the number of samples to plot
        dist_thresh is the distance for the thresholded intra-distance measure intersection 
        '''
    yesilst = generate_points_multiQuery(query_list_y, centroid_Q, centroid_G, dist, plotting_samples=1, dist_thresh=100)
    nolist = generate_points_multiQuery(query_list_n, centroid_Q, centroid_G, dist, plotting_samples=1, dist_thresh=100) 
    
    radius_check=[] #store spatially relevant centroids here 

    for i in yesilst: 
        if i not in nolist: 
            distpts, ids = find_several_cents_to_pt( (i ), nolist)
            for idx, d in enumerate(distpts): 
                if d >= dist_thresh: 
                    # print('Blist: ',list(Blist[idx]) , type(Blist[idx]),  ' radius: ', radius_check) 
                    if yesilst[idx] not in radius_check: # [Blist[idx][0], Blist[idx][1]] not in radius_check: 
                        radius_check.append(Blist[idx]) 
              
    return radius_check


def generate_union_multiQuery(query_list, centroid_Q, centroid_G, dist, plotting_samples): 
    '''
    This function gather's the union given a list of centroids to pull from 
        query_list: a list of click point centroids from user 
        centroid_Q is the centroid list from the query set         
        centroid_G is the centroid list from the gallery set
        dist is the distance matrix to manipulate 
        plotting_samples is the number of samples to plot
        '''
    cent_list=[] #store list of centroids for each click here 
    for int_cent1 in query_list: 
        sD1a, sI1a = find_several_cents_to_pt((int_cent1 ), centroid_Q)
        #query Cell
        target_num1a =sI1a[0] 
        s1a = dist[target_num1a]
        #IF YOU WANT TO FILTER BY FEATURE DISTANCE, UNCOMMENT: 
        plotting_samplesDist = (np.sum((s1a < plotting_samples)*1) ) 
        # print('Plotting_samples: ', plotting_samplesDist) 
        sort_index1a = np.argsort(s1a)
        top10_cent_indx1a = [ centroid_G[i] for i in sort_index1a[0:plotting_samplesDist] ]
        cent_list.append(top10_cent_indx1a) #add to overall list of centroids 
    similar_centroid_list = [] 
    for i in cent_list: 
        similar_centroid_list+=i
    # print('centroids built', len(similar_centroid_list) ) 
    similar_centroid_list = set(tuple(i) for i in similar_centroid_list)
    similar_centroid_list = list(similar_centroid_list) 
    # print('duplicates removed', len(similar_centroid_list) ) 
    return similar_centroid_list

def generate_intersection_multiQuery(query_list, centroid_Q, centroid_G, dist, plotting_samples): 
    '''
    This function gather's the union given a list of centroids to pull from 
        query_list: a list of click point centroids from user 
        centroid_Q is the centroid list from the query set 
        centroid_G is the centroid list from the gallery set
        dist is the distance matrix to manipulate 
        plotting_samples is the number of samples to plot
        '''
    cent_list=[] #store list of centroids for each click here 
    for int_cent1 in query_list: 
        sD1a, sI1a = find_several_cents_to_pt((int_cent1 ), centroid_Q)
        #query Cell
        target_num1a =sI1a[0] 
        s1a = dist[target_num1a]
        #IF YOU WANT TO FILTER BY FEATURE DISTANCE, UNCOMMENT: 
        plotting_samples = (np.sum((s1a < plotting_samples)*1) ) 
        print('Plotting samples: ', plotting_samples) 
        sort_index1a = np.argsort(s1a)
        top10_cent_indx1a = [ centroid_G[i] for i in sort_index1a[0:plotting_samples] ]
        cent_list.append(top10_cent_indx1a) #add to overall list of centroids 
    similar_centroid_list = [] 
    for i in range(len(cent_list) - 1 ): 
        a = cent_list[i]
        b = cent_list[i+1] 
        intersectionAB = [ele1 for ele1 in a for ele2 in b if ele1 == ele2]
        # intersectionAB = list(set(a).intersection(set(b)))
        similar_centroid_list += intersectionAB
    return similar_centroid_list

def get_set_intersection(cent_list): 
    similar_centroid_list = [] 
    for i in range(len(cent_list) - 1 ): 
        a = cent_list[i]
        b = cent_list[i+1] 
        intersectionAB = [ele1 for ele1 in a for ele2 in b if ele1 == ele2]
        # intersectionAB = list(set(a).intersection(set(b)))
        similar_centroid_list += intersectionAB
    return similar_centroid_list

# def generate_threshintersection_multiQuery(query_list, centroid_Q, centroid_G, dist, plotting_samples, Dthresh): 
#     '''
#     This function gather's the union given a list of centroids to pull from 
#         query_list: a list of click point centroids from user 
#         centroid_Q is the centroid list from the query set 
#         centroid_G is the centroid list from the gallery set
#         dist is the distance matrix to manipulate 
#         plotting_samples is the number of samples to plot
#         '''
#     cent_list=[] #store list of centroids for each click here 
#     for int_cent1 in query_list: 
#         sD1a, sI1a = find_several_cents_to_pt((int_cent1 ), centroid_Q)
#         #query Cell
#         target_num1a =sI1a[0] 
#         s1a = dist[target_num1a]
#         #IF YOU WANT TO FILTER BY FEATURE DISTANCE, UNCOMMENT: 
#         # plotting_samples = (np.sum((s1a < plotting_samples)*1) ) 
#         sort_index1a = np.argsort(s1a)
#         top10_cent_indx1a = [ centroid_G[i] for i in sort_index1a[0:plotting_samples] ]
#         cent_list.append(top10_cent_indx1a) #add to overall list of centroids 
#     similar_centroid_list = [] 
#     for i in range(len(cent_list) - 1 ): 
#         a = cent_list[i]
#         b = cent_list[i+1] 
#         intersectionAB = [ele1 
#                           for ele1 in a #the original centroid location 
#                           for ele2, _ = find_several_cents_to_pt( (i ), Blist) in b 
#                           if ele2 <= Dthresh]
#         similar_centroid_list += intersectionAB
#     return similar_centroid_list
