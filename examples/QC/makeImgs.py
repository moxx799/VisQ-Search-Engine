''' Make images from set of click points:

must use: srun -t 1:00:00 --mem=300gb  --pty /bin/bash -l

'''
import sys
sys.path.append('/project/roysam/rwmills/repos/cluster-contrast-reid/')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from QC_functions import * 

#set the downsize value if we're minimizing 
downsize_val = 12

#In[] 

#Define the library of networks: 
#       Paths for each trained network by name 
# cell_type1 = ['cellType_5','all_brain_Cells', 75, 'brain' ] #dn, dn2, data_size, data_type 
cell_type2 = ['cellType_5','UNet_SAEA_cltl_cell0_TESTALL', 75, 'brain' ] #dn, dn2, data_size, data_type 
myelo0 = ['all_brain_noRECA_FULL','all_brain_noRECA_IM', 150, 'brain' ] #dn, dn2, data_size, data_type 

# no_reca = ['all_brain_noRECA', 'all_brain_noRECA_IM', 150, 'brain'] #neuronal phenotypic markers 
# myeloN1 = [  'connectivity0_noRECA_NeuN', 'UNet_CA_myelo' , 150, 'brain'] # connectivity0_S10N
# myeloN2 = [  'connectivity0_noRECA_NeuN', 'UNet_SAEA_myelo2' , 150, 'brain'] # 
myeloN3 = [  'connectivity0_noRECA_NeuN_TESTALL', 'UNet_ftTL_myelo2' , 150, 'brain'] # 
vasculature = [  'valsulature1_TESTALL', 'UNet_SAEA_cltl_vas0' , 150, 'brain'] # 
cyto_glial = [  'connectivity0_noRECA_Glial_TESTALL', 'UNet_SAEA_cltl_glial0' , 150, 'brain'] # 
neun_panel= [  'NeuronalMarkers_NeuN_ALLTEST', 'UNet_SAEA_cltl_JustNeuN0' , 175, 'brain'] #
glial_panel= [  'Glial_NoRECA_ALLTEST', 'UNet_SAEA_cltl_Justglial0' , 175, 'brain'] #


phenotype = []
death = [] 
#In[] 
#Define which DNNs are desired for user experiment 
# --------------------------------------------------------***USER INPUT***------------------------------------------------------------------------
# channel_collection = [ round1,round2] #, myeloN] 
channel_collection = [myeloN3] #,myeloN2 ]

#In[] 
#Extract Datasets for each and store in either a list or dictionary or something 
dist_array, centroid_Q_array, centroid_G_array, dataset_array ,distance_rerank_array= [],[],[],[],[]
for channel_set in channel_collection: 
    dist, distance_rerank, centroid_Q, centroid_G, dataset= set_up_data(channel_set[0], channel_set[1], channel_set[2], channel_set[3] )
    # dist_c, _, _, _= set_up_dataCosine(channel_set[0], channel_set[1], channel_set[2], channel_set[3] )
    dist_array.append(dist)
    distance_rerank_array.append(distance_rerank)
    cQ = [ [int(i[0]/downsize_val), int(i[1] / downsize_val)] for i in centroid_Q]
    centroid_Q = cQ 
    centroid_Q_array.append(centroid_Q) 
    # centroid_Q_array.append( [ [int(i[0]/4), int(i[1] / 4)] for i in centroid_Q]) 
    # centroid_G_array.append(centroid_G/4) 
    cG = [ [int(i[0]/downsize_val), int(i[1] / downsize_val)] for i in centroid_G]
    centroid_G = cG 
    centroid_G_array.append(centroid_G) 
    # centroid_G_array.append( [ [int(i[0]/4), int(i[1] / 4)] for i in centroid_G] ) 
    dataset_array.append(dataset) 

#In[] 

#Either Load in x queries, or denote a centroid 
print(os.listdir('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/clicks/'))
clicks = np.load('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/clicks/clicks_Swanson0.npy') 

#In[] 
print('Creating backgorund image')
brain_img_pix = np.zeros((int(29398/downsize_val), int(43054/downsize_val))) 
# brain_img_orig= resize(brain_img_orig, (int(29398/downsize_val), int(43054/downsize_val)),anti_aliasing=True)

# brain_img_pix= np.dstack((brain_img_pix, brain_img_pix, brain_img_pix))

def quickplot_overlay_brainBbx_centHEAT(cent, cent_List, Distances): 

    save_dir = '/project/roysam/rwmills/data/brain/50Plex/Sw_ds3/'
    save_path = os.path.join(save_dir, str(cent) +'.png') 
    if os.path.exists(save_dir) is False: 
        print('Making directory') 
        os.mkdir(save_dir) 
    #distnaces are sorted!!
    #image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :] 
    cent_list = cent_List.copy()
    cent_list.reverse()
    distances = Distances.copy()
    distances.reverse() 
    
    heatmap, xedges = np.histogram(distances, bins=1000)
    # print(heatmap, xedges) 
    heatmap=heatmap/np.amax(heatmap) 
    extent = [xedges[0], xedges[-1]]

    # fig = plt.figure(figsize=( 45*(43054/29398), 45 ))#works
    fig = plt.figure(figsize=( 100*(43054/29398), 100 ))

    # fig = plt.figure(figsize=( 45*(3587/2449), 45))
    # fig = plt.figure(figsize=(43054, 29398))

    plt.rcParams['axes.facecolor'] = 'black'
    
    y=[i[0] for i in cent_list] 
    x=[i[1] for i in cent_list] 
    plt.scatter(x, y, c=distances, cmap='viridis_r', s=100, marker='s') #maker ','
    # plt.scatter(x, y, c=norm(distances), cmap='viridis_r') #maker ','
    '''
    for ind, i in enumerate(cent_list ): 
        col_indx = np.where(distances[ind] <= xedges)[0][0] 
        val = heatmap[col_indx-1] 
        # plt.plot(i[1] , i[0], marker='.', color=(val,val,val), markersize=15)
        # plt.scatter(i[1] , i[0], marker='.', c=distances[ind], cmap='viridis')
        plt.scatter(i[1] , i[0], marker='.', c=val, cmap='viridis')
    '''
    # plt.plot(cent[0], cent[1], marker='*', color='red', markersize=15)
    # plt.plot(cent[1] + 10763, cent[0] + 10763, marker='*', color='red', markersize=15)
    
    plt.plot(cent[1] , cent[0], marker='x', color='red', markersize=50)
    cb = plt.colorbar() 
    cb.remove()
    '''
    y=[i[0] for i in brain_img_pix] 
    x=[i[1] for i in brain_img_pix] 
    plt.scatter(x,y, color = 'red', s=0.01)  
    '''
    plt.gca().invert_yaxis()
    # plt.imshow(brain_img_orig,cmap=plt.get_cmap('gray')) #, vmin=0, vmax=255) 
    plt.imshow(brain_img_pix,cmap=plt.get_cmap('gray')) #, vmin=0, vmax=255) 

    # plt.imshow(np.zeros(brain_img_orig.shape) , cmap=plt.get_cmap('gray') ) 
    plt.axis('off')
    # plt.show() 
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0) 
    plt.close() 

#In[] 
from random import shuffle
clicks2 = clicks 
shuffle(clicks2)
dont_redo=[] 
print('Making Images')
#plot them all 
for int_cent in clicks2:
    print(int_cent)
    sD, sI = find_several_cents_to_pt((int_cent ), centroid_Q)
    target_num = sI[0]
    print('     target: ', target_num)
    if target_num not in dont_redo: 
        print('Plotting!')
        for i, centroid_G in enumerate(centroid_G_array): 
            select_num =len(centroid_G) # 500# np.sum((s < 1)*1) 
            s = dist_array[i][target_num]
            # littles = []
            # for i in s: 
                # if i< 1: 
                    # littles.append(i)
            # print('Best cutoff: ', len(littles)/len(s)*100 ,'%') 
            sort_index = np.argsort(s)
            top10_cent_indx = [ centroid_G[i] for i in sort_index[0:select_num] ]#top 100
            quickplot_overlay_brainBbx_centHEAT(centroid_Q[target_num], top10_cent_indx, list(np.sort(s)[0:select_num]))
            # plt.close() 
    dont_redo.append(target_num)