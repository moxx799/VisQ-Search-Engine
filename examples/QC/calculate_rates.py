import h5py 
import os 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('/project/roysam/rwmills/repos/cluster-contrast-reid/clustercontrast/') 
print(os.getcwd() )
from QC_functions import * 
#---------------------------------------------------------------------------------------------------------

table_orig1 = pd.read_csv('/project/roysam/rwmills/data/brain/50Plex/S1/classification_results/regions/regions_table.csv') 

lab = LabelEncoder()

#perform label encoding on 'team' column
# table_orig1['encoded_labels'] = table_orig1['region']
enc_labs=[]
#eliminate left and right specificity for the sake of the labeling the areas
for count, i in enumerate(table_orig1['region']) :
    # print(i)
    if 'L-' in i:
        newi=i.split('L-')[1]
        # print(newi)
        # print(table_orig1['encoded_labels'][count])
        enc_labs.append(newi)
        # table_orig1['encoded_labels'][count] == newi
    elif 'R-' in i:
        newi=i.split('R-')[1]
        # print(newi)
        # print(table_orig1['encoded_labels'][count])
        enc_labs.append(newi)
        # table_orig1['encoded_labels'][count] == newi
    else:
        enc_labs.append(i)
table_orig1['encoded_labels'] = enc_labs
table_orig1['simplified_labels'] = enc_labs

# print(table_orig1['encoded_labels'].head())
table_orig1['encoded_labels'] = lab.fit_transform(table_orig1['encoded_labels'])

layer1 = table_orig1[table_orig1['encoded_labels'] == 25]
layer23 = table_orig1[table_orig1['encoded_labels'] == 26]
layer4 = table_orig1[table_orig1['encoded_labels'] == 27]
layer5 = table_orig1[table_orig1['encoded_labels'] == 28]
layer6a = table_orig1[table_orig1['encoded_labels'] == 29]
layer6b = table_orig1[table_orig1['encoded_labels'] == 30]

#hippocampal layers: 
hlayer1 = table_orig1[table_orig1['encoded_labels'] == 38]
hlayer2 = table_orig1[table_orig1['encoded_labels'] == 39]
hlayer3 = table_orig1[table_orig1['encoded_labels'] == 40]
c = table_orig1[table_orig1['encoded_labels'] == 41]
hlayerfi = table_orig1[table_orig1['encoded_labels'] == 42]
#--------------------------------------------------------------------------------------------------------

#Define the library of networks: 
#       Paths for each trained network by name 
# cell_type1 = ['cellType_5','all_brain_Cells', 75, 'brain' ] #dn, dn2, data_size, data_type 
cell_type2 = ['cellType_5','UNet_SAEA_cltl_cell0_TESTALL', 75, 'brain' ] #dn, dn2, data_size, data_type 
myelo0 = ['all_brain_noRECA_FULL','all_brain_noRECA_IM', 150, 'brain' ] #dn, dn2, data_size, data_type 

no_reca = ['all_brain_noRECA', 'all_brain_noRECA_IM', 150, 'brain'] #neuronal phenotypic markers 
no_reca_unet = ['all_brain_noRECA', 'UNet_SAEA_cltl_allBrainNoReca', 150, 'brain'] 
# myeloN1 = [  'connectivity0_noRECA_NeuN', 'UNet_CA_myelo' , 150, 'brain'] # connectivity0_S10N
# myeloN2 = [  'connectivity0_noRECA_NeuN', 'UNet_SAEA_myelo2' , 150, 'brain'] # 
myeloN3 = [  'connectivity0_noRECA_NeuN_TESTALL', 'UNet_ftTL_myelo2' , 150, 'brain'] # 
vasculature = [  'valsulature1_TESTALL', 'UNet_SAEA_cltl_vas0' , 150, 'brain'] # 
cyto_glial = [  'connectivity0_noRECA_Glial_TESTALL', 'UNet_SAEA_cltl_glial0' , 150, 'brain'] # 
neun_panel= [  'NeuronalMarkers_all_TESTALL', 'UNet_SAEA_cltl_JustNeuN0' , 175, 'brain'] #
glial_panel= [  'Glial_NoRECA_ALLTEST', 'UNet_SAEA_cltl_Justglial0' , 175, 'brain'] #


phenotype = []
death = [] 

downsize_val = 1
# downsize_vala=(29398/sw.shape[0]) 
# downsize_valb=(43054/sw.shape[1])

channel_collection = [myeloN3, neun_panel, glial_panel, vasculature] 


#Extract Datasets for each and store in either a list or dictionary or something 
dist_array, centroid_Q_array, centroid_G_array, dataset_array ,distance_rerank_array= [],[],[],[],[]
for channel_set in channel_collection: 
    dist, distance_rerank, centroid_Q, centroid_G, dataset= set_up_data(channel_set[0], channel_set[1], channel_set[2], channel_set[3] )
    # dist, _, _, _= set_up_dataCosine(channel_set[0], channel_set[1], channel_set[2], channel_set[3] )
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
    dataset_array.append(dataset) #add images 


#--------------------------------------

#pick your query 

layer_list = [layer1, layer23, layer4, layer5, layer6a, layer6b, hlayer1, 
              hlayer2, hlayer3, c, hlayerfi] 

for layer in layer_list: 
    samplesi = len(layer) 
    #--------------------------------------------
    query_list = [] 
    index_list=[] 
    for i in range(samplesi): 
        x_val, y_val, reg = layer.iloc[i]['centroid_x'], layer.iloc[i]['centroid_y'], layer.iloc[i]['encoded_labels']
        if [x_val, y_val] not in query_list: 
            query_list.append([y_val, x_val])

    # Plot the query location for each set: 
    all_good=[] 
    target_profile=[] 
    for ii, int_cent1 in enumerate(query_list) : 
        good_num = 0 
        for i, centroid_Q in enumerate(centroid_Q_array): #if there's only one set of results we're going through, i should always be 0
            sD1a, sI1a = find_several_cents_to_pt((int_cent1 ), centroid_Q)
            indx =sI1a[0] #this is the query 
            if indx in target_profile: 
                continue
            else: 
                target_profile.append(indx) 
                #--------------------------------------------------------------------
                s = dist_array[i][indx] # samples to the query 
                sort_index = np.argsort(s)[0:len(layer)]  #sorted samples 
                for j in sort_index: #going through each similar sample 
                    # print(dataset_array[i].gallery[j][1]) 
                    search_val=dataset_array[i].gallery[j][1]
                    if search_val == layer['encoded_labels'].iloc[0]: 
                        # print(dataset_array[i].gallery[j])
                        good_num+=1 #have it in common 
                all_good.append(good_num / len(layer))
                top_similar = [ centroid_G_array[i][ii] for ii in sort_index[0:len(layer)] ]
                # quickplot_overlay_brainBbx_cent(query_list[0], top_similar)
                # print(makeIOU(query_list, top_similar)) 
                # if ii%25 ==0: 
                    # print(np.mean(all_good))
    print(np.mean(all_good) )

