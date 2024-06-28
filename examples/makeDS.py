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
from skimage.io import imsave
import gc
from statistics import mean
import torch
import torchvision.transforms as transforms



# functions
def select_random(carray):
    carray.sample(frac=1)[:5000]
    return carray.sample(frac=1)[:5000]

def zero_pad(image, dim):
    """
    pad zeros to the image in the first and second dimension
    :param image: image array [width*height*channel]
    :param dim: new dimension
    :return: padded image
    """
    pad_width = ((np.ceil((dim - image.shape[0]) / 2), np.floor((dim - image.shape[0]) / 2)),
                    (np.ceil((dim - image.shape[1]) / 2), np.floor((dim - image.shape[1]) / 2)),
                    (0, 0))
    return np.pad(image, np.array(pad_width, dtype=int), 'constant')

def get_crop(image, bbx, margin=20):
    """
    crop large image with extra margin
    :param image: large image
    :param bbx: [xmin, ymin, xmax, ymax]
    :param margin: margin from each side
    :return:
    """
    # Y-coordinates: ymin - margin : ymax + margin
    # X-coordinates: xmin - margin : xmax + margin
    # Z-coordinates: everything (:)
    try:
        return image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :]
    except:
        return image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin]

def get_crop_no_pad(image, cen, crop_size):
    x_marg = int(crop_size[0] / 2 )
    y_marg = int(crop_size[1] / 2)
    try:
        # return image[cen[0] -x_marg : cen[0] + x_marg, cen[1] - y_marg : cen[1] + y_marg, :]
        return image[cen[1] -x_marg : cen[1] + x_marg, cen[0] - y_marg : cen[0] + y_marg, :]
    except:
        # return image[cen[0] -x_marg : cen[0] + x_marg, cen[1] - y_marg : cen[1] + y_marg]
        return image[cen[1] -x_marg : cen[1] + x_marg, cen[0] - y_marg : cen[0] + y_marg]

    #main script
def main(input_dir, bbxs_file, channel_names, dst, inside_box=[8000, 4000, 34000, 24000], topN=7000, margin=5,
            crop_size=(50, 50), parallel=True):
    """
    Prepare training dataset file (data.h5) to train CapsNet for cell type classification
    :param input_dir: Path to the input dir containing biomarker images
    :param bbxs_file: Path to the bbxs_detection.txt file generated from cell nuclei detectio module
    :param channel_names: List of filnames for channels in the order: [dapi, histone, neun, s100, olig2, iba1, reca1]
    :param output_dir: Path to save the h5 file
    :param inside_box: Select cell inside this box to skip cells in the border with false phenotyping
    :param topN: Select top N brightest cells in each channel
    :param margin: Add extra margin to each dimension to capture information in soma
    :param crop_size: Size of the bounding box after reshape (input to the network)
    :param parallel: Process the file in multiprocessing or not
    :return:
    """

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default

    # prepare dict for biomarkers
    # for not-existing channel put ''
    # biomarkers = {'DAPI': channel_names[0],
    #                 'Histones': channel_names[1],
    #                 'NeuN': channel_names[2],
    #                 'S100': channel_names[3] , 
    #                 'Olig2': channel_names[4] ,
    #                 'Iba1': channel_names[5],
    #                 'RECA1': channel_names[6]}
    biomarkers = {'DAPI': channel_names[0],
                  'Histones': channel_names[1],
                  'NeuN': channel_names[2],
                  'S100': channel_names[3],
                  'Olig2': channel_names[4] ,
                  'Iba1': channel_names[5],
                  'RECA1': channel_names[6],
                  'other1': channel_names[7]} # ,
    #               'other2': channel_names[8],
    #               'other3': channel_names[9]}
    print('Biomarkers for this set are: ', biomarkers)
    # read bbxs file
    assert os.path.isfile(bbxs_file), '{} not found!'.format(bbxs_file)
    table_orig1 = pd.read_csv(bbxs_file)
    # cents_bbx = table_orig1[['centroid_x', 'centroid_y','xmin', 'ymin', 'xmax', 'ymax']].values
    # bbxs = table_orig1[['xmin', 'ymin', 'xmax', 'ymax']].values
    # labels = table_orig1[['NeuN','S100','Olig2','Iba1','RECA1']].values #cell type values

    #load the mask file - we have 5 general regions here
    regions = table_orig1[['region']]
    # cortex_keys = ['R-CX-Pir', 'L-CX-Pir', 'L-CX-PLCo', 'L-CX-ACo', 'R-CX-PLCo', 'R-CX-ACo', 'L-CX-Ven', 'R-CX-Ven', 'L-CX-Den', 'R-CXX-AIP', 'R-CX-DI', 'L-CX-DI', 'R-CX-GI', 'L-CX-GI', 'R-CX-S2', 'L-CX-S2', 'R-CX-S1ULp', 'L-CX-S1ULp', 'R-CX-S1BF-LAYR-6A', 'L-CX-S1BF-LAYER-6B', 'L-CX-S1BF-LAYER-6A', 'R-CX-S1BF-LAYER-5', 'L-CX-S1BF-LAYER-5', 'R-CX-S1BF-LAYER-4', 'R-CX-S1BF-LAYER-4', 'R-CX-S1BF-LAYER-1', 'L-CX-S1BF-LAYER-2_3', 'L-CX-S1BF-LAYER-1', 'R-CX-S1DZ', 'R-CX-RSGc', 'L-CX-CX-S1Sh', 'L-CX-S1Sh', 'R-CX-S1HL', 'L-CX-S1HL', 'R-CX-M1', 'R-CX-M2', 'R-CX-RSD', 'L-CX-RSD', 'L-CX-M1', 'L-CX-M2']
    # hippo_keys = ['L-HT-VMHC', 'R-HT-VMHC', 'R-HC-FI', 'L-HC-FI', 'R-HC-DG', 'L-HC-DG', 'R-HC-CA3', 'L-HC-CA3', 'R-HC-CA1', 'L-HC-CA12']
    # thalamus_keys = ['R-TH-ic', 'L-TH-ic', 'R-TH-EP', 'TH-PaXi', 'L-TH-EP', 'TH-Xi', 'TH-Re', 'R-TH-VRe', 'L-TH-VRe', 'R-TH-mt', 'R-TH-SubV', 'L-TH-SubV', 'L-TH-mt', 'R-TH-VM', 'L-TH-VM', 'L-TH-SubD', 'TH-Rh', 'R-TH-SubD', 'L-TH-Rt', 'L-TH-VPL', 'R-TH-Rt', 'R-TH-VPL', 'R-TH-VL+VA', 'L-TH-VL+VA', 'L-TH-AM', 'R-TH-AM', 'TH-CM+PC+CL', 'L-TH-MDL', 'R-TH-MDL', 'L-TH-MDM', 'R-TH-MDM', 'R-TH-AVDM', 'TH-PV', 'L-TH-AVDM', 'R-TH-LDLV', 'L-TH-LDLV', 'L-TH-LDDM', 'R-TH-LDDM', 'R-TH-sm', 'R-TH-LHb', 'R-TH-MHb', 'L-TH-sm', 'L-TH-LHb', 'L-TH-MHb', 'R-TH-AD', 'L-TH-AD']
    # ht_keys = ['L-HT-ArcL', 'R-HT-ArcL', 'L-HT-ArcM', 'R-HT-ArcM', 'R-HT-VMHSh', 'R-HT-VMHVL', 'L-HT-VMHSh', 'L-HT-VMHVL', 'L-HT-ArcD', 'L-HT-VMHC', 'R-HT-ArcD', 'R-HT-VMHC', 'L-HT-VMHDM', 'R-HT-VMHDM', 'L-HT-DMD', 'R-HT-DMD', 'R-HT-AHP', 'L-HT-AHP', 'R-HT-Stg', 'L-HT-DA', 'R-HT-DA', 'L-HT-Stg', 'R-HT-A13', 'L-HT-A13', 'R-HT-PaPo', 'L-HT-PaPo']
    # AMGD_keys = ['L-AMGD-BMA', 'L-AMGD-BLV', 'R-AMGD-BLV', 'R-AMGD-BMA', 'L-AMGD-MeAV', 'L-AMGD-BLA', 'L-AMGD-MeAD', 'R-AMGD-BLA', 'R-AMGD-MeAV', 'L-AMGD-STIA', 'R-AMGD-MeAD', 'R-AMGD-STIA', 'L-AMGD-CeC', 'L-AMGD-CeM', 'L-AMGD-LaDL', 'R-AMGD-CeC', 'R-AMGD-CeM', 'R-AMGD-LaDL', 'L-AMGD-CeL', 'R-AMGD-CeL']
    # other_keys = ['L-RcHL', 'MEE', 'R-RcHL', 'V3V', 'MEI', 'R-SOR', 'R-mbf', '3V-EP', 'L-SOR', 'L-mbf', 'Pe', 'L-opt', 'CC', 'L-sox', 'R-opt', 'R-f', 'R-sox', 'L-f', 'L-STR-CPu ', 'R-STR-CPu', 'L-STR-GP', 'R-STR-GP', 'R-ZIR', 'L-ZIR', 'R-STR-st', 'L-STR-st', 'R-SVZ', 'D3V', 'R-LV', 'L-SVZ', 'L-LV', 'R-CG', 'L-CG']
    #create instance of label encoder
    cortex_keys, hippo_keys, thalamus_keys, ht_keys, AMGD_keys, other_keys=[],[],[],[],[],[]
    for i in range(len(regions)):
        ii = regions.iloc[i].tolist()[0]
        if 'CX-' in ii:
            cortex_keys.append(ii)
        elif 'HC-' in ii:
            hippo_keys.append(ii)
        elif 'HT-' in ii:
            ht_keys.append(ii)
        elif'TH-' in ii:
            thalamus_keys.append(ii)
        elif 'AMGD-' in ii:
            AMGD_keys.append(ii)
        else:
            other_keys.append(ii)
    cortex_keys, hippo_keys, thalamus_keys, ht_keys, AMGD_keys, other_keys = list(set(cortex_keys)),list(set(hippo_keys)), list(set(thalamus_keys)), list(set(ht_keys)), list(set(AMGD_keys)), list(set(other_keys))

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
    # print(table_orig1['encoded_labels'].head())
    table_orig1['encoded_labels'] = lab.fit_transform(table_orig1['encoded_labels'])

    # labels = table_orig1[['Parvalbumin', 'CNPase', 'GFAP', 'NeuN', 'Olig2', 'Tyrosine', 'S100'  ]].values
    # get bounding boxes in the center of brain
    # get images
    image_size = io.imread_collection(os.path.join(input_dir, biomarkers['NeuN']), plugin='tifffile')[0].shape
    images = np.zeros((image_size[0], image_size[1], len(biomarkers)), dtype=np.uint16)
    # for each biomarker read the image and replace the black image if the channel is defined
    for i, bioM in enumerate(biomarkers.keys()):
        if biomarkers[bioM] != "":
            images[:, :, i] = io.imread(os.path.join(input_dir, biomarkers[bioM]))

    #now lets filter so that we only grab ones in certain regions
    # desired_combo = hippo_keys + thalamus_keys + ht_keys + other_keys
    # desired_combo = cortex_keys
    mini_mask = table_orig1 #all regions
    # mini_mask = pd.DataFrame(table_orig1.loc[table_orig1['region'] == desired_combo[0]])
    # for i in desired_combo[1::]:
    #     a=table_orig1.loc[table_orig1['region'] == i]
    #     mini_mask = mini_mask.append(a) #build up table with desired regions

    # inidividual cells for boundingbox
    neun=mini_mask[mini_mask['NeuN']==1]
    # s100 = mini_mask[mini_mask['S100']==1]
    # olig2 = mini_mask[mini_mask['Olig2']==1]
    # iba1= mini_mask[mini_mask['Iba1']==1]
    reca1= mini_mask[mini_mask['RECA1'] == 1]

    #cells types
    cls_specific = neun
    # cls_specific= cls_specific.append(s100)
    # cls_specific= cls_specific.append(olig2)
    # cls_specific= cls_specific.append(iba1) #no microglia or endotelial cells for internal regions but try microglia for cortex
    cls_specific= cls_specific.append(reca1)

    cents= cls_specific[['centroid_x', 'centroid_y']].values
    bbxs = cls_specific[['xmin', 'ymin', 'xmax', 'ymax']].values
    # labels = cls_specific[['NeuN','S100','Olig2','Iba1']].values # , 'RECA1']].values #cell type values
    # labels = cls_specific[['NeuN','S100','Olig2','Iba1', 'RECA1']].values #cell type values
    labels = cls_specific[['NeuN', 'RECA1']].values # , 'RECA1']].values #cell type values
    e_labels = cls_specific[['encoded_labels']].values
    cell_ids = cls_specific[['ID']].values

    del table_orig1 #remove table to save space

    #prepare specified Data
    # cells = [get_crop(images, bbx, margin=100 ) for bbx in bbxs]
    cells = [get_crop_no_pad(images, cen, crop_size ) for cen in cents]

    del images #delete images to save space
    gc.collect() #free memory

    print('1. Size of cells is: ', len(cells) )
    print('1. Size of first entry is: ', cells[0].shape)
    print('1. Size of second entry is: ', cells[1].shape)
    print('1. Size of third entry is: ', cells[2].shape)

    # max_dim = np.max([cell.shape[:2] for cell in cells])  # find maximum in each dimension
    max_dim = crop_size[0]
    if parallel:
        print('Padding')
        zero_pad_x = partial(zero_pad, dim=max_dim)
        with multiprocessing.Pool(processes=cpus) as pool:
            new_cells = pool.map(zero_pad_x, cells)
    else:
        new_cells = [zero_pad(cell, max_dim) for cell in cells]
    print('1. Size of cells is: ', len(new_cells))
    print('1. Size of cells is: ', len(cells) )
    print('1. Size of first entry is: ', cells[0].shape)
    print('1. Size of second entry is: ', cells[1].shape)
    print('1. Size of third entry is: ', cells[2].shape)
    del cells #remove cells to save space
    gc.collect() #free memory

    # resize image specific size
    if parallel:
        print('Resizing')
        resize_x = partial(resize, output_shape=crop_size, mode='constant', preserve_range=True)
        with multiprocessing.Pool(processes=cpus) as pool:
            cells = pool.map(resize_x, new_cells)
    else:
        cells = [resize(cell, crop_size, mode='constant', preserve_range=True) for cell in new_cells]
    print('1. Size of cells is: ', len(new_cells))
    print('1. Size of cells is: ', len(cells) )
    print('1. Size of first entry is: ', cells[0].shape)
    print('1. Size of second entry is: ', cells[1].shape)
    print('1. Size of third entry is: ', cells[2].shape)
    print('We finished with the whole set, size of new_new_cells is: ', len(cells))


    # new_cells = [(img16/256).astype('uint8') for img16 in cells]

    print('1. Size of cells is: ', len(cells))

    # cells=np.array(new_new_cells) # cells_save)

    del new_cells
    gc.collect() #free memory



    #now lets do the stuff for reid
    #make the folders
    if os.path.exists(dst) is False:
        os.mkdir(dst)
    if os.path.exists(os.path.join(dst, 'bounding_box_train')) is False:
        os.mkdir(os.path.join(dst, 'bounding_box_train'))
    if os.path.exists(os.path.join(dst, 'bounding_box_test')) is False:
        os.mkdir(os.path.join(dst, 'bounding_box_test'))
    if os.path.exists(os.path.join(dst, 'query')) is False:
        os.mkdir(os.path.join(dst, 'query'))

    #create naming schematic and save:
    for i, x in enumerate(cells):
        c_id = cell_ids[i][0]
        y=labels[i] #get the class ID
        try: 
            y= np.where(y==1)[0][0]
        except: 
            y=np.where(y==1)[0] 
        cent = str(cents[i][0]) + '_' + str(cents[i][1])  #get the atlas location
        loc = e_labels[i][0]
        pid = str(loc).zfill(3) # str(y) + str(loc).zfill(3) #get the cell identifier
        # pid = str(y).zfill(3) #cellt ype 
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




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_DIR', type=str, default='/project/ece/roysam/datasets/TBI/G4_mFPI_Li+VPA/G4_BR#19_HC_12R/final', help='/path/to/input/dir')
    parser.add_argument('--OUTPUT_DIR', type=str, default='/project/ece/roysam/rachel/brain/partition_brains/h5_files',help='/path/to/output/dir')
    parser.add_argument('--BBXS_FILE', type=str, default='/project/ece/roysam/datasets/TBI/G4_mFPI_Li+VPA/G4_BR#19_HC_12R/detection_results/bbxs_detection.txt',help='/path/to/bbxs/file')
    parser.add_argument('--DAPI', type=str, default='R1C1.tif', help='<dapi.tif> | None')
    parser.add_argument('--HISTONES', type=str, default='', help='<histones.tif> | None')
    parser.add_argument('--NEUN', type=str, default='R1C4.tif', help='<NeuN.tif> | None')
    parser.add_argument('--S100', type=str, default='R2C6.tif', help='<S100.tif> | None')
    parser.add_argument('--OLIG2', type=str, default='', help='<Olig2.tif> | None')
    parser.add_argument('--IBA1', type=str, default='R1C7.tif', help='<Iba1.tif> | None')
    parser.add_argument('--RECA1', type=str, default='R1C6.tif', help='<RECA1.tif> | None')
    parser.add_argument('--other1', type=str, default='R1C6.tif', help='<RECA1.tif> | None')
    # parser.add_argument('--other2', type=str, default='R1C6.tif', help='<RECA1.tif> | None')
    # parser.add_argument('--other3', type=str, default='R1C6.tif', help='<RECA1.tif> | None')

    args = parser.parse_args()
    # args, unknown_args = parser.parse_known_args()
    # print(args, unknown_args)

    start = time.time()
    main(args.INPUT_DIR, args.BBXS_FILE, [args.DAPI, args.HISTONES, args.NEUN, args.S100, args.OLIG2 , args.IBA1, args.RECA1 , args.other1], #, args.other2, args.other3 ],
            args.OUTPUT_DIR, inside_box=[8000, 4000, 34000, 24000], parallel=True, margin=5, crop_size=(175, 175), topN=5000) # crop_size 175    print('*' * 50)
    print('*' * 50)
    duration = time.time() - start
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    print('Data prepared for training in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))
    print('Data saved in {}'.format(args.OUTPUT_DIR))
