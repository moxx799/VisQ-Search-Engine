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
    # # for not-existing channel put ''
    # biomarkers = {'DAPI': channel_names[0],
    #                 'Histones': channel_names[1],
    #                 'NeuN': channel_names[2],
    #                 'S100': channel_names[3],
    #                 'Olig2': channel_names[4],
    #                 'Iba1': channel_names[5],
    #                 'RECA1': channel_names[6]}
    biomarkers = {'DAPI': channel_names[0],
                  'Histones': channel_names[1],
                  'NeuN': channel_names[2],
                  'S100': channel_names[3],
                  'Olig2': channel_names[4],
                  'Iba1': channel_names[5],
                  'RECA1': channel_names[6],
                  'other1': channel_names[7],
                  'other2': channel_names[8],
                  'other3': channel_names[9]}

    # read bbxs file
    assert os.path.isfile(bbxs_file), '{} not found!'.format(bbxs_file)
    table_orig1 = pd.read_csv(bbxs_file)
    # cents_bbx = table_orig1[['centroid_x', 'centroid_y','xmin', 'ymin', 'xmax', 'ymax']].values
    # bbxs = table_orig1[['xmin', 'ymin', 'xmax', 'ymax']].values
    # labels = table_orig1[['NeuN','S100','Olig2','Iba1','RECA1']].values #cell type values

    print('keys: ', table_orig1.keys() )
    # get bounding boxes in the center of brain
    # get images
    image_size = io.imread_collection(os.path.join(input_dir, biomarkers['NeuN']), plugin='tifffile')[0].shape
    images = np.zeros((image_size[0], image_size[1], len(biomarkers)), dtype=np.uint16)
    # for each biomarker read the image and replace the black image if the channel is defined
    for i, bioM in enumerate(biomarkers.keys()):
        if biomarkers[bioM] != "":
            images[:, :, i] = io.imread(os.path.join(input_dir, biomarkers[bioM]))

    mini_mask = table_orig1 #copy table

    # inidividual cells for boundingbox
    neun=mini_mask[mini_mask['NeuN']==1]
    s100 = mini_mask[mini_mask['S100']==1]
    olig2 = mini_mask[mini_mask['Olig2']==1]
    iba1= mini_mask[mini_mask['IBA1']==1] #sometimes change
    reca1= mini_mask[mini_mask['RECA1'] == 1]

    #cells types
    cls_specific = neun
    cls_specific= cls_specific.append(s100)
    cls_specific= cls_specific.append(olig2)
    cls_specific= cls_specific.append(iba1) #no microglia or endotelial cells for internal regions but try microglia for cortex
    cls_specific= cls_specific.append(reca1)

    cents= cls_specific[['centroid_x', 'centroid_y']].values
    bbxs = cls_specific[['xmin', 'ymin', 'xmax', 'ymax']].values
    labels = cls_specific[['NeuN','S100','Olig2','IBA1', 'RECA1']].values #cell type values
    # e_labels = cls_specific[['encoded_labels']].values
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
        zero_pad_x = partial(zero_pad, dim=max_dim)
        with multiprocessing.Pool(processes=cpus) as pool:
            new_cells = pool.map(zero_pad_x, cells)
    else:
        new_cells = [zero_pad(cell, max_dim) for cell in cells]
    print('1. Size of cells is: ', len(new_cells))

    del cells #remove cells to save space

    # resize image specific size
    if parallel:
        resize_x = partial(resize, output_shape=crop_size, mode='constant', preserve_range=True)
        with multiprocessing.Pool(processes=cpus) as pool:
            cells = pool.map(resize_x, new_cells)
    else:
        cells = [resize(cell, crop_size, mode='constant', preserve_range=True) for cell in new_cells]
    # print('We finished with the whole set, size of new_new_cells is: ', len(new_new_cells))


    # new_cells = [(img16/256).astype('uint8') for img16 in cells]

    # print('1. Size of cells is: ', len(new_new_cells))

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
        y= np.where(y==1)[0][0]
        cent = str(cents[i][0]) + '_' + str(cents[i][1])  #get the atlas location
        loc =y #  e_labels[i][0]
        pid = str(loc).zfill(3) # str(y) + str(loc).zfill(3) #get the cell identifier
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
    parser.add_argument('--other2', type=str, default='R1C6.tif', help='<RECA1.tif> | None')
    parser.add_argument('--other3', type=str, default='R1C6.tif', help='<RECA1.tif> | None')

    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()
    print(args, unknown_args)

    start = time.time()
    main(args.INPUT_DIR, args.BBXS_FILE, [args.DAPI, args.HISTONES, args.NEUN, args.S100, args.OLIG2, args.IBA1, args.RECA1, args.other1, args.other2, args.other3 ],
            args.OUTPUT_DIR, inside_box=[8000, 4000, 34000, 24000], parallel=True, margin=5, crop_size=(150, 150), topN=5000) # crop_size 175
    print('*' * 50)
    print('*' * 50)
    duration = time.time() - start
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    print('Data prepared for training in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))
    print('Data saved in {}'.format(args.OUTPUT_DIR))
