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
from sklearn.preprocessing import LabelEncoder
from skimage.io import imsave
import gc
from statistics import mean
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import channel_lists

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
    
    
class ReadDataset(Dataset):
    # get the data from folder
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = io.imread(img_name)
        image = image.astype(np.uint8)
        if self.transform:
            image = self.transform(image)
        return image
    
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
    x_marg = int(crop_size / 2)
    y_marg = int(crop_size / 2)
    x_start = cen[0] - x_marg
    x_end = cen[0] + x_marg
    y_start = cen[1] - y_marg
    y_end = cen[1] + y_marg
    # Ensure the indices are within the image bounds
    x_start = max(0, x_start)
    x_end = min(image.shape[1], x_end)
    y_start = max(0, y_start)
    y_end = min(image.shape[0], y_end)
    return image[y_start:y_end, x_start:x_end, :]

def save_batch(batch_cells, batch_names):
    for cell, filename in zip(batch_cells, batch_names):
        imsave(filename, cell)


def main(input_dir, bbxs_file, biomarkers, dst, inside_box=[8000, 4000, 34000, 24000], topN=7000, margin=5,
            crop_size=175, parallel=True, saved_every = 5000):
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

    print('Biomarkers for this set are: ', biomarkers)
    assert os.path.isfile(bbxs_file), '{} not found!'.format(bbxs_file)
    table_orig1 = pd.read_csv(bbxs_file)
    
    
    
    # if table_orig1 has the column 'region', then use it
    # else, create a random region
    if 'region' in table_orig1.columns:
        regions = table_orig1[['region']]
        
    else:
        random_int = np.random.randint(1, 128)  
        # change the inter to str
        random_int = str(random_int)
        table_orig1['region'] = random_int
        regions = table_orig1[['region']]
        
        
    # cortex_keys, hippo_keys, thalamus_keys, ht_keys, AMGD_keys, other_keys=[],[],[],[],[],[]
    # for i in range(len(regions)):
    #     ii = regions.iloc[i].tolist()[0]
    #     if 'CX-' in ii:
    #         cortex_keys.append(ii)
    #     elif 'HC-' in ii:
    #         hippo_keys.append(ii)
    #     elif 'HT-' in ii:
    #         ht_keys.append(ii)
    #     elif'TH-' in ii:
    #         thalamus_keys.append(ii)
    #     elif 'AMGD-' in ii:
    #         AMGD_keys.append(ii)
    #     else:
    #         other_keys.append(ii)
    # cortex_keys, hippo_keys, thalamus_keys, ht_keys, AMGD_keys, other_keys = list(set(cortex_keys)),list(set(hippo_keys)), list(set(thalamus_keys)), list(set(ht_keys)), list(set(AMGD_keys)), list(set(other_keys))
    
    #make sure the regions are str not int
    table_orig1['region'] = table_orig1['region'].astype(str)
    
    lab = LabelEncoder()
    enc_labs=[]
    #eliminate left and right specificity for the sake of the labeling the areas
    for count, i in enumerate(table_orig1['region']) :
        # print(i)
        if 'L-' in i:
            newi=i.split('L-')[1]
            enc_labs.append(newi)

        elif 'R-' in i:
            newi=i.split('R-')[1]
            enc_labs.append(newi)

        else:
            enc_labs.append(i)
    table_orig1['encoded_labels'] = enc_labs
    table_orig1['encoded_labels'] = lab.fit_transform(table_orig1['encoded_labels'])

    example_key = list(biomarkers.keys())[0]
    #if the name contains S1_, get the name
    # else, use _ as the separator
    try:
        image_size = io.imread_collection(os.path.join(input_dir, biomarkers[example_key].split('_')[1]), plugin='tifffile')[0].shape
        images = np.zeros((image_size[0], image_size[1], len(biomarkers)), dtype=np.uint16)
        # for each biomarker read the image and replace the black image if the channel is defined
        for i, bioM in enumerate(biomarkers.keys()):
            if biomarkers[bioM] != "":
                images[:, :, i] = io.imread(os.path.join(input_dir, biomarkers[bioM].split('_')[1]))
        
    except:
        image_size = io.imread_collection(os.path.join(input_dir, biomarkers[example_key]), plugin='tifffile')[0].shape
        images = np.zeros((image_size[0], image_size[1], len(biomarkers)), dtype=np.uint16)
        
        for i, bioM in enumerate(biomarkers.keys()):
            if biomarkers[bioM] != "":
                images[:, :, i] = io.imread(os.path.join(input_dir, biomarkers[bioM]))
        
        
        
    # mini_mask = table_orig1
    allKeys = list(biomarkers.keys())
    # bbxs_chanel_list = ['NeuN','S100','Olig2','Iba1','RECA1','Cleaved Caspase-3','Tyrosine Hydroxylase','Blood Brain Barrier','GFP',
    #                     'PDGFR beta','Parvalbumin','Choline Acetyltransferase','GFAP','Smooth Muscle Actin','Glutaminase','Doublecortin',
    #                     'Sox2','PCNA','Vimentin','GAD67','Tbr1','Eomes','Calretinin','Nestin','Aquaporin-4','Calbindin']

    cents= table_orig1[['centroid_x', 'centroid_y']].values
    bbxs = table_orig1[['xmin', 'ymin', 'xmax', 'ymax']].values
    
    if 'NeuN' in table_orig1.columns:
        labels = table_orig1['NeuN'].values # , 'RECA1']].values #cell type values
    else:
        labels = None
    e_labels = table_orig1[['encoded_labels']].values
    cell_ids = table_orig1[['ID']].values

    del table_orig1 #remove table to save space
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(dst, 'bounding_box_train'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'bounding_box_test'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'query'), exist_ok=True)

 
    crop_shape = (crop_size, crop_size)
    batch_cells = []
    batch_names = []

    for i, cen in enumerate(cents):
        # -- 1) Crop and resize the cell --
        cell = get_crop_no_pad(images, cen, crop_size)
        cell = zero_pad(cell, crop_size)
        cell = resize(cell, crop_shape, mode='constant', preserve_range=True)
        
        # -- 2) Construct the output filename info --
        c_id = cell_ids[i][0]
        if labels is not None:
            y = labels[i]  # get the class ID
            try: 
                y = np.where(y == 1)[0][0]
            except:
                y = np.where(y == 1)[0]
        else:
            y = 1
        
        cent = f"{cents[i][0]}_{cents[i][1]}"  # atlas location
        loc = e_labels[i][0]
        pid = str(loc).zfill(3)               # cell identifier

        # Decide which set to put the cell into
        if i % 2 == 0:
            dst_set = os.path.join(dst, 'bounding_box_test')
        elif i % 51 == 0:
            dst_set = os.path.join(dst, 'query')
        else:
            dst_set = os.path.join(dst, 'bounding_box_train')
        
        new_name = os.path.join(
            dst_set, f"{pid}_c{cent}s1_{str(c_id).zfill(6)}.tif"
        )

        if np.amax(cell) > 0:
            batch_cells.append(cell)
            batch_names.append(new_name)
        else:
            print(f"Image is just black: {cell.shape}, continuing...")
        
        # -- 4) If we’ve hit 5000 samples (or the end), save them to disk --
        # Use (i+1) because `i` starts from 0; we want to save at 5000, 10000, ...
        if (i + 1) % saved_every == 0:
            save_batch(batch_cells, batch_names)
            print(f"Saved up to sample index {i+1} ...")
            # Reset containers for the next chunk
            batch_cells = []
            batch_names = []
    
   
    if batch_cells:
        save_batch(batch_cells, batch_names)
        print(f"Saved the final batch of {len(batch_cells)} cells.")

    preprocess = transforms.Compose([ transforms.ToTensor(),])
    dataset_path = os.path.join(dst, 'bounding_box_train')
    dataset = ReadDataset(dataset_path, transform=preprocess)
    test_set = ReadDataset(os.path.join(dst, 'bounding_box_test'), transform=preprocess)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1, pin_memory=False,)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,num_workers=1, pin_memory=False,)
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(data_loader):
        batch = batch_target # size: [1,20,175,175]
        
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)
    for i_batch, batch_target in enumerate(test_loader):

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
    print('Mean: ', mean)
    print('Std: ', std)
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    # Save the mean and std to 'mean_std.txt'
    with open(os.path.join(dst, 'mean_std.txt'), 'w') as f:
        # Write the mean values as the first row
        f.write(' '.join([f'{m:.6f}' for m in mean_list]) + '\n')
        # Write the std values as the second row
        f.write(' '.join([f'{s:.6f}' for s in std_list]) + '\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_DIR', type=str, default='/project/xwu/lhuang37/roysam/data/brain/50Plex/S1/final', help='/path/to/input/dir')
    parser.add_argument('--OUTPUT_DIR', type=str, default='/data/',help='/path/to/output/dir')
    parser.add_argument('--BBXS_FILE', type=str, default='/project/xwu/lhuang37/roysam/data/brain/50Plex/S1/regions_table.csv',help='/path/to/bbxs/file')
    parser.add_argument('--CROP_SIZE', type=int, default=175, help='Patch size')
    parser.add_argument('--BIO_MARKERS_PANNEL', type=str, default='cell_type_panel',help='panel name')
    args = parser.parse_args()


    start = time.time()
     
    #read the biomarkers rom channel_lists.py which contains several dictionaries
    biomarkers = getattr(channel_lists, args.BIO_MARKERS_PANNEL)
    print('biomarkers',biomarkers)
    main(args.INPUT_DIR, args.BBXS_FILE, biomarkers=biomarkers,
            dst=args.OUTPUT_DIR, inside_box=[8000, 4000, 34000, 24000], parallel=True, margin=5, crop_size=args.CROP_SIZE, topN=5000) # crop_size 175    print('*' * 50)
    print('*' * 50)
    duration = time.time() - start
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    print('Data prepared for training in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))
    print('Data saved in {}'.format(args.OUTPUT_DIR))
