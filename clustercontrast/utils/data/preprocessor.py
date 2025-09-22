from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
from skimage.io import imread 


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = imread(fpath)
        # try: 
        #     #img = Image.open(fpath).convert('RGB')
        #     img = imread(fpath) #RWM read in multi channel image
        # except: 
        #     #img = imread(fpath) #RWM read in multi channel image
        #     img = np.load(fpath) #for Lins stuff 
        #     # img = Image.fromarray(imgnp, mode="CMYK") #convert to PIL 

        if self.transform is not None:
            # print(img.shape, type(img)) 
            #try: 
            # if img.dtype == 'uint16': 
            #     #make the type change for S2 and S3 
            #     img = img.astype('float64') 

            img = self.transform(img)
            #except: 
                #img = (img/256).astype('uint8')

        return img, fname, pid, camid, index
