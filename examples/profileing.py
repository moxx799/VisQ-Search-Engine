from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd  # Import pandas
print('We are here: ', os.getcwd())
sys.path.append('/home/lhuang37/repos/VisQ-Search-Engine/')
sys.path.append('/home/lhuang37/repos/VisQ-Search-Engine/clusterconstrast')
from clustercontrast import datasets
from clustercontrast.utils.data.preprocessor import Preprocessor
import time
import channel_lists

class Profiling(object):
    def __init__(self, panel):
        if not isinstance(panel, dict):
            raise ValueError("Panel must be a dictionary with channel names as keys.")
        self.panel = list(panel.keys())  # Ensure consistent channel order
        super(Profiling, self).__init__()

    def extract_coordinates(self, fid):
        """Extracts cx, cy coordinates from filename using regex"""
        cx, cy = fid.split('_c')[1].split('s')[0].split('_') 
        return cx, cy

    def process(self, data_loader, output_path,debug_mode=False):
        print('Starting feature extraction...')
        print('Start time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        # Prepare output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize list to collect features for DataFrame
        all_features = []
        
        # Process in batch mode
        with torch.no_grad():
            for batch_idx, (imgs, fnames, _, _, _) in enumerate(data_loader):
                imgs = imgs.float()
                
                # if debug_mode is True:
                #     print(imgs.shape)
                #     return None
                # Process each image in batch
                for j in range(imgs.size(0)):
                    
                    cx, cy = self.extract_coordinates(fnames[j])
                    cx, cy = int(cx), int(cy)  # Convert to integers
                    features = {'cx': cx, 'cy': cy}
                    
                    # Calculate per-channel statistics
                    for i, channel in enumerate(self.panel):
                        channel_img = imgs[j, :, :, i]
                        features[f'{channel}_mean'] = channel_img.mean().item()
                        features[f'{channel}_std'] = channel_img.std().item()
                    
                    all_features.append(features)
                
                if batch_idx % 10 == 0:
                    print(f'Processed batch {batch_idx} | Time: {time.strftime("%H:%M:%S")}')

        # Create DataFrame from collected features
        df = pd.DataFrame(all_features)
        
        # Define column order
        fieldnames = [ 'cx', 'cy']
        for channel in self.panel:
            fieldnames.append(f'{channel}_mean')
            fieldnames.append(f'{channel}_std')
        
        # Reorder DataFrame columns
        df = df[fieldnames]
        
        # Save DataFrame to CSV
        df.to_csv(output_path, index=False)
        
        print('Feature extraction completed at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        return None

def get_data(dataset_name, data_dir, batch_size, workers, name='brain'):
    if 'brain' in name:
        root = osp.join(data_dir, dataset_name)
        dataset = datasets.create(name, root)
    else:
        root = osp.join(data_dir, name)
        dataset = datasets.create(name, root)

    data_list = set(dataset.train)

    if hasattr(dataset, 'test'):
        print('add test data')
        data_list |= set(dataset.test)
    if hasattr(dataset, 'query'):
        print('add query data')
        data_list |= set(dataset.query)
    if hasattr(dataset, 'gallery'):
        print('add gallery data')
        data_list |= set(dataset.gallery)

    test_loader = DataLoader(
        Preprocessor(list(data_list), root=dataset.images_dir),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    
    print('size of the dataset: ', len(data_list))
    return dataset, test_loader

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    print("==========\nArgs:{}\n==========".format(args))
    _, test_loader = get_data(args.dataset, args.data_dir, args.batch_size, args.workers)
    panel = getattr(channel_lists, args.panel)
    profil = Profiling(panel)
    profil.process(test_loader, args.outputCSV)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--panel', type=str, default='resnet50')
    parser.add_argument('--outputCSV', type=str, default='./profiling.csv')
    parser.add_argument('--seed', type=int, default=1)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/lhuang37/repos/VisQ-Search-Engine/examples/data')
    main()