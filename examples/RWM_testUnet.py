from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import os
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
print('We are here: ', os.getcwd())
sys.path.append('/home/lhuang37/repos/VisQ-Search-Engine/')
sys.path.append('/home/lhuang37/repos/VisQ-Search-Engine/clusterconstrast')
from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.dsbn import convert_dsbn, convert_bn
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
#-------------------------------------------------------------------------------
import time
import random
import pickle #RWM
from tqdm import tqdm
from collections import OrderedDict


class RWM_Evaluator(object):

    def __init__(self, model):
        super(RWM_Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, output_dir, cmc_flag=False, rerank=False, tag='', sim_mat=False):
        print('Getting features...')
        print('starting time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.model = nn.DataParallel(self.model)
        self.model.eval()
        self.model.cuda()
        
        raw_features_path = os.path.join(output_dir, tag + 'raw_features.pkl')
        fids_path = os.path.join(output_dir, tag + 'fids.pkl')
        
        # Initialize containers for all features and filenames
        all_features = OrderedDict()
        all_fids = []

        with torch.no_grad():
            for imgs, fnames, pids, _, _ in data_loader:
                imgs = imgs.cuda().float()
                outputs, _ = self.model(imgs)
                outputs = outputs.detach().cpu().numpy()  # Convert to numpy array
                # print(outputs.type())
                
                # Update the global OrderedDict and filename list
                for fname, output in zip(fnames, outputs):
                    all_features[fname] = output
                all_fids.extend(fnames)  # Use extend() for batch filenames

        # Save the aggregated results once
        with open(raw_features_path, "wb") as f_features, open(fids_path, "wb") as f_fids:
            pickle.dump(all_features, f_features)
            pickle.dump(all_fids, f_fids)
        print('Finished getting features', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if sim_mat is False:
            return None
        
        return None
#-------------------------------------------------------------------------------

def get_data(name, dataset_name, data_dir, height, width, mean_array, std_array, batch_size, workers):
    if 'brain' in name:
        root = osp.join(data_dir, dataset_name) #RWM
        dataset = datasets.create(name, root)
    else:
        root = osp.join(data_dir, name)
        dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=mean_array, std=std_array)
    test_transformer= T.Compose([ T.ToTensor(),
                T.Resize((height, width), interpolation=3),
                normalizer
                ])

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
        Preprocessor(list(data_list),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    # print dataset size
    print('size of the dataset: ', len(data_list))
    

    return dataset, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test_attention.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    my_file = open(os.path.join(args.data_dir, args.datasetname, "mean_std.txt"), "r")
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

    # Create data loaders
    dataset, test_loader = get_data(args.dataset, args.datasetname, args.data_dir, args.height,
                                    args.width, mean_array, std_array, args.batch_size, args.workers)

    # Create model
    if args.arch =='unet': 
        model = models.create(args.arch, img_size=args.height, num_features=args.features, new_in_channels=args.num_channels, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type) 
    else: 
        model = models.create(args.arch, pretrained=False, num_features=args.features, new_in_channels = args.num_channels, dropout=args.dropout,
                            num_classes=0, pooling_type=args.pooling_type)

    if args.dsbn:
        print("==> Load the model with domain-specific BNs")
        convert_dsbn(model)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model, strip='module.')

    if args.dsbn:
        print("==> Test with {}-domain BNs".format("source" if args.test_source else "target"))
        convert_bn(model, use_target=(not args.test_source))

    
    evaluator = RWM_Evaluator(model)
    evaluator.evaluate(test_loader, dataset.train, dataset.train, log_dir, cmc_flag=True, 
                       rerank=args.rerank, tag=args.output_tag,sim_mat=args.sim_mat)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('-nc', '--num_channels', type=int, default=7)
    parser.add_argument('-dn', '--datasetname', type=str, default='brain1') #for the specific brain id

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)

    parser.add_argument('--resume', type=str,
                        default="/media/yixuan/DATA/cluster-contrast/market-res50/logs/model_best.pth.tar",
                        metavar='PATH')
    # testing configs
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--dsbn', action='store_true',
                        help="test on the model with domain-specific BN")
    parser.add_argument('--test-source', action='store_true',
                        help="test on the source domain")
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/project/xwu/lhuang37/roysam/repos/richal/VisQ-Search-Engine/examples/data')
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--embedding_features_path', type=str,
                        default='/media/yixuan/Project/guangyuan/workpalces/SpCL/embedding_features/mark1501_res50_ibn/')
    parser.add_argument('--output-tag', type = str, default = 'best')
    parser.add_argument('--sim_mat', action='store_true', help="If save the similarity matrix")
    
    main()
