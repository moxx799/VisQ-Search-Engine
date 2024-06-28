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
sys.path.append('/project/ece/roysam/lhuang37/cluster-contrast-reid/')
sys.path.append('/project/ece/roysam/lhuang37/cluster-contrast-reid/clusterconstrast')
# sys.path.append('/project/roysam/rwmills/repos/cluster-contrast-reid/clusterconstrast/RWM_evaluators')
from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.dsbn import convert_dsbn, convert_bn
# from clustercontrast.RWM_evaluators import RMW_Evaluator
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict

#-------------------------------------------------------------------------------

import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy
import pickle #RWM
import gc

from clustercontrast.evaluation_metrics import cmc, mean_ap
from clustercontrast.utils.meters import AverageMeter
from clustercontrast.utils.rerank import re_ranking
from clustercontrast.utils import to_torch

import h5py  #RWM

def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs, attention = model(inputs.float())
    outputs, attention= outputs.data.cpu(), attention.data.cpu() 
    return outputs, attention


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    weights = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs, attention= extract_cnn_feature(model, imgs)
            for fname, output, pid, wei in zip(fnames, outputs, pids, attention):
                features[fname] = output
                labels[fname] = pid
                weights[fname] = wei

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels, weights

def attention_weight_extraction(features, query=None, gallery=None):
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)

    # x = F.normalize(x, p=2, dim=1)
    # y = F.normalize(y, p=2, dim=1)

    return x.numpy(), y.numpy()

def pairwise_distance_cosine(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = F.normalize(x, p=2, dim=1)
        dist_m = 1 - torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)

    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)


    dist_m = 1 - torch.mm(x, y.t())


    return dist_m, x.numpy(), y.numpy()

def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False,
                 image_flag=False):
    ''' function that does all the evaluation RWM '''
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP

    if (not image_flag):
        '''
        RWM:
        lets figure out  how to return a similar gallery to image query
        '''
        query_ids



class RWM_Evaluator(object): #RWM change name
    def __init__(self, model):
        super(RWM_Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, output_dir, cmc_flag=False, rerank=False):
        print('Getting features...')
        features, _ ,weights= extract_features(self.model, data_loader)
        # f = open(os.path.join(output_dir, 'raw_features.pkl',"wb"))
        # write the python object (dict) to pickle file
        # pickle.dump(features,f)
        # f.close()
        # f = open(os.path.join(output_dir, 'pid_labels.pkl',"wb"))
        # write the python object (dict) to pickle file
        # pickle.dump(_,f)
        # f.close()
        print('Calculating distances... ')
        # distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        print('Extracting attention weights: ')
        query_weights, gallery_weights = attention_weight_extraction(weights, query, gallery)
        print('Saving euclidean distance file...')
        with h5py.File(os.path.join(output_dir, 'feature_results.h5'), 'w') as f:
            # f.create_dataset('orig_features', data=features)
            f.create_dataset('distmat', data=distmat)
            f.create_dataset('query_features', data=query_features)
            f.create_dataset('gallery_features', data=gallery_features)
            f.create_dataset('query_weights', data=query_weights)
            f.create_dataset('gallery_weights', data=gallery_weights)
        print('Writing results...')
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        del distmat, query_features, gallery_features 
        gc.collect() #free memory
        print('Cosine distances')

        distmat, query_features, gallery_features = pairwise_distance_cosine(features, query, gallery)
        #RWM - save distmat, query features and gallery features for post processing analysis
        print('Saving cosine distnace file...')
        with h5py.File(os.path.join(output_dir, 'feature_results_Cosine.h5'), 'w') as f:
            # f.create_dataset('orig_features', data=features)
            f.create_dataset('distmat', data=distmat)
            f.create_dataset('query_features', data=query_features)
            f.create_dataset('gallery_features', data=gallery_features)

        # print('Applying person re-ranking ...')
        # distmat_qq, _, _ = pairwise_distance(features, query, query)
        # distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        # distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        # print('Saving reranking file...')
        # np.save(os.path.join(output_dir, 'rerankingDist.npy', distmat))

        

        # if (not rerank):
            # return results
        return results # evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
    def prep_rank(self, data_loader, query, gallery):
        print('Getting features...')
        features, _ = extract_features(self.model, data_loader)
        print('Calculating distances... ')
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        return features, distmat

    def evaluate_rerank(self, features, distmat, query, gallery, output_dir, cmc_flag=False, rerank=False):
        print('Getting reranking features and distance matrix...')
        # features, _ = extract_features(self.model, data_loader)
        # features, distmat = self.prep_rank(data_loader, query, gallery)
        print('Applying person re-ranking ...')
        print('For q')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        print('For g')
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        print('reranking')
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        print('Saving reranking file...')
        np.save(os.path.join(output_dir, 'rerankingDist.npy'), distmat )

        print('Writing results...')
        results = evaluate_all(distmat_qq, distmat_gg, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
        return results

#-------------------------------------------------------------------------------

def get_data(name, dataset_name, data_dir, height, width, mean_array, std_array, batch_size, workers):
    if 'brain' in name:
        root = osp.join(data_dir, dataset_name) #RWM
        dataset = datasets.create(name, root)
    else:
        root = osp.join(data_dir, name)
        dataset = datasets.create(name, root)

    # normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalizer_7ch = T.Normalize(mean=[0.1012, 0.1008, 0.0997, 0.0637, 0.0404, 0.0986, 0.0982],
                                #  std=[0.2360, 0.2351, 0.2339, 0.1883, 0.1514, 0.2325, 0.2315])
    normalizer = T.Normalize(mean=mean_array, std=std_array)
    # test_transformer = T.Compose([
    #          T.Resize((height, width), interpolation=3),
    #          T.ToTensor(),
    #          normalizer
    #      ])

    #maybe for 7 channel:
    test_transformer= T.Compose([ T.ToTensor(),
                T.Resize((height, width), interpolation=3),
                normalizer
                ])
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
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

    model.cuda()
    model = nn.DataParallel(model)

    # Evaluator
    model.eval()
    evaluator = RWM_Evaluator(model)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, log_dir, cmc_flag=True, rerank=args.rerank)
    feats, dist = evaluator.prep_rank(test_loader, dataset.query, dataset.gallery)
    evaluator.evaluate_rerank(feats, dist, dataset.query, dataset.gallery, log_dir, cmc_flag=True, rerank=args.rerank)
    # evaluator.evaluate(test_loader, dataset.query, dataset.gallery, log_dir, cmc_flag=True, rerank=args.rerank)
    return


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
                        default='/project/ece/roysam/lhuang37/cluster-contrast-reid/examples/data')
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--embedding_features_path', type=str,
                        default='/media/yixuan/Project/guangyuan/workpalces/SpCL/embedding_features/mark1501_res50_ibn/')
    # parser.add_argument('--output_dir', type = str, default = '/project/roysam/rwmills/repos/cluster-contrast-reid/results/')
    main()
