from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
from clustercontrast.utils.loss import *

class TLTrainer(object):
    def __init__(self, encoder, tri_loss=None, memory=None):
        super(TLTrainer, self).__init__()
        self.encoder = encoder
        self.tri_loss = tri_loss
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            if inputs != None: 
                # process inputs
                inputs, labels, indexes = self._parse_data(inputs)
    
                # forward
                f_out,_ = self._forward(inputs.float()) #RWM convert to float 
                # print("f_out shape: {}".format(f_out.shape))
                # compute loss with the hybrid memory
                # loss = self.memory(f_out, indexes)
                loss1 = self.memory(f_out, labels)
                loss2, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(self.tri_loss, f_out, labels,normalize_feature=False)
                #loss, _ = self.tri_loss(f_out, labels)
                loss = loss1 + loss2 # +loss1 #RWM combined loss 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                losses.update(loss.item())
    
                # print log
                batch_time.update(time.time() - end)
                end = time.time()
    
                if (i + 1) % print_freq == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          'Loss {:.3f} ({:.3f})'
                          .format(epoch, i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

