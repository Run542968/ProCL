import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from torch.autograd import Variable
import time
import pdb

from losses import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import losses


def train(itr, dataset, args, model, optimizer, logger, device, writer):
    model.train()

    if args.use_ms:    
        ms_features, ms_labels, rand_sampleid, idx = dataset.load_data(is_single=False) # ms_features:[ms_num,B,T_i,d]
        if (itr > args.pseudo_iter) or (args.pretrained_ckpt != None): # only one scale data is input into model
            ms_outputs = dict()
            for i,ms_feat in enumerate(ms_features):
                features = torch.from_numpy(ms_feat).float().to(device) #[B,T_i,d]
                labels = torch.from_numpy(ms_labels).float().to(device)

                outputs = model(features,opt=args)
                ms_outputs[i] = outputs # save the output of multi-scale logits

        else:# pretrain stage
            ms_outputs = dict()
            for i,ms_feat in enumerate(ms_features):
                if i==0:
                    features = torch.from_numpy(ms_feat).float().to(device) #[B,T_i,d]
                    labels = torch.from_numpy(ms_labels).float().to(device)

                    outputs = model(features,opt=args)
                    ms_outputs[i] = outputs # only save the 0-th scale output
                else:
                    continue

        total_loss,loss_dict,pseudo_label_back,uncertainty_index = model.ms_criterion(ms_outputs, 
                                                                                    labels,
                                                                                    device=device,
                                                                                    logger=logger,
                                                                                    opt=args,
                                                                                    itr=itr)

        if args.show_log:
            logger.info(loss_dict)

        for key in loss_dict.keys():
            writer.add_scalar(key,loss_dict[key],itr)

        optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        return total_loss.data.cpu().numpy()

    else: # only use single-scale data for pcl

        features, labels, rand_sampleid, idx = dataset.load_data(is_single=True) # ms_features:[ms_num,B,T_i,d]

        features = torch.from_numpy(features).float().to(device) #[B,T_i,d]
        labels = torch.from_numpy(labels).float().to(device)

        outputs = model(features,opt=args)

        total_loss,loss_dict,pseudo_label_back,uncertainty_index = model.criterion(outputs, 
                                                                                    labels,
                                                                                    device=device,
                                                                                    logger=logger,
                                                                                    opt=args,
                                                                                    itr=itr)

        if args.show_log:
            logger.info(loss_dict)

        for key in loss_dict.keys():
            writer.add_scalar(key,loss_dict[key],itr)

        optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        return total_loss.data.cpu().numpy()

