# -*- coding:UTF-8 -*-
from __future__ import print_function
import argparse
import os
import torch
# from model import Model
import model_newest
import multiprocessing as mp
import wsad_dataset
import pdb
import random
# from video_dataset import Dataset,ANTDataset
from test import test
from train import train
from utils.wsad_utils import get_logger,get_timestamp
# from utils import Recorder as Logger
from torch.utils.tensorboard import SummaryWriter

import options
import numpy as np
from torch.optim import lr_scheduler
from tqdm import tqdm
import shutil

# ####### NNI ###########
# import nni


torch.set_default_tensor_type('torch.cuda.FloatTensor')
def setup_seed(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

import torch.optim as optim

if __name__ == '__main__':
   pool = mp.Pool(5)

   # import pdb
   # pdb.set_trace()

   args = options.parser.parse_args()

   seed=args.seed
   print('=============seed: {}, pid: {}============='.format(seed,os.getpid()))
   setup_seed(seed)

   device = torch.device("cuda")

   if not os.path.exists('./ckpt/'):
      os.makedirs('./ckpt/')
   # contruct logger
   if not os.path.exists('./logs/'):
      os.makedirs('./logs/')
   if os.path.exists('./logs/' + args.model_name+'.log'):
      os.remove('./logs/' + args.model_name+'.log')
   logger=get_logger('./logs/' + args.model_name +'.log')
   if not os.path.exists('./summary/'+args.model_name):
      os.makedirs('./summary/'+args.model_name)
   if os.path.exists('./summary/'+args.model_name): 
      shutil.rmtree('./summary/'+args.model_name)
   if os.path.exists('./results/'+ args.dataset_name +"_"+args.model_name + "_results.csv"): 
      os.remove('./results/'+ args.dataset_name +"_"+args.model_name + "_results.csv")
   writer=SummaryWriter('./summary/'+args.model_name)
   logger.info(args)

   # load dataset
   dataset = getattr(wsad_dataset,args.dataset)(args)

   # load model
   model = getattr(model_newest,args.use_model)(dataset.feature_size, dataset.num_class,opt=args).to(device)
   logger.info(model)

   if args.pretrained_ckpt is not None:
      model.load_state_dict(torch.load(args.pretrained_ckpt))

   optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

   total_loss = 0
   if 'Thumos' in args.dataset_name or 'BEOID' in args.dataset_name or 'GTEA' in args.dataset_name:
      max_map=[0]*9
   else:
      max_map=[0]*10

   for itr in tqdm(range(0 if args.pretrained_ckpt==None else args.pseudo_iter,args.max_iter)): 
      loss = train(itr, dataset, args, model, optimizer, logger, device, writer)
      total_loss+=loss
      if itr % args.interval == 0 and not itr == 0:
         logger.info('Iteration: %d, Loss: %.5f' %(itr, total_loss/args.interval))
         total_loss = 0
         torch.save(model.state_dict(), './ckpt/last_' + args.model_name + '.pkl')

         iou,dmap,pl_precision = test(itr, dataset, args, model, logger, device, writer, pool)

         if 'Thumos' in args.dataset_name or 'BEOID' in args.dataset_name or 'GTEA' in args.dataset_name:
            cond=sum(dmap)>sum(max_map) 
         else:
            cond=np.mean(dmap)>np.mean(max_map)
         if cond:
            torch.save(model.state_dict(), './ckpt/best_' + args.model_name + '.pkl') 
            max_map = dmap

         logger.info('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i],max_map[i]*100) for i in range(len(iou))]))
         max_map = np.array(max_map)
         logger.info('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(max_map[:5])*100,np.mean(max_map[:7])*100,np.mean(max_map)*100))
         logger.info("------------------pid: {}--------------------".format(os.getpid()))
