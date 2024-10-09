# -*- coding: UTF-8 -*-

import copy
from textwrap import indent
from tkinter.messagebox import NO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import losses
import utils.wsad_utils as utils
from torch.nn import init
import time

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # torch_init.xavier_uniform_(m.weight)
        # import pdb
        # pdb.set_trace()
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

class Model_Thumos(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim=2048
        dropout_ratio=args['opt'].dropout_ratio
        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2) )
        self.att_rgb_encoder = nn.Sequential(
            nn.Conv1d(n_feature//2, embed_dim//2, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.att_flow_encoder = nn.Sequential(
            nn.Conv1d(n_feature//2, embed_dim//2, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(nn.Dropout(dropout_ratio),nn.Conv1d(embed_dim, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            nn.Conv1d(embed_dim, n_class+1, 1,bias=True))

        self.attention_rgb = nn.Sequential(nn.Conv1d(embed_dim//2, 1, 1))
        self.attention_flow = nn.Sequential(nn.Conv1d(embed_dim//2, 1, 1))

        self.apply(weights_init)

    def forward(self, inputs, **args):
        x = inputs.transpose(-1, -2)
        feat = self.feat_encoder(x)
        x_cls = self.classifier(feat)

        x_rgb_atn_logit = self.attention_rgb(self.att_rgb_encoder(x[:,:1024]))
        x_flow_atn_logit = self.attention_flow(self.att_flow_encoder(x[:,1024:]))

        x_rgb_atn = torch.sigmoid(x_rgb_atn_logit)
        x_flow_atn = torch.sigmoid(x_flow_atn_logit)

        atn_fusion_weight = args['opt'].delta
        x_atn = atn_fusion_weight*x_rgb_atn + (1-atn_fusion_weight)*x_flow_atn

        return feat.transpose(-1, -2), x_cls.transpose(-1, -2), x_atn.transpose(-1, -2), x_rgb_atn_logit.transpose(-1, -2), x_flow_atn_logit.transpose(-1, -2),x_rgb_atn.transpose(-1,-2),x_flow_atn.transpose(-1,-2)

    def criterion(self, outputs, labels, **args):

        feat, element_logits, x_atn, x_rgb_atn_logit, x_flow_atn_logit,x_rgb_atn,x_flow_atn = outputs #[B,T,2048] [B,T,C+1] [B,T,1]
        b,n,c = element_logits.shape

        element_logits_flow_supp = self._multiply(element_logits, x_flow_atn,include_min=True)
        element_logits_rgb_supp = self._multiply(element_logits, x_rgb_atn,include_min=True)

        # CLS
        loss_cls_orig, _ = self.topkloss(element_logits,
                                    labels,
                                    is_back=True,
                                    rat=args['opt'].k,
                                    reduce=None)
        # CLS_sup_flow
        loss_cls_flow_supp, _ = self.topkloss(element_logits_flow_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)
        # CLS_sup_rgb
        loss_cls_rgb_supp, _ = self.topkloss(element_logits_rgb_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)

        # CLL
        if args['opt'].lambda_cll !=0 :
            loss_3_orig_negative = self.negative_loss(element_logits,labels,is_back=True)
        else:
            loss_3_orig_negative = torch.tensor([0]).detach()

        # PL
        if args['opt'].lambda_pl !=0 and (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None):
            pseudo_label,_ = self.Pseudo_Label_Generation(element_logits,x_atn,labels,args['opt'],args['device'],is_back=True)
            loss_pl = self.Pseudo_Label_Loss(element_logits,pseudo_label,x_rgb_atn,x_flow_atn,args['opt'])
        else:
            loss_pl = torch.tensor([0]).detach()


        # PCL & PLG
        if args['opt'].lambda_pcl !=0 and (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None):
            pseudo_complementary_label,uncertainty_index = self.Pseudo_Complementary_Label_Generation(element_logits,x_atn,labels,args['opt'],args['device'],is_back=True)
            loss_pcl = self.Pseudo_Complementary_Loss(element_logits,pseudo_complementary_label,x_rgb_atn,x_flow_atn,args['opt'])
        else:
            loss_pcl = torch.tensor([0]).detach()
            pseudo_complementary_label = None
            uncertainty_index = None

        # SCL loss
        if args['opt'].lambda_scl !=0 : 
            if args['opt'].SCL_method == 'no_pcl':
                loss_scl = self.SCL_Loss(element_logits,uncertainty_index,x_atn,x_rgb_atn,x_flow_atn,args['opt'])
            elif (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None): 
                loss_scl = self.SCL_Loss_uncertainty(element_logits,uncertainty_index,x_atn,x_rgb_atn,x_flow_atn,args['opt'])
            else: 
                loss_scl = torch.tensor([0]).detach()
        else:
            loss_scl = torch.tensor([0]).detach()

        total_loss = args['opt'].lambda_cls * (loss_cls_orig.mean()+loss_cls_flow_supp.mean()+loss_cls_rgb_supp.mean()) +\
                        args['opt'].lambda_cll * loss_3_orig_negative +\
                        args['opt'].lambda_pcl * loss_pcl +\
                        args['opt'].lambda_scl * loss_scl +\
                        args['opt'].lambda_pl * loss_pl

        loss_dict={
            'loss':(total_loss).item(),
            'loss_Cls':(loss_cls_orig.mean()+loss_cls_flow_supp.mean()+loss_cls_rgb_supp.mean()).item(),
            'loss_CLL':loss_3_orig_negative.item(),
            'loss_PCL':loss_pcl.item(),
            'loss_SCL':loss_scl.item(),
            'loss_PL':loss_pl.item()
        }

        return total_loss,loss_dict,pseudo_complementary_label,uncertainty_index

    def ms_criterion(self,ms_outputs,labels,**args):

        feat, element_logits_target, x_atn, x_rgb_atn_logit, x_flow_atn_logit,x_rgb_atn,x_flow_atn = ms_outputs[0] #[B,T,2048] [B,T,C+1] [B,T,1]

        element_logits_flow_supp = self._multiply(element_logits_target, x_flow_atn,include_min=True)
        element_logits_rgb_supp = self._multiply(element_logits_target, x_rgb_atn,include_min=True)

        # CLS
        loss_cls_orig, _ = self.topkloss(element_logits_target,
                                    labels,
                                    is_back=True,
                                    rat=args['opt'].k,
                                    reduce=None)
        # CLS_sup_flow
        loss_cls_flow_supp, _ = self.topkloss(element_logits_flow_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)
        # CLS_sup_rgb
        loss_cls_rgb_supp, _ = self.topkloss(element_logits_rgb_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)

        # CLL
        if args['opt'].lambda_cll !=0 :
            loss_3_orig_negative = self.negative_loss(element_logits_target,labels,is_back=True)
        else:
            loss_3_orig_negative = torch.tensor([0]).detach()

        # PL
        if args['opt'].lambda_pl !=0 and (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None):
            pseudo_label,_ = self.Pseudo_Label_Generation(element_logits_target,x_atn,labels,args['opt'],args['device'],is_back=True)
            loss_pl = self.Pseudo_Label_Loss(element_logits_target,pseudo_label,x_rgb_atn,x_flow_atn,args['opt'])
        else:
            loss_pl = torch.tensor([0]).detach()


        # Multi_Scale_Label_Propagation
        ms_pcl = None # placeholder
        if args['opt'].lambda_lpl != 0 and (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None):
            lpl_loss,pseudo_complementary_label,uncertainty_index_median = self.Label_Propagation_Loss(ms_outputs,ms_pcl,labels,args['device'],args['opt'])
        else:
            lpl_loss = torch.tensor([0]).detach()
            pseudo_complementary_label = None
            uncertainty_index_median = None

        # Foreground_Background_Consistency for median scale
        if args['opt'].lambda_mscl !=0 : # whether to use Foreground_Background_Discrimination loss
            if args['opt'].SCL_method == 'no_pcl': # use it when the pcl is not adopted, so can't use the uncertainty index to distinguish ambiguity snippet, only used for the experiment: MIL+CL+FBD (without PCL)
                loss_mscl = self.SCL_Loss(element_logits_target,uncertainty_index_median,x_atn,x_rgb_atn,x_flow_atn,args['opt'])
            elif (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None): # using uncertainty index
                loss_mscl = self.SCL_Loss_uncertainty(element_logits_target,uncertainty_index_median,x_atn,x_rgb_atn,x_flow_atn,args['opt'])
            else: 
                loss_mscl = torch.tensor([0]).detach()
        else:
            loss_mscl = torch.tensor([0]).detach()


        total_loss = args['opt'].lambda_cls * (loss_cls_orig.mean()+loss_cls_flow_supp.mean()+loss_cls_rgb_supp.mean()) +\
                        args['opt'].lambda_cll * loss_3_orig_negative +\
                        args['opt'].lambda_lpl * lpl_loss +\
                        args['opt'].lambda_mscl * loss_mscl

        loss_dict={
            'loss':(total_loss).item(),
            'loss_Cls':(loss_cls_orig.mean()+loss_cls_flow_supp.mean()+loss_cls_rgb_supp.mean()).item(),
            'loss_CLL':loss_3_orig_negative.item(),
            'loss_LPL':lpl_loss.item(),
            'loss_mSCL':loss_mscl.item()
        }

        return total_loss, loss_dict,pseudo_complementary_label,uncertainty_index_median

    def _multiply(self, x, atn, dim1=-1, dim2=-2, include_min=False):
        '''
        class_min
        '''
        if include_min:
            _min = x.min(dim=dim1, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def _KL_divergence(self,A_prob, B_prob):
        return A_prob * (torch.log(A_prob) - torch.log(B_prob))

    def _symKL(self,A_prob, B_prob):
        return 0.5*A_prob * (torch.log(A_prob) - torch.log(B_prob))+0.5*B_prob*(torch.log(B_prob)-torch.log(A_prob))
     
    def _sKL(self,A_prob, B_prob):
        return A_prob * (torch.log(A_prob) - torch.log(B_prob))

    def _dKL(self,A_prob, B_prob):
        return A_prob.detach() * (torch.log(A_prob.detach()) - torch.log(B_prob))

    def _JS(self,A_prob, B_prob):
        M_prob = (A_prob+B_prob)/2
        return 0.5*A_prob * (torch.log(A_prob) - torch.log(M_prob))+0.5*B_prob*(torch.log(B_prob)-torch.log(M_prob))

    def _Tanh(self,unconsistency,slope,coefficient=1):
        weight = (2/(1+torch.exp(-2*slope*unconsistency)))-1
        return weight

    def _binary_cross_entropy(self,pred, target):
        return -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)

    def _MSE(self,pred,target):
        return ((pred-target)**2).mean()

    def _MAE(self,pred,target):
        return (pred-target).abs().mean()

    def negative_loss(self,element_logits,labels,is_back=False,focal=False):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        b,n,c = element_logits.shape

        element_prob = F.softmax(element_logits,-1)

        labels_with_back = labels_with_back.unsqueeze(1).expand(b,n,c)

        s_loss = -((1-labels_with_back)*torch.log(1 - element_prob+1e-8)).sum(-1)

        return s_loss.mean()

    def topkloss(self,element_logits,labels,is_back=True,lab_rand=None,rat=8,reduce=None,focal=False):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)


        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )

        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)

        milloss = (-(labels_with_back *
                         F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))

        return milloss, topk_ind

    def Pseudo_Label_Generation(self,element_logits,element_atn,labels,args,device,is_back=True):

        b,n,c = element_logits.shape
        pseudo_label_back = np.zeros((b,n,c)) #[B,T,C+1]

        if is_back:
            pseudo_label_back[:,:,-1] = 1 #[B,T,C+1]
        else: 
            pass

        labels = labels.cpu().numpy() 
        element_logits = element_logits.detach().cpu().numpy()
        element_atn = element_atn.detach().cpu().numpy()

        # att_thresh_list = np.arange(0.4, 0.925, 0.025)# Follow ASM_loc
        att_thresh_list = np.arange(args.PLG_act_thres[0],args.PLG_act_thres[1],args.PLG_act_thres[2])

        batch_proposal_list = []
        for v in range(b): 
            v_logits = element_logits[v].copy() #[T,C+1]
            v_atn = element_atn[v].copy() #[T,1]
            v_atn_logits = v_logits*v_atn #[T,C+1]
            v_gt_index = np.where(labels[v]>0)[0]
            v_gt_atn_logits = v_atn_logits[:,v_gt_index]  #[T,gt]

            # generate proposals
            proposal_dict = {}
            for att_thresh in att_thresh_list:
                seg_list = []

                for c in v_gt_index:
                    if args.PLG_proposal_mode == 'atn':
                        pos = np.where(v_atn[:,0]>att_thresh)[0]
                    elif args.PLG_proposal_mode == 'logits':
                        pos = np.where(v_logits[:,c]>att_thresh)[0]
                    elif args.PLG_proposal_mode == 'atn_logits':
                        pos = np.where(v_atn_logits[:,c]>att_thresh)[0]
                    else:
                        raise ValueError("Don't define this proposal_mode. ")
                    seg_list.append(pos)

                proposals = utils.get_pseudo_proposal_oic(seg_list,
                                    v_gt_atn_logits,
                                    v_gt_index
                                    )

                for j in range(len(proposals)):
                    try:
                        class_id = proposals[j][0][0]

                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []

                        proposal_dict[class_id] += proposals[j]
                    except IndexError:
                        raise IndexError(f"Index error")

            # nms and update pseudo label for current video
            pseudo_label_back,final_proposal=utils.nms_wiht_pseudo_label(proposal_dict,
                                                            pseudo_label_back,
                                                            v,
                                                            iou_thr=0.0
                                                            )

            batch_proposal_list.append(final_proposal)

        return torch.from_numpy(pseudo_label_back).float().to(device),batch_proposal_list

    def Pseudo_Complementary_Label_Generation(self,element_logits,element_atn,labels,args,device,is_back=True):
        b,n,c = element_logits.shape
        bg_id = c-1 # the index of background category
        pseudo_label_back = torch.ones((b,n,c)) #[B,T,C+1]
        uncertainty_index = torch.zeros((b,n,1)) #[B,T,1]

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else: 
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        labels = labels_with_back #  [B,C+1]
        element_logits = element_logits.detach()

        for v in range(b): 
            v_logits = element_logits[v] # [T,C+1]

            if args.PLG_logits_mode == 'norm':
                norm_v_logits = (v_logits-torch.min(v_logits))/(torch.max(v_logits)-torch.min(v_logits)) # min-max normalization
            elif args.PLG_logits_mode == 'none':
                norm_v_logits = v_logits
            else:
                raise ValueError("Don't define this PLG_logits_mode. ")

            v_gt_index = torch.where(labels[v]>0)[0] 
            v_gt_logits = norm_v_logits[:,v_gt_index] # [T,c]
            # print("v_gt_logits.shape:",v_gt_logits.shape)
            v_softmax_gt_logits = torch.softmax(v_gt_logits,dim=-1) # [T,c]
            # print("v_softmax_gt_logits.shape:",v_softmax_gt_logits.shape)
            v_softmax_gt_mean = torch.mean(v_softmax_gt_logits,dim=-1) # [T]
            # print("v_gt_mean.shape:",v_gt_mean.shape)
            v_fg_info = torch.sum(v_softmax_gt_logits[:,:-1],dim=-1) # [T]
            # print("v_fg_info.shape:",v_fg_info.shape)
            v_bg_info = v_softmax_gt_logits[:,-1] #[T]
            # print("v_bg_info.shape:",v_bg_info.shape)
            v_info_entropy = -v_fg_info*torch.log(v_fg_info)-v_bg_info*torch.log(v_bg_info) #[T]
            # print("v_info_entropy.shape:",v_info_entropy.shape)
            for t in range(n):
                if v_info_entropy[t] >= args.PLG_thres: # uncertainty snippet
                    uncertainty_index[v,t,:] = 1
                else:
                    if len(v_gt_index)>2: # the number of gt category at least three
                        indices = torch.where(v_softmax_gt_logits[t] < v_softmax_gt_mean[t])[0] 
                        class_indices = v_gt_index[indices] # the categories that are excluded

                        max_indice = torch.argmax(v_softmax_gt_logits[t])
                        class_max_indice = v_gt_index[max_indice] 

                        if (bg_id not in class_indices) and (class_max_indice==bg_id):
                            pseudo_label_back[v,t,v_gt_index] = 0 
                            pseudo_label_back[v,t,bg_id] = 1
                        elif (bg_id not in class_indices) and (class_max_indice!=bg_id):
                            pseudo_label_back[v,t,class_indices] = 0
                            pseudo_label_back[v,t,bg_id] = 0 
                    else:
                        indices = torch.where(v_softmax_gt_logits[t] < v_softmax_gt_mean[t])[0] 
                        class_indices = v_gt_index[indices]
                        pseudo_label_back[v,t,class_indices] = 0

        return pseudo_label_back.long().to(device),uncertainty_index.long().to(device)

    def Pseudo_Complementary_Loss(self,element_logits,pseudo_label_back,rgb_atn,flow_atn,args,std_weight=None):
        b,n,c = element_logits.shape #[B,T,C+1]
        pseudo_label = pseudo_label_back.clone() #[B,T,C+1]
        element_prob = F.softmax(element_logits,-1) #[B,T,C+1]
        
        pcl_loss = -((1-pseudo_label)*torch.log(1-element_prob + 1e-8)).sum(-1)

        return pcl_loss.mean()

    def Pseudo_Label_Loss(self,element_logits,pseudo_label_back,rgb_atn,flow_atn,args):
        b,n,c = element_logits.shape #[B,T,C+1]
        pseudo_label = pseudo_label_back #[B,T,C+1]
        element_prob = F.softmax(element_logits,-1) #[B,T,C+1]

        pll_loss = -(pseudo_label*torch.log(element_prob + 1e-8)).mean()

        return pll_loss.mean()

    def SCL_Loss(self,element_logits,uncertainty_index,x_atn,rgb_atn,flow_atn,args):
        bg_atn = 1-x_atn #  [B,T,1]
        element_prob = torch.softmax(element_logits, dim=-1)[..., [-1]] # [B,T,1]

        kl_att_cls = self._KL_divergence(bg_atn+1e-8, element_prob+1e-8)
        kl_cls_att = self._KL_divergence(element_prob+1e-8, bg_atn+1e-8)
 
        if args.SCL_align=='BCE':
            s_loss = args.factor *(self._binary_cross_entropy(bg_atn, element_prob.detach())* \
                            (torch.exp(-args.SCL_alpha * kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*kl_att_cls))).detach()).sum()
            s_loss += (1 - args.factor) * (self._binary_cross_entropy(element_prob, bg_atn.detach())* \
                                    (torch.exp(-args.SCL_alpha * kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        elif args.SCL_align=='MSE':
            s_loss = args.factor *(self._MSE(bg_atn, element_prob.detach())* \
                            (torch.exp(-args.SCL_alpha * kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*kl_att_cls))).detach()).sum()
            s_loss += (1 - args.factor) * (self._MSE(element_prob, bg_atn.detach())* \
                                    (torch.exp(-args.SCL_alpha * kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        elif args.SCL_align=='symKL':
            s_loss = args.factor *(self._symKL(bg_atn, element_prob.detach())* \
                            (torch.exp(-args.SCL_alpha * kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*kl_att_cls))).detach()).sum()
            s_loss += (1 - args.factor) * (self._symKL(element_prob, bg_atn.detach())* \
                                    (torch.exp(-args.SCL_alpha * kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        elif args.SCL_align=='MAE':
            s_loss = args.factor *(self._MAE(bg_atn, element_prob.detach())* \
                            (torch.exp(-args.SCL_alpha * kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*kl_att_cls))).detach()).sum()
            s_loss += (1 - args.factor) * (self._MAE(element_prob, bg_atn.detach())* \
                                    (torch.exp(-args.SCL_alpha * kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        elif args.SCL_align=='Tanh':
            s_loss = args.factor *(self._binary_cross_entropy(bg_atn, element_prob.detach())* \
                            (self._Tanh(kl_att_cls,args.SCL_alpha)/torch.sum(self._Tanh(kl_att_cls,args.SCL_alpha))).detach()).sum()
            s_loss += (1 - args.factor) * (self._binary_cross_entropy(element_prob, bg_atn.detach())* \
                                    (self._Tanh(kl_cls_att,args.SCL_alpha)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        else:
            raise ValueError("Don't define this SCL_align mode.")

        return s_loss

    def SCL_Loss_uncertainty(self,element_logits,uncertainty_index,x_atn,rgb_atn,flow_atn,args):
        bg_atn = 1-x_atn
        b,n,c = element_logits.shape
        element_prob = torch.softmax(element_logits, dim=-1)[..., [-1]] # [B,T,1]

        kl_att_cls = self._KL_divergence(bg_atn+1e-8, element_prob+1e-8) # [B,T,1]
        kl_cls_att = self._KL_divergence(element_prob+1e-8, bg_atn+1e-8) # [B,T,1]

        for v in range(b):
            v_bg_atn = bg_atn[v] #[B,T,1]
            v_bg_logits = element_prob[v] #[T,1]
            v_ut_index = torch.where(uncertainty_index[v] > 0)[0] #[un_index_num]
            v_kl_att_cls = kl_att_cls[v]
            v_kl_cls_att = kl_cls_att[v]

            v_ut_bg_logits = v_bg_logits[v_ut_index,:] #[uncertainty_index,1]
            v_ut_bg_atn = v_bg_atn[v_ut_index,:] #[uncertainty_index,1]
            v_ut_kl_att_cls = v_kl_att_cls[v_ut_index,:]
            v_ut_kl_cls_att = v_kl_cls_att[v_ut_index,:]
            
            # for element_prob
            if args.SCL_align=='BCE':
                s_loss = args.factor *(self._binary_cross_entropy(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._binary_cross_entropy(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='MSE':
                s_loss = args.factor *(self._MSE(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._MSE(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='symKL':
                s_loss = args.factor *(self._symKL(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._symKL(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='MAE':
                s_loss = args.factor *(self._MAE(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._MAE(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='sKL': 
                s_loss = (self._sKL(v_ut_bg_atn, v_ut_bg_logits)* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
            elif args.SCL_align=='dKL': 
                s_loss = args.factor *(self._dKL(v_ut_bg_atn, v_ut_bg_logits)* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
            elif args.SCL_align=='JS': 
                s_loss = args.factor *(self._JS(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._JS(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='Tanh':
                s_loss = args.factor *(self._binary_cross_entropy(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (self._Tanh(v_ut_kl_att_cls,args.SCL_alpha)/torch.sum(self._Tanh(v_ut_kl_att_cls,args.SCL_alpha))).detach()).sum()
                s_loss += (1 - args.factor) * (self._binary_cross_entropy(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (self._Tanh(v_ut_kl_att_cls,args.SCL_alpha)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='Dexp':
                s_loss = args.factor *(self._binary_cross_entropy(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                ((torch.exp(args.SCL_alpha * v_ut_kl_att_cls)-1)/torch.sum(torch.exp(args.SCL_alpha*v_ut_kl_att_cls)-1)).detach()).sum()
                s_loss += (1 - args.factor) * (self._binary_cross_entropy(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        ((torch.exp(args.SCL_alpha * v_ut_kl_cls_att)-1)/torch.sum(torch.exp(args.SCL_alpha*v_ut_kl_cls_att)-1)).detach()).sum()
            else:
                raise ValueError("Don't define this SCL_align mode.")
        s_loss = s_loss / b

        return s_loss
    
    def Label_Propagation_Loss(self,ms_outputs:dict,ms_pcl:dict,labels,device,args):
        _, element_logits_target, x_atn_target, _, _,x_rgb_atn_target,x_flow_atn_target = ms_outputs[0] #[B,T,2048] [B,T,C+1] [B,T,1]
        b,t,c = element_logits_target.shape
        logits_stack = [element_logits_target]
        for i in range(1,len(ms_outputs)):
            if args.rescale_mode=='linear':
                logits = F.interpolate(ms_outputs[i][1].transpose(-1,-2),size=t,mode="linear").transpose(-1,-2)
            elif args.rescale_mode=='nearest':
                logits = F.interpolate(ms_outputs[i][1].transpose(-1,-2),size=t,mode="nearest").transpose(-1,-2)
            elif args.rescale_mode=='area': 
                logits = F.interpolate(ms_outputs[i][1].transpose(-1,-2),size=t,mode="area").transpose(-1,-2)
            elif args.rescale_mode=='nearest-exact':
                logits = F.interpolate(ms_outputs[i][1].transpose(-1,-2),size=t,mode="nearest-exact").transpose(-1,-2)
            else:
                raise ValueError("Don't define this rescale model. ")
            
            logits_stack.append(logits)
        std_logits = torch.std(torch.stack(logits_stack,dim=0),dim=0) # [N,B,T,C+1]->[B,T,C+1]

        weighted_norm_logits_stack=[]
        for logits,weight in zip(logits_stack,args.ensemble_weight):
            if args.lpl_norm=='min_max':
                # global min_max normalization, then class_softmax
                _min = torch.min(torch.min(logits,-1,keepdim=True)[0],-2,keepdim=True)[0] #[B,1,1]
                _max = torch.max(torch.max(logits,-1,keepdim=True)[0],-2,keepdim=True)[0] #[B,1,1]
                norm_logits = (logits-_min)/(_max-_min)  #[B,T,C+1]
            elif args.lpl_norm == 'positive':
                _min = torch.min(torch.min(logits,-1,keepdim=True)[0],-2,keepdim=True)[0] #[B,1,1]
                norm_logits = logits-_min
            elif args.lpl_norm == 'none':
                norm_logits = logits
            else:
                raise ValueError("Don't define this lpl_norm model. ")
            weighted_norm_logits = norm_logits*weight
            weighted_norm_logits_stack.append(weighted_norm_logits)

        mean_logits_stack = torch.stack(weighted_norm_logits_stack,dim=0) # [N,B,T,C+1]
        mean_logits = torch.sum(mean_logits_stack,dim=0) #[B,T,C+1]
        std_logits = std_logits.detach()

        pseudo_label_back,uncertainty_index = self.Pseudo_Complementary_Label_Generation(mean_logits,None,labels,args,device,is_back=True)

        if args.multi_back: # multi-scale back-propagation
            element_prob = F.softmax(mean_logits,-1) #[B,T,C+1]
            lpl_loss = -(torch.exp(-args.alpha*std_logits)*(1-pseudo_label_back.detach())*torch.log(1-element_prob + 1e-8)).sum(-1)
        else: # single scale back-propagation
            element_prob = F.softmax(element_logits_target,-1) #[B,T,C+1]
            lpl_loss = -(torch.exp(-args.alpha*std_logits)*(1-pseudo_label_back.detach())*torch.log(1-element_prob + 1e-8)).sum(-1)

        return lpl_loss.mean(),pseudo_label_back,uncertainty_index

class Model_Ant(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim=2048
        dropout_ratio=args['opt'].dropout_ratio
        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2))
        self.att_rgb_encoder = nn.Sequential(
            nn.Conv1d(n_feature//2, embed_dim//2, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.att_flow_encoder = nn.Sequential(
            nn.Conv1d(n_feature//2, embed_dim//2, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(nn.Dropout(dropout_ratio),nn.Conv1d(embed_dim, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Dropout(0.7),nn.Conv1d(embed_dim, n_class+1, 1,bias=True))

        self.attention_rgb = nn.Sequential(nn.Conv1d(embed_dim//2, 1, 1))
        self.attention_flow = nn.Sequential(nn.Conv1d(embed_dim//2, 1, 1))

        _kernel=int(args['opt'].t//2*2+1)
        self.pool=nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()
        self.apply(weights_init)

    def forward(self, inputs, **args):
        x = inputs.transpose(-1, -2)
        feat = self.feat_encoder(x)
        x_cls = self.pool(self.classifier(feat))

        x_rgb_atn_logit = self.attention_rgb(self.att_rgb_encoder(x[:,:1024]))
        x_flow_atn_logit = self.attention_flow(self.att_flow_encoder(x[:,1024:]))

        x_rgb_atn = self.pool(torch.sigmoid(x_rgb_atn_logit))
        x_flow_atn = self.pool(torch.sigmoid(x_flow_atn_logit))
    
        atn_fusion_weight = args['opt'].delta
        x_atn = atn_fusion_weight*x_rgb_atn + (1-atn_fusion_weight)*x_flow_atn

        return feat.transpose(-1, -2), x_cls.transpose(-1, -2), x_atn.transpose(-1, -2), x_rgb_atn_logit.transpose(-1, -2), x_flow_atn_logit.transpose(-1, -2),x_rgb_atn.transpose(-1,-2),x_flow_atn.transpose(-1,-2)

    def criterion(self, outputs, labels, **args):

        feat, element_logits, x_atn, x_rgb_atn_logit, x_flow_atn_logit,x_rgb_atn,x_flow_atn = outputs #[B,T,2048] [B,T,C+1] [B,T,1]
        b,n,c = element_logits.shape

        element_logits_flow_supp = self._multiply(element_logits, x_flow_atn,include_min=True)
        element_logits_rgb_supp = self._multiply(element_logits, x_rgb_atn,include_min=True)

        # CLS
        loss_cls_orig, _ = self.topkloss(element_logits,
                                    labels,
                                    is_back=True,
                                    rat=args['opt'].k,
                                    reduce=None)
        # CLS_sup_flow
        loss_cls_flow_supp, _ = self.topkloss(element_logits_flow_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)
        # CLS_sup_rgb
        loss_cls_rgb_supp, _ = self.topkloss(element_logits_rgb_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)

        # CLL
        if args['opt'].lambda_cll !=0 :
            loss_3_orig_negative = self.negative_loss(element_logits,labels,is_back=True)
        else:
            loss_3_orig_negative = torch.tensor([0]).detach()

        # PL
        if args['opt'].lambda_pl !=0 and (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None):
            pseudo_label,uncertainty_index = self.Pseudo_Label_Generation(element_logits,x_atn,labels,args['opt'],args['device'],is_back=True)
            loss_pl = self.Pseudo_Label_Loss(element_logits,pseudo_label,x_rgb_atn,x_flow_atn,args['opt'])
        else:
            loss_pl = torch.tensor([0]).detach()


        # PCL & PLG
        if args['opt'].lambda_pcl !=0 and (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None):
            pseudo_complementary_label,uncertainty_index = self.Pseudo_Complementary_Label_Generation(element_logits,x_atn,labels,args['opt'],args['device'],is_back=True)
            loss_pcl = self.Pseudo_Complementary_Loss(element_logits,pseudo_complementary_label,x_rgb_atn,x_flow_atn,args['opt'])
        else:
            loss_pcl = torch.tensor([0]).detach()
            pseudo_complementary_label = None
            uncertainty_index = None

        # SCL loss
        if args['opt'].lambda_scl !=0 : 
            if args['opt'].SCL_method == 'no_pcl': 
                loss_scl = self.SCL_Loss(element_logits,uncertainty_index,x_atn,x_rgb_atn,x_flow_atn,args['opt'])
            elif (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None): 
                loss_scl = self.SCL_Loss_uncertainty(element_logits,uncertainty_index,x_atn,x_rgb_atn,x_flow_atn,args['opt'])
            else: 
                loss_scl = torch.tensor([0]).detach()
        else:
            loss_scl = torch.tensor([0]).detach()

        total_loss = args['opt'].lambda_cls * (loss_cls_orig.mean()+loss_cls_flow_supp.mean()+loss_cls_rgb_supp.mean()) +\
                        args['opt'].lambda_cll * loss_3_orig_negative +\
                        args['opt'].lambda_pcl * loss_pcl +\
                        args['opt'].lambda_scl * loss_scl +\
                        args['opt'].lambda_pl * loss_pl

        loss_dict={
            'loss':(total_loss).item(),
            'loss_Cls':(loss_cls_orig.mean()+loss_cls_flow_supp.mean()+loss_cls_rgb_supp.mean()).item(),
            'loss_CLL':loss_3_orig_negative.item(),
            'loss_PCL':loss_pcl.item(),
            'loss_SCL':loss_scl.item(),
            'loss_PL':loss_pl.item()
        }

        return total_loss,loss_dict,pseudo_complementary_label,uncertainty_index

    def ms_criterion(self,ms_outputs,labels,**args):

        feat, element_logits_target, x_atn, x_rgb_atn_logit, x_flow_atn_logit,x_rgb_atn,x_flow_atn = ms_outputs[0] #[B,T,2048] [B,T,C+1] [B,T,1]

        element_logits_flow_supp = self._multiply(element_logits_target, x_flow_atn,include_min=True)
        element_logits_rgb_supp = self._multiply(element_logits_target, x_rgb_atn,include_min=True)

        # CLS
        loss_cls_orig, _ = self.topkloss(element_logits_target,
                                    labels,
                                    is_back=True,
                                    rat=args['opt'].k,
                                    reduce=None)
        # CLS_sup_flow
        loss_cls_flow_supp, _ = self.topkloss(element_logits_flow_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)
        # CLS_sup_rgb
        loss_cls_rgb_supp, _ = self.topkloss(element_logits_rgb_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)

        # CLL
        if args['opt'].lambda_cll !=0 :
            loss_3_orig_negative = self.negative_loss(element_logits_target,labels,is_back=True)
        else:
            loss_3_orig_negative = torch.tensor([0]).detach()

        # PL
        if args['opt'].lambda_pl !=0 and (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None):
            pseudo_label,_ = self.Pseudo_Label_Generation(element_logits_target,x_atn,labels,args['opt'],args['device'],is_back=True)
            loss_pl = self.Pseudo_Label_Loss(element_logits_target,pseudo_label,x_rgb_atn,x_flow_atn,args['opt'])
        else:
            loss_pl = torch.tensor([0]).detach()

        # Multi_Scale_Label_Propagation
        ms_pcl = None # placeholder
        if args['opt'].lambda_lpl != 0 and (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None):
            lpl_loss,pseudo_complementary_label,uncertainty_index_median = self.Label_Propagation_Loss(ms_outputs,ms_pcl,labels,args['device'],args['opt'])
        else:
            lpl_loss = torch.tensor([0]).detach()
            pseudo_complementary_label = None
            uncertainty_index_median = None

        # Foreground_Background_Consistency for median scale
        if args['opt'].lambda_mscl !=0 : 
            if args['opt'].SCL_method == 'no_pcl': 
                loss_mscl = self.SCL_Loss(element_logits_target,uncertainty_index_median,x_atn,x_rgb_atn,x_flow_atn,args['opt'])
            elif (args['itr'] > args['opt'].pseudo_iter or args['opt'].pretrained_ckpt != None): 
                loss_mscl = self.SCL_Loss_uncertainty(element_logits_target,uncertainty_index_median,x_atn,x_rgb_atn,x_flow_atn,args['opt'])
            else: 
                loss_mscl = torch.tensor([0]).detach()
        else:
            loss_mscl = torch.tensor([0]).detach()


        total_loss = args['opt'].lambda_cls * (loss_cls_orig.mean()+loss_cls_flow_supp.mean()+loss_cls_rgb_supp.mean()) +\
                        args['opt'].lambda_cll * loss_3_orig_negative +\
                        args['opt'].lambda_lpl * lpl_loss +\
                        args['opt'].lambda_mscl * loss_mscl

        loss_dict={
            'loss':(total_loss).item(),
            'loss_Cls':(loss_cls_orig.mean()+loss_cls_flow_supp.mean()+loss_cls_rgb_supp.mean()).item(),
            'loss_CLL':loss_3_orig_negative.item(),
            'loss_LPL':lpl_loss.item(),
            'loss_mSCL':loss_mscl.item()
        }

        return total_loss, loss_dict,pseudo_complementary_label,uncertainty_index_median

    def _multiply(self, x, atn, dim1=-1, dim2=-2, include_min=False):
        '''
        class_min
        '''
        if include_min:
            _min = x.min(dim=dim1, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def _KL_divergence(self,A_prob, B_prob):
        return A_prob * (torch.log(A_prob) - torch.log(B_prob))

    def _symKL(self,A_prob, B_prob):
        return 0.5*A_prob * (torch.log(A_prob) - torch.log(B_prob))+0.5*B_prob*(torch.log(B_prob)-torch.log(A_prob))
    
    def _sKL(self,A_prob, B_prob):
        return A_prob * (torch.log(A_prob) - torch.log(B_prob))

    def _dKL(self,A_prob, B_prob):
        return A_prob.detach() * (torch.log(A_prob.detach()) - torch.log(B_prob))

    def _JS(self,A_prob, B_prob):
        M_prob = (A_prob+B_prob)/2
        return 0.5*A_prob * (torch.log(A_prob) - torch.log(M_prob))+0.5*B_prob*(torch.log(B_prob)-torch.log(M_prob))

    def _binary_cross_entropy(self,pred, target):
        return -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)

    def _MSE(self,pred,target):
        return ((pred-target)**2).mean()

    def _MAE(self,pred,target):
        return (pred-target).abs().mean()

    def _Tanh(self,unconsistency,slope,coefficient=1):
        weight = (2/(1+torch.exp(-2*slope*unconsistency)))-1
        return weight

    def negative_loss(self,element_logits,labels,is_back=False,focal=False):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        b,n,c = element_logits.shape

        element_prob = F.softmax(element_logits,-1)

        labels_with_back = labels_with_back.unsqueeze(1).expand(b,n,c)

        s_loss = -((1-labels_with_back)*torch.log(1 - element_prob+1e-8)).sum(-1)

        return s_loss.mean()

    def topkloss(self,element_logits,labels,is_back=True,lab_rand=None,rat=8,reduce=None,focal=False):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)


        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )

        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)

        milloss = (-(labels_with_back *
                         F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))

        return milloss, topk_ind

    def Pseudo_Label_Generation(self,element_logits,element_atn,labels,args,device,is_back=True):

        b,n,c = element_logits.shape
        pseudo_label_back = np.zeros((b,n,c)) #[B,T,C+1]
        if is_back:
            pseudo_label_back[:,:,-1] = 1 #[B,T,C+1]
        else: 
            pass

        labels = labels.cpu().numpy() 
        element_logits = element_logits.detach().cpu().numpy()
        element_atn = element_atn.detach().cpu().numpy()

        # att_thresh_list = np.arange(0.4, 0.925, 0.025) # Follow ASM_loc
        att_thresh_list = np.arange(args.PLG_act_thres[0],args.PLG_act_thres[1],args.PLG_act_thres[2])

        batch_proposal_list = []
        for v in range(b): 
            v_logits = element_logits[v].copy() # [T,C+1]
            v_atn = element_atn[v].copy() #[T,1]
            v_atn_logits = v_logits*v_atn #[T,C+1]
            v_gt_index = np.where(labels[v]>0)[0]
            v_gt_atn_logits = v_atn_logits[:,v_gt_index]  #[T,gt]

            # generate proposals
            proposal_dict = {}
            for att_thresh in att_thresh_list:
                seg_list = []

                for c in v_gt_index:
                    if args.PLG_proposal_mode == 'atn':
                        pos = np.where(v_atn[:,0]>att_thresh)[0]
                    elif args.PLG_proposal_mode == 'logits':
                        pos = np.where(v_logits[:,c]>att_thresh)[0]
                    elif args.PLG_proposal_mode == 'atn_logits':
                        pos = np.where(v_atn_logits[:,c]>att_thresh)[0]
                    else:
                        raise ValueError("Don't define this proposal_mode. ")
                    seg_list.append(pos)

                proposals = utils.get_pseudo_proposal_oic(seg_list,
                                    v_gt_atn_logits,
                                    v_gt_index
                                    )

                for j in range(len(proposals)):
                    try:
                        class_id = proposals[j][0][0]

                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []

                        proposal_dict[class_id] += proposals[j]
                    except IndexError:
                        raise IndexError(f"Index error")

            # nms and update pseudo label for current video
            pseudo_label_back,final_proposal=utils.nms_wiht_pseudo_label(proposal_dict,
                                                            pseudo_label_back,
                                                            v,
                                                            iou_thr=0.0
                                                            )

            batch_proposal_list.append(final_proposal)

        return torch.from_numpy(pseudo_label_back).float().to(device),batch_proposal_list

    def Pseudo_Complementary_Label_Generation(self,element_logits,element_atn,labels,args,device,is_back=True):

        b,n,c = element_logits.shape
        bg_id = c-1 
        pseudo_label_back = torch.ones((b,n,c)) #[B,T,C+1]
        uncertainty_index = torch.zeros((b,n,1)) #[B,T,1]

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else: 
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        labels = labels_with_back # [B,C+1]
        element_logits = element_logits.detach()

        for v in range(b): 
            v_logits = element_logits[v] # [T,C+1]

            if args.PLG_logits_mode == 'norm':
                norm_v_logits = (v_logits-torch.min(v_logits))/(torch.max(v_logits)-torch.min(v_logits)) # min-max normalization
            elif args.PLG_logits_mode == 'none':
                norm_v_logits = v_logits
            else:
                raise ValueError("Don't define this PLG_logits_mode. ")

            v_gt_index = torch.where(labels[v]>0)[0] 
            v_gt_logits = norm_v_logits[:,v_gt_index] # [T,c]
            # print("v_gt_logits.shape:",v_gt_logits.shape)
            v_softmax_gt_logits = torch.softmax(v_gt_logits,dim=-1) # [T,c]
            # print("v_softmax_gt_logits.shape:",v_softmax_gt_logits.shape)
            v_softmax_gt_mean = torch.mean(v_softmax_gt_logits,dim=-1) # [T]
            # print("v_gt_mean.shape:",v_gt_mean.shape)
            v_fg_info = torch.sum(v_softmax_gt_logits[:,:-1],dim=-1) # [T]
            # print("v_fg_info.shape:",v_fg_info.shape)
            v_bg_info = v_softmax_gt_logits[:,-1] #[T]
            # print("v_bg_info.shape:",v_bg_info.shape)
            v_info_entropy = -v_fg_info*torch.log(v_fg_info)-v_bg_info*torch.log(v_bg_info) #[T]
            # print("v_info_entropy.shape:",v_info_entropy.shape)
            for t in range(n):
                if v_info_entropy[t] >= args.PLG_thres: # uncertainty snippet
                    uncertainty_index[v,t,:] = 1
                else:
                    if len(v_gt_index)>2:
                        indices = torch.where(v_softmax_gt_logits[t] < v_softmax_gt_mean[t])[0] 
                        class_indices = v_gt_index[indices] 

                        max_indice = torch.argmax(v_softmax_gt_logits[t])
                        class_max_indice = v_gt_index[max_indice] 

                        if (bg_id not in class_indices) and (class_max_indice==bg_id): 
                            pseudo_label_back[v,t,v_gt_index] = 0 
                            pseudo_label_back[v,t,bg_id] = 1
                        elif (bg_id not in class_indices) and (class_max_indice!=bg_id): 
                            pseudo_label_back[v,t,class_indices] = 0 
                            pseudo_label_back[v,t,bg_id] = 0 
                    else:
                        indices = torch.where(v_softmax_gt_logits[t] < v_softmax_gt_mean[t])[0] 
                        class_indices = v_gt_index[indices]
                        pseudo_label_back[v,t,class_indices] = 0

        return pseudo_label_back.long().to(device),uncertainty_index.long().to(device)

    def Pseudo_Complementary_Loss(self,element_logits,pseudo_label_back,rgb_atn,flow_atn,args,std_weight=None):

        b,n,c = element_logits.shape #[B,T,C+1]
        pseudo_label = pseudo_label_back.clone() #[B,T,C+1]
        element_prob = F.softmax(element_logits,-1) #[B,T,C+1]
        
        pcl_loss = -((1-pseudo_label)*torch.log(1-element_prob + 1e-8)).sum(-1)

        return pcl_loss.mean()

    def SCL_Loss(self,element_logits,uncertainty_index,x_atn,rgb_atn,flow_atn,args):
        bg_atn = 1-x_atn # [B,T,1]
        element_prob = torch.softmax(element_logits, dim=-1)[..., [-1]] # [B,T,1]

        kl_att_cls = self._KL_divergence(bg_atn+1e-8, element_prob+1e-8)
        kl_cls_att = self._KL_divergence(element_prob+1e-8, bg_atn+1e-8)

        if args.SCL_align=='BCE':
            s_loss = args.factor *(self._binary_cross_entropy(bg_atn, element_prob.detach())* \
                            (torch.exp(-args.SCL_alpha * kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*kl_att_cls))).detach()).sum()
            s_loss += (1 - args.factor) * (self._binary_cross_entropy(element_prob, bg_atn.detach())* \
                                    (torch.exp(-args.SCL_alpha * kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        elif args.SCL_align=='MSE':
            s_loss = args.factor *(self._MSE(bg_atn, element_prob.detach())* \
                            (torch.exp(-args.SCL_alpha * kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*kl_att_cls))).detach()).sum()
            s_loss += (1 - args.factor) * (self._MSE(element_prob, bg_atn.detach())* \
                                    (torch.exp(-args.SCL_alpha * kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        elif args.SCL_align=='symKL':
            s_loss = args.factor *(self._symKL(bg_atn, element_prob.detach())* \
                            (torch.exp(-args.SCL_alpha * kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*kl_att_cls))).detach()).sum()
            s_loss += (1 - args.factor) * (self._symKL(element_prob, bg_atn.detach())* \
                                    (torch.exp(-args.SCL_alpha * kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        elif args.SCL_align=='MAE':
            s_loss = args.factor *(self._MAE(bg_atn, element_prob.detach())* \
                            (torch.exp(-args.SCL_alpha * kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*kl_att_cls))).detach()).sum()
            s_loss += (1 - args.factor) * (self._MAE(element_prob, bg_atn.detach())* \
                                    (torch.exp(-args.SCL_alpha * kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        elif args.SCL_align=='Tanh':
            s_loss = args.factor *(self._binary_cross_entropy(bg_atn, element_prob.detach())* \
                            (self._Tanh(kl_att_cls,args.SCL_alpha)/torch.sum(self._Tanh(kl_att_cls,args.SCL_alpha))).detach()).sum()
            s_loss += (1 - args.factor) * (self._binary_cross_entropy(element_prob, bg_atn.detach())* \
                                    (self._Tanh(kl_cls_att,args.SCL_alpha)/torch.sum(torch.exp(-args.SCL_alpha*kl_cls_att))).detach()).sum()
        else:
            raise ValueError("Don't define this SCL_align mode.")

        return s_loss

    def SCL_Loss_uncertainty(self,element_logits,uncertainty_index,x_atn,rgb_atn,flow_atn,args):
        bg_atn = 1-x_atn
        b,n,c = element_logits.shape
        element_prob = torch.softmax(element_logits, dim=-1)[..., [-1]] # [B,T,1]

        kl_att_cls = self._KL_divergence(bg_atn+1e-8, element_prob+1e-8) # [B,T,1]
        kl_cls_att = self._KL_divergence(element_prob+1e-8, bg_atn+1e-8) # [B,T,1]

        for v in range(b):
            v_bg_atn = bg_atn[v] #[B,T,1]
            v_bg_logits = element_prob[v] #[T,1]
            v_ut_index = torch.where(uncertainty_index[v] > 0)[0] #[un_index_num]
            v_kl_att_cls = kl_att_cls[v]
            v_kl_cls_att = kl_cls_att[v]

            v_ut_bg_logits = v_bg_logits[v_ut_index,:] #[uncertainty_index,1]
            v_ut_bg_atn = v_bg_atn[v_ut_index,:] #[uncertainty_index,1]
            v_ut_kl_att_cls = v_kl_att_cls[v_ut_index,:]
            v_ut_kl_cls_att = v_kl_cls_att[v_ut_index,:]
            
            # for element_prob
            if args.SCL_align=='BCE':
                s_loss = args.factor *(self._binary_cross_entropy(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._binary_cross_entropy(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='MSE':
                s_loss = args.factor *(self._MSE(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._MSE(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='symKL':
                s_loss = args.factor *(self._symKL(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._symKL(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='MAE':
                s_loss = args.factor *(self._MAE(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._MAE(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='sKL': 
                s_loss = (self._sKL(v_ut_bg_atn, v_ut_bg_logits)* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
            elif args.SCL_align=='dKL': 
                s_loss = args.factor *(self._dKL(v_ut_bg_atn, v_ut_bg_logits)* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
            elif args.SCL_align=='JS':
                s_loss = args.factor *(self._JS(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (torch.exp(-args.SCL_alpha * v_ut_kl_att_cls)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_att_cls))).detach()).sum()
                s_loss += (1 - args.factor) * (self._JS(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (torch.exp(-args.SCL_alpha * v_ut_kl_cls_att)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='Tanh':
                s_loss = args.factor *(self._binary_cross_entropy(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                (self._Tanh(v_ut_kl_att_cls,args.SCL_alpha)/torch.sum(self._Tanh(v_ut_kl_att_cls,args.SCL_alpha))).detach()).sum()
                s_loss += (1 - args.factor) * (self._binary_cross_entropy(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        (self._Tanh(v_ut_kl_att_cls,args.SCL_alpha)/torch.sum(torch.exp(-args.SCL_alpha*v_ut_kl_cls_att))).detach()).sum()
            elif args.SCL_align=='Dexp':
                s_loss = args.factor *(self._binary_cross_entropy(v_ut_bg_atn, v_ut_bg_logits.detach())* \
                                ((torch.exp(args.SCL_alpha * v_ut_kl_att_cls)-1)/torch.sum(torch.exp(args.SCL_alpha*v_ut_kl_att_cls)-1)).detach()).sum()
                s_loss += (1 - args.factor) * (self._binary_cross_entropy(v_ut_bg_logits, v_ut_bg_atn.detach())* \
                                        ((torch.exp(args.SCL_alpha * v_ut_kl_cls_att)-1)/torch.sum(torch.exp(args.SCL_alpha*v_ut_kl_cls_att)-1)).detach()).sum()
            else:
                raise ValueError("Don't define this SCL_align mode.")
        s_loss = s_loss / b

        return s_loss
    
    def Pseudo_Label_Loss(self,element_logits,pseudo_label_back,rgb_atn,flow_atn,args):
        b,n,c = element_logits.shape #[B,T,C+1]
        pseudo_label = pseudo_label_back #[B,T,C+1]
        element_prob = F.softmax(element_logits,-1) #[B,T,C+1]

        pll_loss = -(pseudo_label*torch.log(element_prob + 1e-8)).mean()

        return pll_loss.mean()

    def Label_Propagation_Loss(self,ms_outputs:dict,ms_pcl:dict,labels,device,args):
        _, element_logits_target, x_atn_target, _, _,x_rgb_atn_target,x_flow_atn_target = ms_outputs[0] #[B,T,2048] [B,T,C+1] [B,T,1]
        b,t,c = element_logits_target.shape
        logits_stack = [element_logits_target]
        for i in range(1,len(ms_outputs)):
            if args.rescale_mode=='linear':
                logits = F.interpolate(ms_outputs[i][1].transpose(-1,-2),size=t,mode="linear").transpose(-1,-2)
            elif args.rescale_mode=='nearest':
                logits = F.interpolate(ms_outputs[i][1].transpose(-1,-2),size=t,mode="nearest").transpose(-1,-2)
            else:
                raise ValueError("Don't define this rescale model. ")
            
            logits_stack.append(logits)
        std_logits = torch.std(torch.stack(logits_stack,dim=0),dim=0) # [N,B,T,C+1]->[B,T,C+1]

        weighted_norm_logits_stack=[]
        for logits,weight in zip(logits_stack,args.ensemble_weight):
            if args.lpl_norm=='min_max':
                # global min_max normalization, then class_softmax
                _min = torch.min(torch.min(logits,-1,keepdim=True)[0],-2,keepdim=True)[0] #[B,1,1]
                _max = torch.max(torch.max(logits,-1,keepdim=True)[0],-2,keepdim=True)[0] #[B,1,1]
                norm_logits = (logits-_min)/(_max-_min)  #[B,T,C+1]
            elif args.lpl_norm == 'positive':
                _min = torch.min(torch.min(logits,-1,keepdim=True)[0],-2,keepdim=True)[0] #[B,1,1]
                norm_logits = logits-_min
            elif args.lpl_norm == 'none':
                norm_logits = logits
            else:
                raise ValueError("Don't define this lpl_norm model. ")
            weighted_norm_logits = norm_logits*weight
            weighted_norm_logits_stack.append(weighted_norm_logits)

        mean_logits_stack = torch.stack(weighted_norm_logits_stack,dim=0) # [N,B,T,C+1]
        mean_logits = torch.sum(mean_logits_stack,dim=0) #[B,T,C+1]
        std_logits = std_logits.detach()

        pseudo_label_back,uncertainty_index = self.Pseudo_Complementary_Label_Generation(mean_logits,None,labels,args,device,is_back=True)

        if args.multi_back: 
            element_prob = F.softmax(mean_logits,-1) #[B,T,C+1]
            lpl_loss = -(torch.exp(-args.alpha*std_logits)*(1-pseudo_label_back.detach())*torch.log(1-element_prob + 1e-8)).sum(-1)
        else: 
            element_prob = F.softmax(element_logits_target,-1) #[B,T,C+1]
            lpl_loss = -(torch.exp(-args.alpha*std_logits)*(1-pseudo_label_back.detach())*torch.log(1-element_prob + 1e-8)).sum(-1)

        return lpl_loss.mean(),pseudo_label_back,uncertainty_index


