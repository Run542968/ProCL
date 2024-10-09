from tkinter.messagebox import NO
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils.wsad_utils as utils
import numpy as np
from torch.autograd import Variable
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.eval_detection import ANETdetection
import wsad_dataset
from eval.detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
import pdb
import multiprocessing as mp
import options
import model_newest
import proposal_methods as PM
import pandas as pd
from collections import defaultdict
from utils.wsad_utils import get_logger,get_timestamp
import os
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def Pseudo_Label_Generation(element_logits,element_atn,labels,args,device,is_back=True):

    b,n,c = element_logits.shape
    pseudo_label_back = np.zeros((b,n,c)) #[B,T,C+1]

    if is_back:
        pseudo_label_back[:,:,-1] = 1 #[B,T,C+1]
    else: 
        pass

    labels = labels 
    element_logits = element_logits.detach().cpu().numpy()
    element_atn = element_atn.detach().cpu().numpy()

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

    return pseudo_label_back,batch_proposal_list

def Pseudo_Complementary_Label_Generation(element_logits,element_atn,labels,args,device,is_back=True):
    b,n,c = element_logits.shape
    bg_id = c-1 
    pseudo_label_back = torch.ones((b,n,c)) #[B,T,C+1]
    uncertainty_index = torch.zeros((b,n,1)) #[B,T,1]
    labels = torch.from_numpy(labels).unsqueeze(dim=0) # [1,C]

    if is_back:
        labels_with_back = torch.cat(
            (labels, torch.ones_like(labels[:, [0]])), dim=-1)
    else: 
        labels_with_back = torch.cat(
            (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

    labels = labels_with_back 
    element_logits = element_logits.detach()

    for v in range(b): 
        v_logits = element_logits[v] 

        if args.PLG_logits_mode == 'norm':
            norm_v_logits = (v_logits-torch.min(v_logits))/(torch.max(v_logits)-torch.min(v_logits)) # min-max normalization
        elif args.PLG_logits_mode == 'none':
            norm_v_logits = v_logits
        else:
            raise ValueError("Don't define this PLG_logits_mode. ")

        v_gt_index = torch.where(labels[v]>0)[0] 
        v_gt_logits = norm_v_logits[:,v_gt_index] # [T,c]

        v_softmax_gt_logits = torch.softmax(v_gt_logits,dim=-1) # [T,c]
        v_softmax_gt_mean = torch.mean(v_softmax_gt_logits,dim=-1) # [T]

        v_fg_info = torch.sum(v_softmax_gt_logits[:,:-1],dim=-1) # [T]
        v_bg_info = v_softmax_gt_logits[:,-1] #[T]
        v_info_entropy = -v_fg_info*torch.log(v_fg_info)-v_bg_info*torch.log(v_bg_info) #[T]

        for t in range(n):
            if v_info_entropy[t] >= args.PLG_thres: # uncertainty snippet
                uncertainty_index[v,t,:] = 1
            else:
                if len(v_gt_index)>2: 
                    indices = torch.where(v_softmax_gt_logits[t] < v_softmax_gt_mean[t])[0] 
                    indices = indices.to(v_gt_index.device)
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
                    indices = indices.to(v_gt_index.device)
                    class_indices = v_gt_index[indices]
                    pseudo_label_back[v,t,class_indices] = 0

    return pseudo_label_back.long().detach().cpu().numpy(), uncertainty_index.long().detach().cpu().numpy()


@torch.no_grad()
def test(itr, dataset, args, model, logger, device,writer,pool):
    model.eval()
    done = False
    instance_logits_stack = []
    element_logits_stack = {}
    logits_stack={}
    attn_stack={}
    video_list_stack=[]
    labels_stack = []
    pseudo_label_stack = {}
    unertainty_index_stack = {}

    proposals = []
    count = 0 

    while not done:
        if dataset.currenttestidx % (len(dataset.testidx) // 5) == 0:
            print('Testing test data point %d of %d' % (dataset.currenttestidx, len(dataset.testidx)))

        data_dict=dataset.load_data(is_training=False)
        features,labels,vn,done=data_dict['feat'],data_dict['lab'],data_dict['vn'],data_dict['done'] # labels: [C]


        seq_len = [features.shape[0]]
        if seq_len == 0:
            continue
        features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(Variable(features), is_training=False, opt=args, seq_len=seq_len)

            _, element_logits, atn_supp, _, _, _, _ = outputs #_,logits,atn,_,_,_,_

            proposals.append(getattr(PM,args.test_proposal_method)(vn,args, element_logits, atn_supp))
            
            logits = element_logits.squeeze(0)
        tmp = F.softmax(torch.mean(torch.topk(logits, k=int(np.ceil(len(features) / 8)), dim=0)[0], dim=0),
                        dim=0).cpu().data.numpy()
        
        if args.PLG_method == 'Pseudo_Label_Generation':
            pseudo_label,_ = Pseudo_Label_Generation(element_logits,atn_supp,labels,args,device,is_back=True)
            uncertainty_index = None
        elif args.PLG_method == 'Pseudo_Complementary_Label_Generation':
            pseudo_label,uncertainty_index = Pseudo_Complementary_Label_Generation(element_logits,atn_supp,labels,args,device,is_back=True)
        else:
            raise ValueError("Don't define this PLG_method. ")
        
        instance_logits_stack.append(tmp) # video-level prediction [C+1]
        element_logits_stack[vn]=(element_logits).detach().cpu().numpy() # [1,T,C+1]
        logits_stack[vn]=logits.detach().cpu().numpy() # [T,C+1]
        attn_stack[vn]=atn_supp.detach().cpu().numpy() # [1,T,1]
        video_list_stack.append(vn)
        labels_stack.append(labels) # [C]
        pseudo_label_stack[vn]=pseudo_label # [1,T,C+1]
        unertainty_index_stack[vn]=uncertainty_index # [1,T,1]

    instance_logits_stack = np.array(instance_logits_stack) # [N,C+1]
    labels_stack = np.array(labels_stack) # [N,C+1]
    proposals = pd.concat(proposals).reset_index(drop=True) # [N,4]

    if 'Thumos' in args.dataset_name or 'BEOID' in args.dataset_name or 'GTEA' in args.dataset_name:
        iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)
        dmap_detect.ground_truth.to_csv('temp/groundtruth_'+str(args.dataset_name)+'.csv')
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()
    else:
        iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, subset='validation')
        dmap_detect.ground_truth.to_csv('temp/groundtruth_'+str(args.dataset_name)+'.csv')
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()


    ############ for Snippet-level classification ######################
    groundTruthGroupByVideoId = dmap_detect.ground_truth.groupby('video-id')
    # to evaluate the snippet-level classification
    import copy
    from sklearn.metrics import accuracy_score
    def calSnippetClassificationAcc(pred, vns, labels):
        labels = copy.deepcopy(labels) #[N,T,C+1]

        preds = []
        for i, vn in enumerate(vns):
            preds.append(pred[vn].squeeze())

        preds = np.concatenate(preds, axis=0) #[N*T,C+1]
        labels = np.concatenate(labels, axis=0) #[N*T,C+1]

        args_preds = np.argmax(preds, axis=-1) # [N*T] 
        args_labels = np.argmax(labels, axis=-1) 

        return accuracy_score(args_labels, args_preds)

    def softmax(pred):
        return np.exp(pred) / np.exp(pred).sum(axis=-1, keepdims=True)

    def calBinaryClassificationAcc(pred,vns,labels,background=False):
        labels = copy.deepcopy(labels)

        preds = []
        for i, vn in enumerate(vns):
            preds.append(pred[vn].squeeze())

        preds = np.concatenate(preds, axis=0) #[N*T,C+1]
        labels = np.concatenate(labels, axis=0) #[N*T,C+1]

        if background:
            preds=1-softmax(preds)
            preds=preds[...,-1]
        foreground_labels=1-labels[...,-1]

        # generate preds to 0-1
        fore_preds=np.array([p>0.5 for p in preds]).astype(float)
        # translate the foreground_labels and foreground_preds
        return ((fore_preds * foreground_labels).sum() +
                ((1 - fore_preds) * (1 - foreground_labels)).sum()) \
               / foreground_labels.shape[0]

    from sklearn.metrics import roc_auc_score
    def calBinaryAUC(pred,vns,labels,background=False):
        labels = copy.deepcopy(labels)

        preds = []
        for i, vn in enumerate(vns):
            preds.append(pred[vn].squeeze())

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        if background:
            # preds = 1 - softmax(preds)
            preds = 1-softmax(preds-np.max(preds,axis=-1,keepdims=True)) # follow https://www.cnblogs.com/guoyaohua/p/8900683.html to solve the NaN
            preds = preds[..., -1]
        foreground_labels = 1 - labels[..., -1]

        return roc_auc_score(foreground_labels.astype(int), preds)

    def calPseudoComplementaryLabelPrecision(pcl,ui,vns,video_gt,snippet_gt):
        snippet_GT = copy.deepcopy(snippet_gt) #[N,T,C+1]

        pcl_preds = []
        for i, vn in enumerate(vns):
            pcl_preds.append(pcl[vn].squeeze()) # [T,C+1]

        preds = np.concatenate(pcl_preds, axis=0) # [N*T,C+1] 
        labels = np.concatenate(snippet_GT, axis=0) # [N*T,C+1] 
        complementary_labels = 1-labels # 
        c_preds = 1-preds # 

        P = c_preds.sum()
        TP = (c_preds*complementary_labels).sum()
        if P==0:
            precision=0
        else:
            precision = TP/P

        return precision

    def calPseudoLabelPrecision(pl,vns,video_gt,snippet_gt):
        snippet_GT = copy.deepcopy(snippet_gt) #[N,T,C+1]

        pl_preds = []
        for i, vn in enumerate(vns):
            pl_preds.append(pl[vn].squeeze()) # [T,C+1]

        preds = np.concatenate(pl_preds, axis=0) #[N*T,C+1] 
        labels = np.concatenate(snippet_GT, axis=0) #[N*T,C+1] 

        P = preds.sum()
        TP = (preds*labels).sum()
        if P==0:
            precision=0
        else:
            precision = TP/P
        return precision
    
    def calmAP(pred,vns,snippet_gt):
        '''
        https://github.com/piergiaj/super-events-cvpr18/blob/50282dc55364cd613f8de440c2357d588547ca76/train_model.py#L234
        '''
        snippet_GT = copy.deepcopy(snippet_gt) #[N,T,C+1]

        preds = []
        for i, vn in enumerate(vns):
            preds.append(pred[vn].squeeze()) # [T,C+1]

        preds = np.concatenate(preds, axis=0) #[N*T,C+1] 
        snippet_GT = np.concatenate(snippet_GT, axis=0) #[N*T,C+1]

        scores = torch.from_numpy(preds).sigmoid()
        targets = torch.from_numpy(snippet_GT)

        if scores.numel() == 0:
            return 0
        ap = torch.zeros(scores.size(1)).cpu()
        rg = torch.range(1, scores.size(0)).float().cpu()

        # compute average precision for each class
        for k in range(scores.size(1)):
            # sort scores
            score = scores[:, k]
            target = targets[:, k]
            _, sortind = torch.sort(score, 0, True)
            truth = target[sortind]

            tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.byte()].sum() / max(truth.sum(), 1)
        
        mAP = torch.sum(100 * ap) / torch.nonzero(100 * ap).size()[0]

        return mAP.numpy()

    snippet_labels=[]
    for vn in video_list_stack: 
        p = logits_stack[vn].squeeze() # [T,C+1]
        # [t,c+1]
        snippet_label = np.zeros_like(p) # [T,C+1]
        snippet_label[:, -1] = 1 

        if 'ActivityNet1.3' not in args.dataset_name:
            gt = groundTruthGroupByVideoId.get_group(vn.decode())
        else:
            gt = groundTruthGroupByVideoId.get_group(vn)
        
        for idx, this_pred in gt.iterrows():
            snippet_label[this_pred['t-start']:this_pred["t-end"], this_pred['label']] = 1
            snippet_label[this_pred['t-start']:this_pred["t-end"], -1] = 0
        snippet_labels.append(snippet_label)

    pred_cmap = calSnippetClassificationAcc(logits_stack, video_list_stack, snippet_labels)
    pred_bin_cmap=calBinaryClassificationAcc(logits_stack,video_list_stack,snippet_labels,background=True)
    att_bin_cmap=calBinaryClassificationAcc(attn_stack,video_list_stack,snippet_labels,background=False)
    pred_bin_auc=calBinaryAUC(logits_stack,video_list_stack,snippet_labels,background=True)
    att_bin_auc=calBinaryAUC(attn_stack,video_list_stack,snippet_labels,background=False)
    if args.PLG_method == 'Pseudo_Label_Generation':
        pseudo_label_precision=calPseudoLabelPrecision(pseudo_label_stack,video_list_stack,labels_stack,snippet_labels)
    elif args.PLG_method == 'Pseudo_Complementary_Label_Generation':
        pseudo_label_precision=calPseudoComplementaryLabelPrecision(pseudo_label_stack,unertainty_index_stack,video_list_stack,labels_stack,snippet_labels)
    else:
        raise ValueError("Don't define this PLG_method. ")
    mAP = calmAP(logits_stack,video_list_stack,snippet_labels)
    
    print('snippet-level classification mAP:{}'.format(pred_cmap),
          'snippet_binary_classification mAP:{}'.format(pred_bin_cmap),
          'snippet_binary_attention mAP:{}'.format(att_bin_cmap),
          'snippet_binary_classification ACU:{}'.format(pred_bin_auc),
          'snippet_binary_attention AUC:{}'.format(att_bin_auc),
          'pseudo_label_Precision:{}'.format(pseudo_label_precision),
          'mAP:{}'.format(mAP)
          )

    #########################################################



    if logger is not None and writer is not None: 
        cmap = cmAP(instance_logits_stack, labels_stack)
        logger.info('Test Classification mAP:{},{}'.format(cmap, itr))
        logger.info('||'.join(['map @ {} = {:.3f} '.format(iou[i], dmap[i] * 100) for i in range(len(iou))]))
        logger.info('mAP Avg ALL: {:.3f}'.format(sum(dmap) / len(iou) * 100))
        
        # acc
        writer.add_scalar('pred_cmap',pred_cmap,itr)
        writer.add_scalar('pred_bin_cmap',pred_bin_cmap,itr)
        writer.add_scalar('att_bin_cmap',att_bin_cmap,itr)
        writer.add_scalar('pred_bin_auc',pred_bin_auc,itr)
        writer.add_scalar('att_bin_auc',att_bin_auc,itr)
        writer.add_scalar('pseudo_label_precision',pseudo_label_precision,itr)

        writer.add_scalar('mAP Avg ALL',sum(dmap) / len(iou) * 100,itr)
        if 'Thumos' in args.dataset_name or 'BEOID' in args.dataset_name or 'GTEA' in args.dataset_name:
            writer.add_scalar('mAP Avg 0.1-0.5',np.mean(dmap[:5])*100,itr)
            writer.add_scalar('mAP Avg 0.1-0.7',np.mean(dmap[:7])*100,itr)
            writer.add_scalar('mAP Avg 0.3-0.7',np.mean(dmap[2:7])*100,itr)
            logger.info('mAP Avg 0.1-0.5: {:.3f}, mAP Avg 0.1-0.7: {:.3f}, mAP Avg 0.3-0.7: {:.3f}'.format(np.mean(dmap[:5])*100,np.mean(dmap[:7])*100,np.mean(dmap[2:7])*100))

        for item in list(zip(dmap, iou)):
            writer.add_scalar('mAP@IoU '+str(item[1]),item[0], itr)
            logger.info('Test Detection mAP @ IoU ={}-{},{} '.format(str(item[1]), item[0], itr))

        utils.write_to_csv("./results/" + args.dataset_name +"_"+args.model_name, dmap, cmap, pseudo_label_precision,mAP, itr,args.dataset_name)


    return iou, dmap, pseudo_label_precision


if __name__ == '__main__':
    args = options.parser.parse_args()
    device = torch.device("cuda")
    dataset = getattr(wsad_dataset, args.dataset)(args)

    model = getattr(model_newest, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load('./ckpt/best_' + args.model_name + '.pkl'))
    pool = mp.Pool(5)

    logger = None
    writer = None
    iou, dmap, pl_precision = test(-1, dataset, args, model, logger, device, writer,pool)
    print('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(dmap[:5]) * 100,
                                                                             np.mean(dmap[:7]) * 100,
                                                                             np.mean(dmap) * 100))
    print('Pseudo Label Precision: ',pl_precision)
