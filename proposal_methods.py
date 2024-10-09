
import numpy as np
import torch
import utils.wsad_utils as utils
from scipy.signal import savgol_filter
import pdb
import pandas as pd
import options
args = options.parser.parse_args()

def filter_segments(segment_predict, vn,factor):
    ambilist = './Thumos14reduced-Annotations/Ambiguous_test.txt'
    try:
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ") for a in ambilist]
    except:
        ambilist = []
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        #s[j], e[j], np.max(seg)+0.7*c_s[c],c]
        for a in ambilist:
            if a[0] == vn:
                gt = range(
                    int(round(float(a[2]) * factor)), int(round(float(a[3]) * factor))
                )
                pd = range(int(segment_predict[i][0]), int(segment_predict[i][1]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(
                    len(set(gt).union(set(pd)))
                )
                if IoU > 0:
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)

def filter_segments_2(segment_predict, vn):
    ambilist = './Thumos14reduced-Annotations/Ambiguous_test.txt'
    try:
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ") for a in ambilist]
    except:
        ambilist = []
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        #s[j], e[j], np.max(seg)+0.7*c_s[c],c]
        for a in ambilist:
            if a[0] == vn:
                if not (max(float(a[2]),segment_predict[i][0])>=min(float(a[3]),segment_predict[i][1])):
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)

def smooth(v, order=2,lens=200):
    l = min(lens, len(v))
    l = l - (1 - l % 2)
    if len(v) <= order:
        return v
    return savgol_filter(v, l, order)

def get_topk_mean(x, k, axis=0):
    return np.mean(np.sort(x, axis=axis)[-int(k):, :], axis=0)

def get_cls_score(element_cls, dim=-2, rat=20, ind=None):

    topk_val, _ = torch.topk(element_cls,
                             k=max(1, int(element_cls.shape[-2] * rat)),
                             # k=max(1,int(element_cls.shape[-2]*rat)),
                             dim=-2)
    instance_logits = torch.mean(topk_val, dim=-2)
    pred_vid_score = torch.softmax(
        instance_logits, dim=-1)[..., :-1].squeeze().data.cpu().numpy()
    return pred_vid_score #[C]

def _get_vid_score(pred):
    # pred : (n, class)
    if args is None:
        k = 8
        topk_mean = self.get_topk_mean(pred, k)
        # ind = topk_mean > -50
        return pred, topk_mean

    win_size = int(args.topk)
    split_list = [i*win_size for i in range(1, int(pred.shape[0]//win_size))]
    splits = np.split(pred, split_list, axis=0)

    tops = []
    #select the avg over topk2 segments in each window
    for each_split in splits:
        top_mean = get_topk_mean(each_split, args.topk2)
        tops.append(top_mean)
    tops = np.array(tops)
    c_s = np.max(tops, axis=0)
    return pred, c_s

def __vector_minmax_norm(vector, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        max_val = np.max(vector)
        min_val = np.min(vector)

    delta = max_val - min_val
    # delta[delta <= 0] = 1
    ret = (vector - min_val) / delta

    return ret

def _multiply(x, atn, dim1=-1, dim2=-2, include_min=False):
    '''
    class_min
    '''
    if include_min:
        _min = x.min(dim=dim1, keepdim=True)[0]
    else:
        _min = 0
    return atn * (x - _min) + _min

def _multiply_v2(x, atn, dim1=-1, dim2=-2, include_min=False):
    '''
    global_min
    '''
    if include_min:
        _min = (x.min(dim=dim1, keepdim=True)[0]).min(dim=dim2,keepdim=True)[0]
    else:
        _min = 0
    return atn * (x - _min) + _min

def min_max_norm(x):
    _min = (x.min(axis=-1,keepdims=True)[0]).min(axis=-2,keepdims=True)[0] #[b,1,1]
    _max = (x.max(axis=-1,keepdims=True)[0]).max(axis=-2,keepdims=True)[0] #[b,1,1]
    x = (x-_min)/(_max-_min)
    return x #[B,T,C+1]

def normalization(p):

    return (p - p.min(axis=0, keepdims=True)) / (p.max(axis=0, keepdims=True) - p.min(axis=0, keepdims=True) + 1e-8)

def softmax(pred):
    return np.exp(pred)/np.exp(pred).sum(axis=-1,keepdims=True)

@torch.no_grad()
def multiple_threshold_hamnet_v3(vid_name,args, elem,element_atn=None):
    logits_atn = _multiply(elem,element_atn,include_min=True) #[1,T,C+1]
    logits_norm = min_max_norm(elem) #[1,T,C+1]

    pred_vid_score = get_cls_score(logits_atn, rat=0.1)

    logits_atn_wobg = logits_atn[...,:-1] #[1,T,C] 
    logits_norm_wobg = logits_norm[...,:-1] #[1,T,C] 
    atn = element_atn #[1,T,1]

    pred = np.where(pred_vid_score >= args.cls_thresh)[0]

    # NOTE: threshold

    att_act_thresh=np.arange(args.att_thresh_params[0],args.att_thresh_params[1],args.att_thresh_params[2])
    cam_act_thresh=np.arange(args.cam_thresh_params[0],args.cam_thresh_params[1],args.cam_thresh_params[2])

    prediction = None
    if len(pred) == 0:
        pred = np.array([np.argmax(pred_vid_score)])
    logits_atn_pred = logits_atn_wobg[0].cpu().numpy()[:, pred] #[T,pred]
    logits_norm_pred = logits_norm_wobg[0].cpu().numpy()[:, pred] #[T,pred]

    num_segments = logits_atn_pred.shape[0]
    logits_atn_pred = np.reshape(logits_atn_pred, (num_segments, -1, 1)) #[T,pred_C,1]
    logits_norm_pred = np.reshape(logits_norm_pred, (num_segments, -1, 1)) #[T,pred_C,1]
    atn_pred = atn[0].cpu().numpy()[:, [0]] #[T,1]
    atn_pred = np.reshape(atn_pred, (num_segments, -1, 1)) #[T,1,1]


    proposal_dict = {}

    if args.test_proposal_mode=='both' or  args.test_proposal_mode=='att':
        # att based proposal generation
        for i in range(len(att_act_thresh)):
            logits_atn_pred_temp = logits_atn_pred.copy()
            logits_norm_pred_temp = logits_norm_pred.copy()
            atn_pred_temp = atn_pred.copy()

            seg_list = []

            for c in range(len(pred)):
                pos = np.where(atn_pred_temp[:, 0, 0] > att_act_thresh[i])
                seg_list.append(pos)

            proposals = utils.get_proposal_oic_2(seg_list,
                                                logits_atn_pred_temp,
                                                pred_vid_score,
                                                pred,
                                                args.scale,
                                                num_segments,
                                                args.feature_fps,
                                                num_segments,
                                                gamma=args.gamma_oic)

            for j in range(len(proposals)):
                try:
                    class_id = proposals[j][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[j]
                except IndexError:
                    logger.error(f"Index error")

    if args.test_proposal_mode=='both' or args.test_proposal_mode=='cam':
        # cam based proposal generation
        for i in range(len(cam_act_thresh)):
            logits_atn_pred_temp = logits_atn_pred.copy()
            logits_norm_pred_temp = logits_norm_pred.copy()
            atn_pred_temp = atn_pred.copy()

            seg_list = []

            for c in range(len(pred)):
                pos=np.where(logits_norm_pred_temp[:, c, 0] > cam_act_thresh[i])
                seg_list.append(pos)

            proposals = utils.get_proposal_oic_2(seg_list,
                                                logits_norm_pred_temp,
                                                pred_vid_score,
                                                pred,
                                                args.scale,
                                                num_segments,
                                                args.feature_fps,
                                                num_segments,
                                                gamma=args.gamma_oic)

            for j in range(len(proposals)):
                try:
                    class_id = proposals[j][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[j]
                except IndexError:
                    logger.error(f"Index error")


    final_proposals = []
    for class_id in proposal_dict.keys():
        if not args.without_nms:
            if args.nms_mode=="soft_nms":
                final_proposals.append(
                    utils.soft_nms_v2(proposal_dict[class_id], args.nms_thresh, sigma=0.3)
                )
            elif args.nms_mode=="nms":
                final_proposals.append(
                    utils.nms(proposal_dict[class_id], args.nms_thresh)
                )
            else:
                raise ValueError
        else:
            final_proposals.append(
                proposal_dict[class_id]
            )

    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end, c_score, c_pred])

    segment_predict = np.array(segment_predict)

    if args.feature_type=='I3D':
        factor=25.0/16.0
    elif args.feature_type=='UNT':
        factor=10.0/4.0
    else:
        factor=30.0/15.0

    if 'Thumos' in args.dataset_name:
        segment_predict = filter_segments(segment_predict, vid_name.decode(),factor)

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction

@torch.no_grad()
def multiple_threshold_hamnet_ant(vid_name,args, elem,element_atn=None):
    element_logits = elem * element_atn
    # element_logits = (elem-torch.min(elem,dim=-1,keepdim=True)[0]) * element_atn+torch.min(elem,dim=-1,keepdim=True)[0]

    pred_vid_score = get_cls_score(element_logits, rat=0.1) #[C]
    # score_np = pred_vid_score.copy()
    # score_np[score_np < 0.2] = 0
    # score_np[score_np >= 0.2] = 1
    cas_supp = element_logits[..., :-1]
    cas_supp_atn = element_atn

    # pred = np.where(pred_vid_score >= 0.1)[0]
    pred = np.where(pred_vid_score >= args.cls_thresh)[0]

    # NOTE: threshold

    min_act=min(cas_supp_atn.tolist())
    max_act=max(cas_supp_atn.tolist())
    # att_act_thresh = np.linspace(min_act, max_act, 10)[1:-1]
    # att_act_thresh = np.linspace(0.1,0.9,10)

    att_act_thresh=np.linspace(args.att_thresh_params[0],args.att_thresh_params[1],int(args.att_thresh_params[2]))

    prediction = None
    if len(pred) == 0:
        pred = np.array([np.argmax(pred_vid_score)])
    cas_pred = cas_supp[0].cpu().numpy()[:, pred] #[T,pred]
    num_segments = cas_pred.shape[0]
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1)) #[T,pred,1]

    cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]

    cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))#[T,1,1]

    proposal_dict = {}

    for i in range(len(att_act_thresh)):
        cas_temp = cas_pred.copy()
        cas_temp_atn = cas_pred_atn.copy()

        seg_list = []

        for c in range(len(pred)):
            pos = np.where(cas_temp_atn[:, 0, 0] > att_act_thresh[i])
            seg_list.append(pos) #[C,pos_num]

        proposals = utils.get_proposal_oic_2(seg_list,
                                             cas_temp,
                                             pred_vid_score,
                                             pred,
                                             args.scale,
                                             num_segments,
                                             args.feature_fps,
                                             num_segments,
                                             gamma=args.gamma_oic)

        for j in range(len(proposals)):
            try:
                class_id = proposals[j][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += proposals[j]
            except IndexError:
                logger.error(f"Index error")

    final_proposals = []
    for class_id in proposal_dict.keys():
        if args.nms_mode=="soft_nms":
            final_proposals.append(
                utils.soft_nms_v2(proposal_dict[class_id], args.nms_thresh, sigma=0.3)
            )
        elif args.nms_mode=="nms":
            final_proposals.append(
                utils.nms(proposal_dict[class_id], args.nms_thresh)
            )
        else:
            raise ValueError
        # final_proposals.append(
        #     utils.soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))
    # self.final_res["results"][vid_name[0]] = utils.result2json(
    # final_proposals, class_dict)

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    # [c_pred[i], c_score, t_start, t_end]
    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end, c_score, c_pred])

    segment_predict = np.array(segment_predict)

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        if 'ActivityNet1.3'==args.dataset_name:
            video_lst.append(vid_name)
        else:
            video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction


def normal_threshold(vid_name,prediction):


    # _data, _label, _, vid_name, vid_num_seg = inputs
    # attn,att_logits,feat_embed, bag_logit, prediction = outputs
    # attn,feat_embed, bag_logit, prediction = outputs
    prediction = prediction[0].cpu().numpy()
    # process the predictions such that classes having greater than a certain threshold are detected only
    prediction_mod = []
    prediction, c_s = _get_vid_score(prediction)
    if args is None:
        thres = 0.5
    else:
        thres = 1 - args.thres

    softmaxed_c_s=np.exp(c_s)/np.sum(np.exp(c_s))
    c_set = []
    for c in range(c_s.shape[0]):
        if np.max(softmaxed_c_s)>=0.15:
            if softmaxed_c_s[c]<0.15:
                continue
            else:
                c_set.append(c)
        else:
            if c != np.argsort(c_s,axis=-1)[-1]:
                continue
            else:
                c_set.append(c)
    # for c in np.argsort(c_s,axis=-1):
    #     if softmaxed_c_s[c]>=0.15:
    #         c_set.append(c)
    # if len(c_set)==0:
    #     c_set.append(np.argsort(c_s,axis=-1)[-1])
    
    segment_predict = []
    for c in c_set:
        tmp=prediction[:,c] if 'Thumos' in args.dataset_name else smooth(prediction[:,c])
        threshold=np.mean(tmp)
        vid_pred = np.concatenate(
            [np.zeros(1), (tmp > threshold).astype("float32"), np.zeros(1)], axis=0
        )
        vid_pred_diff = [
            vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
        ]
        s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]  #the start point set
        e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]  #the end point set
        for j in range(len(s)):
            if e[j] - s[j] >= 1:
                seg=tmp[s[j] : e[j]]
                segment_predict.append(
                    [s[j], e[j], np.max(seg)+0.7*c_s[c],c]
                    # [c,np.max(seg)+0.7*c_s[c],s[j], e[j]]
                )
    if args.feature_type=='I3D':
        factor=25.0/16.0
    elif args.feature_type=='UNT':
        factor=10.0/4.0
    else:
        factor=30.0/15.0


    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode(),factor)
    
   
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction

def normal_threshold_ant(vid_name,prediction):


    # _data, _label, _, vid_name, vid_num_seg = inputs
    # attn,att_logits,feat_embed, bag_logit, prediction = outputs
    # attn,feat_embed, bag_logit, prediction = outputs
    prediction = prediction[0].cpu().numpy()
    # process the predictions such that classes having greater than a certain threshold are detected only
    prediction_mod = []
    prediction, c_s = _get_vid_score(prediction)
    if args is None:
        thres = 0.5
    else:
        thres = 1 - args.thres

    softmaxed_c_s=np.exp(c_s)/np.sum(np.exp(c_s))
    c_set = []
    for c in range(c_s.shape[0]):
        if np.max(softmaxed_c_s)>=0.15:
            if softmaxed_c_s[c]<0.15:
                continue
            else:
                c_set.append(c)
        else:
            if c != np.argsort(c_s,axis=-1)[-1]:
                continue
            else:
                c_set.append(c)
    # for c in np.argsort(c_s,axis=-1):
    #     if softmaxed_c_s[c]>=0.15:
    #         c_set.append(c)
    # if len(c_set)==0:
    #     c_set.append(np.argsort(c_s,axis=-1)[-1])
    
    segment_predict = []
    for c in c_set:
        tmp=prediction[:,c] if 'Thumos' in args.dataset_name else smooth(prediction[:,c])
        threshold=np.mean(tmp) if 'Thumos' in args.dataset_name else np.max(tmp) - (np.max(tmp) - np.min(tmp)) * thres
        vid_pred = np.concatenate(
            [np.zeros(1), (tmp > threshold).astype("float32"), np.zeros(1)], axis=0
        )
        vid_pred_diff = [
            vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
        ]
        s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]  #the start point set
        e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]  #the end point set
        for j in range(len(s)):
            if e[j] - s[j] >= 1:
                seg=tmp[s[j] : e[j]]
                segment_predict.append(
                    [s[j], e[j], np.max(seg)+0.7*c_s[c],c]
                    # [c,np.max(seg)+0.7*c_s[c],s[j], e[j]]
                )
    if args.feature_type=='I3D':
        factor=25.0/16.0
    elif args.feature_type=='UNT':
        factor=10.0/4.0
    else:
        factor=30.0/15.0


    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode(),factor)
    
   
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction


def normal_threshold_IOC(vid_name,prediction):


    # _data, _label, _, vid_name, vid_num_seg = inputs
    # attn,att_logits,feat_embed, bag_logit, prediction = outputs
    # attn,feat_embed, bag_logit, prediction = outputs
    prediction = prediction[0].cpu().numpy()
    # process the predictions such that classes having greater than a certain threshold are detected only
    prediction_mod = []
    prediction, c_s = _get_vid_score(prediction)
    if args is None:
        thres = 0.5
    else:
        thres = 1 - args.thres

    softmaxed_c_s=np.exp(c_s)/np.sum(np.exp(c_s))
    c_set = []
    for c in range(c_s.shape[0]):
        if np.max(softmaxed_c_s)>=0.15:
            if softmaxed_c_s[c]<0.15:
                continue
            else:
                c_set.append(c)
        else:
            if c != np.argsort(c_s,axis=-1)[-1]:
                continue
            else:
                c_set.append(c)
    # for c in np.argsort(c_s,axis=-1):
    #     if softmaxed_c_s[c]>=0.15:
    #         c_set.append(c)
    # if len(c_set)==0:
    #     c_set.append(np.argsort(c_s,axis=-1)[-1])
    
    segment_predict = []
    for c in c_set:
        # prediction[:,c] = __vector_minmax_norm(prediction[:,c])
        tmp=prediction[:,c] if 'Thumos' in args.dataset_name else smooth(prediction[:,c])
        threshold=np.mean(tmp) if 'Thumos' in args.dataset_name else np.max(tmp) - (np.max(tmp) - np.min(tmp)) * thres
        vid_pred = np.concatenate(
            [np.zeros(1), (tmp > threshold).astype("float32"), np.zeros(1)], axis=0
        )
        vid_pred_diff = [
            vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
        ]
        s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]  #the start point set
        e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]  #the end point set
        for j in range(len(s)):
            if e[j] - s[j] >= 1:
                seg=tmp[s[j] : e[j]]
                inner_score = np.mean(seg)

                len_proposal = len(seg)

                outer_s = max(0, int(s[j] - 0.25 * len_proposal))
                outer_e = min(int(prediction.shape[0] - 1), int(e[j]+ 0.25 * len_proposal))

                outer_temp_list = list(range(outer_s, s[j])) + list(range(int(e[j] + 1), outer_e + 1))
                
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(tmp[outer_temp_list])
                
                c_score = inner_score - outer_score + 0.7*c_s[c]

                segment_predict.append(
                    [s[j], e[j], c_score,c]
                    # [c,np.max(seg)+0.7*c_s[c],s[j], e[j]]
                )
    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())
    
   
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction

