import pandas as pd
import pdb
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter
import sys
sys.path.append('..')
from eval.classificationMAP import getClassificationMAP

def smooth(v):
   # return v
    l = min(200, len(v))
    l = l - (1-l%2)
    if len(v) <= 3:
        return v
    return savgol_filter(v, l, 2) #savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)

def task1():
    prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('temp/video_list.npy',allow_pickle=True).tolist()
    atten=np.load('temp/atten.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
        vn = videoname[i]
        pred = prediction[vn].squeeze(0)
        gt = groundTruthGroupByVideoId.get_group(vn.decode())
        if pred.shape[0]<10:
            count+=1
            continue
        for c in range(pred.shape[1]):
            cpred = pred[:,c]
            scpred = smooth(cpred)
            x = list(range(len(cpred)))
            cgt = gt[gt.label==c]
            gtline = np.array([np.min(cpred) for i in range(len(cpred))])
            for idx, this_pred in cgt.iterrows():
                gtline[this_pred['t-start']:this_pred['t-end']] = np.max(cpred)
            if len(cgt) == 0:
                continue
            plt.plot(x, cpred, linewidth=1)
            plt.plot(x, scpred, linewidth=1)
            threshold = np.max(scpred) - (np.max(scpred) - np.min(scpred)) * 0.05

            threshold = np.mean(cpred)
            plt.plot(x, gtline, linewidth=1)
            plt.plot(x, [threshold for i in range(len(cpred))], linewidth=1)
            
            plt.xlabel("clips", fontsize=12)
            plt.ylabel("scores", fontsize=12)
            plt.tick_params(axis='both',labelsize=10)
            # vn=vn.decode()
            plt.title("{} prediction".format(vn),fontsize=20)
            plt.savefig('temp/score_visual/{}_{}.pdf'.format(vn,c))
            print('save temp/score_visual/{}_{}.pdf'.format(vn,c))

            plt.clf()



    print(len(prediction),count)


def task2():
    # prediction = np.load('temp/prediction.npy', allow_pickle=True).tolist()
    videoname = np.load('temp/video_list.npy', allow_pickle=True).tolist()
    atten = np.load('temp/atten_wo_neg.npy', allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0

    import os
    if not os.path.exists('temp/score_visual_wo_neg'):
        os.mkdir('temp/score_visual_wo_neg')

    for i in range(len(videoname)):
        vn = videoname[i]
        # pred = prediction[vn].squeeze(0)
        att=atten[vn].squeeze(0)
        gt = groundTruthGroupByVideoId.get_group(vn.decode())
        if att.shape[0] < 10:
            count += 1
            continue

        x = list(range(att.shape[0]))
        gtline = np.zeros(att.shape[0])

        for idx, this_pred in gt.iterrows():
            gtline[this_pred['t-start']:this_pred['t-end']] = 1

        plt.plot(x,gtline,linewidth=1)
        plt.plot(x,att.squeeze(-1),linewidth=1)

        plt.xlabel("clips", fontsize=12)
        plt.ylabel("scores", fontsize=12)
        plt.tick_params(axis='both', labelsize=10)
        # vn=vn.decode()
        plt.title("{} prediction".format(vn), fontsize=20)
        plt.savefig('temp/score_visual_wo_neg/{}.pdf'.format(vn))
        print('save temp/score_visual_wo_neg/{}.pdf'.format(vn))

        plt.clf()

import pylab
import csv
def task3():
    videoname = np.load('temp/video_list.npy', allow_pickle=True).tolist()
    atten = np.load('temp/atten.npy', allow_pickle=True).tolist()
    atten_wo_neg=np.load('temp/atten_wo_neg.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    vn=b'video_test_0000844'
    # pred = prediction[vn].squeeze(0)
    att = atten[vn].squeeze()
    att_wo_neg=atten_wo_neg[vn].squeeze()
    gt = groundTruthGroupByVideoId.get_group(vn.decode())

    gtline = np.zeros(att.shape[0])

    for idx, this_pred in gt.iterrows():
        gtline[this_pred['t-start']:this_pred['t-end']] = 1

    with open('temp/{}.csv'.format(vn),'w')as f:
        for (a,awn,g) in zip(att,att_wo_neg,gtline):
            f.write('{},{},{}\n'.format(a,awn,g))
    print('finshed {}!'.format(vn))

def calSnippetClassificationMAP(pred,vns,labels):
    preds=[]
    for i, vn in enumerate(vns):
        preds.append(pred[vn].squeeze())

    preds=np.concatenate(preds,axis=0)
    labels=np.concatenate(labels,axis=0)

    return getClassificationMAP(preds,labels)


from sklearn.metrics import accuracy_score
def calSnippetClassificationAcc(pred,vns,labels):
    preds=[]
    for i, vn in enumerate(vns):
        preds.append(pred[vn].squeeze())

    preds=np.concatenate(preds,axis=0)
    labels=np.concatenate(labels,axis=0)

    args_preds=np.argmax(preds,axis=-1)
    args_labels=np.argmax(labels,axis=-1)

    return accuracy_score(args_labels,args_preds)

# snippet-level classification
def task4(n_class=20):
    videoname = np.load('../temp/video_list.npy', allow_pickle=True).tolist()
    pred = np.load('../temp/pred.npy', allow_pickle=True).tolist()
    pred_MIL = np.load('../temp/pred_MIL_wo_SAL.npy', allow_pickle=True).tolist()
    # pred_wo_neg=np.load('../temp/pred_wo_neg.npy',allow_pickle=True).tolist()
    # pred_wo_con=np.load('../temp/pred_wo_con.npy',allow_pickle=True).tolist()
    # pred_wo_pt=np.load('../temp/pred_wo_pt.npy',allow_pickle=True).tolist()
    # pred_wo_sal=np.load('../temp/pred_wo_sal.npy',allow_pickle=True).tolist()

    groundTruth = pd.read_csv('../temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')

    labels=[]
    # to construct the snippet-level groundtruth
    for vn in videoname:
        p=pred[vn].squeeze()
        # [t,c+1]
        snippet_label=np.zeros_like(p)
        snippet_label[:,-1]=1
        gt=groundTruthGroupByVideoId.get_group(vn.decode())
        for idx,this_pred in gt.iterrows():
            snippet_label[this_pred['t-start']:this_pred["t-end"],this_pred['label']]=1
            snippet_label[this_pred['t-start']:this_pred["t-end"],-1]=0
        labels.append(snippet_label)

    # pred_cmap=calSnippetClassificationAcc(pred,videoname,labels)
    # pred_wo_neg_cmap=calSnippetClassificationAcc(pred_wo_neg,videoname,labels)
    # pred_wo_con_cmap=calSnippetClassificationAcc(pred_wo_con,videoname,labels)
    # pred_wo_pt_cmap=calSnippetClassificationAcc(pred_wo_pt,videoname,labels)
    # pred_wo_sal_cmap=calSnippetClassificationAcc(pred_wo_sal,videoname,labels)
    pred_mil_cmap=calSnippetClassificationAcc(pred_MIL,videoname,labels)

    print('MIL+SAL {:.4f}'.format(pred_mil_cmap))
    # print('Full {:.4f}'.format(pred_cmap))
    # print('wo SAL {:.4f}'.format(pred_wo_sal_cmap))
    # print('wo Neg {:.4f}'.format(pred_wo_neg_cmap))
    # print('wo Con {:.4f}'.format(pred_wo_con_cmap))
    # print('wo PT {:.4f}'.format(pred_wo_pt_cmap))
# small picture
def task5():
    # prediction = np.load('temp/prediction.npy', allow_pickle=True).tolist()
    videoname = np.load('../temp/video_list.npy', allow_pickle=True).tolist()
    pred = np.load('../temp/prediction_wo_neg.npy', allow_pickle=True).tolist()
    groundTruth = pd.read_csv('../temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0

    import os
    if not os.path.exists('../temp/score_visual_cls_wo_neg'):
        os.mkdir('../temp/score_visual_cls_wo_neg')

    for i in range(len(videoname)):
        vn = videoname[i]
        # pred = prediction[vn].squeeze(0)
        p=pred[vn].squeeze(0)
        gt = groundTruthGroupByVideoId.get_group(vn.decode())
        if p.shape[0] < 10:
            count += 1
            continue

        x = list(range(p.shape[0]))
        gtline = np.zeros(p.shape[0])

        gt_labels=[]

        for idx, this_pred in gt.iterrows():
            gtline[this_pred['t-start']:this_pred['t-end']] = 1
            gt_labels.append(this_pred['label'])

        # p=np.exp(p)/np.exp(p).sum(axis=-1,keepdims=True)
        p=(p-p.min(axis=0,keepdims=True))/(p.max(axis=0,keepdims=True)-p.min(axis=0,keepdims=True)+1e-8)
        gt_labels=list(set(gt_labels))
        for label in gt_labels:
            plt.plot(x,gtline,linewidth=1)
            plt.plot(x,p[:,label],linewidth=1)
            plt.xlabel("clips", fontsize=12)
            plt.ylabel("scores", fontsize=12)
            plt.tick_params(axis='both', labelsize=10)
            # vn=vn.decode()
            plt.title("{} prediction".format(vn), fontsize=20)
            plt.savefig('../temp/score_visual_cls_wo_neg/{}_{}.pdf'.format(vn,label))
            print('save temp/score_visual_cls_wo_neg/{}_{}.pdf'.format(vn,label))
            plt.clf()

def normalization(p):
    return (p - p.min(axis=0, keepdims=True)) / (p.max(axis=0, keepdims=True) - p.min(axis=0, keepdims=True) + 1e-8)

# output the score file to csv file
def task6():
    videoname = np.load('../temp/video_list.npy', allow_pickle=True).tolist()
    pred = np.load('../temp/prediction.npy', allow_pickle=True).tolist()
    pred_MIL=np.load('../temp/prediction_MIL_wo_SAL.npy',allow_pickle=True).tolist()
    pred_MIL_SAL = np.load('../temp/prediction_MIL.npy', allow_pickle=True).tolist()
    pred_wo_neg=np.load('../temp/prediction_wo_neg.npy',allow_pickle=True).tolist()
    pred_wo_con=np.load('../temp/prediction_wo_con.npy',allow_pickle=True).tolist()
    pred_wo_pt=np.load('../temp/prediction_wo_pt.npy',allow_pickle=True).tolist()
    pred_wo_sal=np.load('../temp/prediction_wo_sal.npy',allow_pickle=True).tolist()
    pred_cls_vcl=np.load('../temp/prediction_Cls_VCL.npy',allow_pickle=True).tolist()
    pred_cls_scl=np.load('../temp/prediction_Cls_SCL.npy',allow_pickle=True).tolist()
    pred_cls_cll=np.load('../temp/prediction_Cls_CLL.npy',allow_pickle=True).tolist()

    groundTruth = pd.read_csv('../temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0

    vn=b'video_test_0000242'
    label=13
    p=normalization(pred[vn].squeeze()[:,label])
    p_mil=normalization(pred_MIL[vn].squeeze()[:,label])
    p_mil_sal=normalization(pred_MIL_SAL[vn].squeeze()[:,label])
    p_wo_neg=normalization(pred_wo_neg[vn].squeeze()[:,label])
    p_wo_con=normalization(pred_wo_con[vn].squeeze()[:,label])
    p_wo_pt=normalization(pred_wo_pt[vn].squeeze()[:,label])
    p_wo_sal=normalization(pred_wo_sal[vn].squeeze()[:,label])
    p_cls_vcl=normalization(pred_cls_vcl[vn].squeeze()[:,label])
    p_cls_scl=normalization(pred_cls_scl[vn].squeeze()[:,label])
    p_cls_cll=normalization(pred_cls_cll[vn].squeeze()[:,label])


    gt = groundTruthGroupByVideoId.get_group(vn.decode())

    gtline = np.zeros(p.shape[0])
    for idx, this_pred in gt.iterrows():
        gtline[this_pred['t-start']:this_pred['t-end']] = 1

    p=normalization(p)

    with open('../temp/{}_multiple.csv'.format(vn),'w')as file:
        for (a,b,c,d,e,f,g,h,i,j,k) in zip(p,p_mil,p_mil_sal,p_wo_neg,p_wo_con,p_wo_pt,p_wo_sal,p_cls_vcl,p_cls_scl,p_cls_cll,gtline):
            file.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(a,b,c,d,e,f,g,h,i,j,k))
    print('finshed {}!'.format(vn))


def task7():
    videoname = np.load('../temp/video_list.npy', allow_pickle=True).tolist()
    att=np.load('../temp/atten.npy',allow_pickle=True).tolist()
    att_wo_neg=np.load('../temp/atten_wo_neg.npy',allow_pickle=True).tolist()
    pred = np.load('../temp/prediction.npy', allow_pickle=True).tolist()
    pred_wo_neg=np.load('../temp/prediction_wo_neg.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('../temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0

    vn=b'video_test_0000785'
    label=4
    p=normalization(pred[vn].squeeze()[:,label])
    p_wo_neg=normalization(pred_wo_neg[vn].squeeze()[:,label])
    a=normalization(att[vn].squeeze())
    a_wo_neg=normalization(att_wo_neg[vn].squeeze())
    gt = groundTruthGroupByVideoId.get_group(vn.decode())

    gtline = np.zeros(p.shape[0])
    for idx, this_pred in gt.iterrows():
        gtline[this_pred['t-start']:this_pred['t-end']] = 1

    p=normalization(p)

    with open('../temp/{}_multiple_supp.csv'.format(vn),'w')as file:
        for (a,b,c,d,e) in zip(p,p_wo_neg,a,a_wo_neg,gtline):
            file.write('{},{},{},{},{}\n'.format(a,b,c,d,e))
    print('finshed {}!'.format(vn))

if __name__ == '__main__':
    task6()
