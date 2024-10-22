import pandas as pd
import pdb
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter

def smooth(v):
   # return v
    l = min(200, len(v))
    l = l - (1-l%2)
    if len(v) <= 3:
        return v
    return savgol_filter(v, l, 2) #savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)

def task1():
    attns = np.load('temp/attn.npy',allow_pickle=True).tolist()
    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('/home/share/fating/ProcessDataset/Thumos14/Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
    # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()
        if not 'test' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        
        cpred_sigmoid,cpred_logit = attns[videoname[i]]
        cpred_sigmoid = cpred_sigmoid.view(-1,).cpu().numpy()
        cpred_logit = cpred_logit.view(-1,).cpu().numpy()
        # pdb.set_trace()
        x = list(range(len(cpred_sigmoid)))
        
        # cgt = gt[gt.label==c]
        gtline = np.array([np.min(cpred_sigmoid) for i in range(len(cpred_sigmoid))])
        for idx, this_pred in gt.iterrows():
            gtline[this_pred['t-start']:this_pred['t-end']] = np.max(cpred_sigmoid)
        # if len(cgt) == 0:
            # continue
        plt.plot(x, cpred_sigmoid, linewidth=1)
        # plt.plot(x, scpred, linewidth=1)
        # threshold = np.max(scpred) - (np.max(scpred) - np.min(scpred)) * 0.05

        # threshold = np.mean(cpred)
        plt.plot(x, gtline, linewidth=1)
        # plt.plot(x, [threshold for i in range(len(cpred))], linewidth=1)
        
        plt.xlabel("clips", fontsize=12)
        plt.ylabel("scores", fontsize=12)
        plt.tick_params(axis='both',labelsize=10)
        plt.title("{} prediction".format(vn),fontsize=20)
        plt.savefig('temp/score_visual/{}.pdf'.format(vn))
        print('save temp/score_visual/{}.pdf'.format(vn))
        plt.clf()

    print(len(prediction),count)
if __name__ == '__main__':
    task1()
