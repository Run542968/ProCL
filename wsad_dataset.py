from __future__ import print_function

import h5py
import numpy as np
import torch
import json
import utils.wsad_utils as utils
import random
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # solve the "OSError: Unable to open file (unable to lock file, errno = 37, error message = 'No locks available')"
import copy
import options
import pdb

class SampleDataset:
    def __init__(self, args, mode="both",sampling='random'):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling=sampling
        self.num_segments_list = args.max_seqlen_list
        self.num_segments_single = args.max_seqlen_single 
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(args.path_dataset,self.dataset_name + "-{}-JOINTFeatures.npy".format(args.feature_type))
        self.path_to_annotations = os.path.join(args.path_dataset,self.dataset_name + "-Annotations/")
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )
        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        # Specific to Thumos14

        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.batch_size = args.batch_size
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]
        try:
            ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]
        except:
            ambilist = []
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode("utf-8") == "validation":  # Specific to Thumos14
                self.trainidx.append(i)
            elif s.decode("utf-8") == "test":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self,is_single=True,is_training=True):
        if is_training:
            
            if is_single: # pretrain num segment
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                feat = []
                for i in idx:
                    ifeat = self.features[i]
                    if self.sampling == 'random':
                        sample_idx = self.random_perturb(self.num_segments_single,ifeat.shape[0])
                    elif self.sampling == 'uniform':
                        sample_idx = self.uniform_sampling(self.num_segments_single,ifeat.shape[0])
                    elif self.sampling == "all":
                        sample_idx = np.arange(ifeat.shape[0])
                    else:
                        raise AssertionError('Not supported sampling !')
                    ifeat = ifeat[sample_idx]
                    feat.append(ifeat)
                feat = np.array(feat)
                labels = np.array([self.labels_multihot[i] for i in idx])
                if self.mode == "rgb":
                    feat = feat[..., : self.feature_size]
                elif self.mode == "flow":
                    feat = feat[..., self.feature_size :]
                
                return feat, labels,rand_sampleid,idx

            else: # multi-scale num segments
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                ms_feat = []
                for segments in self.num_segments_list:
                    feat = []
                    for i in idx:
                        ifeat = self.features[i]
                        if self.sampling == 'random':
                            sample_idx = self.random_perturb(segments,ifeat.shape[0])
                        elif self.sampling == 'uniform':
                            sample_idx = self.uniform_sampling(segments,ifeat.shape[0])
                        elif self.sampling == "all":
                            sample_idx = np.arange(ifeat.shape[0])
                        else:
                            raise AssertionError('Not supported sampling !')
                        ifeat = ifeat[sample_idx]
                        feat.append(ifeat)
                    feat = np.array(feat)
                    labels = np.array([self.labels_multihot[i] for i in idx])
                    if self.mode == "rgb":
                        feat = feat[..., : self.feature_size]
                    elif self.mode == "flow":
                        feat = feat[..., self.feature_size :]
                    
                    ms_feat.append(feat)

                return ms_feat, labels,rand_sampleid,idx

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            # feat = utils.process_feat(feat, normalize=self.normalize)
            # feature = feature[sample_idx]
            vn = self.videonames[self.testidx[self.currenttestidx]]
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            # return feat, np.array(labs),vn, done
            output_dict = {
                'feat': feat,
                'lab': np.array(labs),
                'vn': vn,
                'done': done,
                'mode': 'All'
                # 'sample_idx': sample_idx,
            }

            return output_dict  # feat, np.array(labs),vn, done

    def random_avg(self, x, segm=None):
        if len(x) < self.num_segments:
            ind = self.random_perturb(len(x))
            x_n = x[ind]
            segm = segm[ind] if segm is not None else None
            return x_n, segm
        else:
            inds = np.array_split(np.arange(len(x)), self.num_segments)
            x_n = np.zeros((self.num_segments, x.shape[-1])).astype(x.dtype)
            segm_n = np.zeros(
                (self.num_segments, segm.shape[-1])).astype(x.dtype)
            for i, ind in enumerate(inds):
                x_n[i] = np.mean(x[ind], axis=0)
                if segm is not None:
                    segm_n[i] = segm[(ind[0] + ind[-1]) // 2]
            return x_n, segm_n if segm is not None else None

    def random_pad(self, x, segm=None):
        length = self.num_segments
        if x.shape[0] > length:
            strt = np.random.randint(0, x.shape[0] - length)
            x_ret = x[strt:strt + length]
            if segm is not None:
                segm = segm[strt:strt + length]
                return x_ret, segm
        elif x.shape[0] == length:
            return x, segm
        else:
            pad_len = length - x.shape[0]
            x_ret = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            if segm is not None:
                segm = np.pad(segm, ((0, pad_len), (0, 0)), mode='constant')
            return x_ret, segm

    def random_perturb(self, num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        for i in range(num_segments):
            if i < num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        samples = np.floor(samples)
        return samples.astype(int)

class AntSampleDataset:
    def __init__(self, args, mode="both",sampling='random'):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling=sampling
        self.num_segments_list = args.max_seqlen_list
        self.num_segments_single = args.max_seqlen_single 
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(args.path_dataset,self.dataset_name + "-I3D-JOINTFeatures.npy")
        self.path_to_annotations = os.path.join(args.path_dataset,self.dataset_name + "-Annotations/")
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )
        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )

        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.duration=np.load(self.path_to_annotations+'duration.npy',allow_pickle=True)

        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen_single
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]
        try:
            ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]
        except:
            ambilist = []
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024
        self.filter()
    def filter(self):
        new_testidx = []
        for idx in self.testidx:
            feat = self.features[idx]
            if len(feat)>10:
                new_testidx.append(idx)
        self.testidx = new_testidx

        new_trainidx = []
        for idx in self.trainidx:
            feat = self.features[idx]
            if len(feat)>10:
                new_trainidx.append(idx)
        self.trainidx = new_trainidx
    
    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode("utf-8") == "training":  # Specific to Thumos14
                self.trainidx.append(i)
            elif s.decode("utf-8") == "validation":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                if self.features[i].sum() == 0:
                    continue
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self,is_single=True,is_training=True):
        if is_training:

            if is_single:
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                feat = []
                for i in idx:
                    ifeat = self.features[i]

                    if self.sampling == 'random':
                        sample_idx = self.random_perturb(self.num_segments_single,ifeat.shape[0])
                        ifeat = ifeat[sample_idx]
                    elif self.sampling == 'uniform':
                        sample_idx = self.uniform_sampling(self.num_segments_single,ifeat.shape[0])
                        ifeat = ifeat[sample_idx]
                    elif self.sampling == "all":
                        sample_idx = np.arange(ifeat.shape[0])
                        ifeat = ifeat[sample_idx]
                    elif self.sampling =='average':
                        ifeat=utils.average_to_fixed_length(ifeat,self.num_segments_single)
                    else:
                        raise AssertionError('Not supported sampling !')

                    feat.append(ifeat)

                feat = np.array(feat)
                labels = np.array([self.labels_multihot[i] for i in idx])
                if self.mode == "rgb":
                    feat = feat[..., : self.feature_size]
                elif self.mode == "flow":
                    feat = feat[..., self.feature_size :]

                return feat, labels,rand_sampleid,idx
            else:
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                ms_feat = []
                for segments in self.num_segments_list:
                    feat = []
                    for i in idx:
                        ifeat = self.features[i]

                        if self.sampling == 'random':
                            sample_idx = self.random_perturb(segments,ifeat.shape[0])
                            ifeat = ifeat[sample_idx]
                        elif self.sampling == 'uniform':
                            sample_idx = self.uniform_sampling(segments,ifeat.shape[0])
                            ifeat = ifeat[sample_idx]
                        elif self.sampling == "all":
                            sample_idx = np.arange(ifeat.shape[0])
                            ifeat = ifeat[sample_idx]
                        elif self.sampling =='average':
                            ifeat=utils.average_to_fixed_length(ifeat,segments)
                        else:
                            raise AssertionError('Not supported sampling !')

                        feat.append(ifeat)

                    feat = np.array(feat)
                    labels = np.array([self.labels_multihot[i] for i in idx])
                    if self.mode == "rgb":
                        feat = feat[..., : self.feature_size]
                    elif self.mode == "flow":
                        feat = feat[..., self.feature_size :]

                    ms_feat.append(feat)
                return ms_feat, labels,rand_sampleid,idx

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            dur=self.duration[self.testidx[self.currenttestidx]]
            # feat = utils.process_feat(feat, normalize=self.normalize)
            # feature = feature[sample_idx]
            vn = self.videonames[self.testidx[self.currenttestidx]]
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            output_dict={
                'feat':feat,
                'lab':np.array(labs),
                'vn':vn,
                'done':done,
                'mode':'All',
                'dur': dur
                # 'sample_idx':sample_idx,
            }

            return output_dict #feat, np.array(labs),vn, done    def random_avg(self, x, segm=None):

    def random_avg(self, x, segm=None):
        if len(x) < self.num_segments:
            ind = self.random_perturb(len(x))
            x_n = x[ind]
            segm = segm[ind] if segm is not None else None
            return x_n, segm
        else:
            inds = np.array_split(np.arange(len(x)), self.num_segments)
            x_n = np.zeros((self.num_segments, x.shape[-1])).astype(x.dtype)
            segm_n = np.zeros(
                (self.num_segments, segm.shape[-1])).astype(x.dtype)
            for i, ind in enumerate(inds):
                x_n[i] = np.mean(x[ind], axis=0)
                if segm is not None:
                    segm_n[i] = segm[(ind[0] + ind[-1]) // 2]
            return x_n, segm_n if segm is not None else None

    def random_pad(self, x, segm=None):
        length = self.num_segments
        if x.shape[0] > length:
            strt = np.random.randint(0, x.shape[0] - length)
            x_ret = x[strt:strt + length]
            if segm is not None:
                segm = segm[strt:strt + length]
                return x_ret, segm
        elif x.shape[0] == length:
            return x, segm
        else:
            pad_len = length - x.shape[0]
            x_ret = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            if segm is not None:
                segm = np.pad(segm, ((0, pad_len), (0, 0)), mode='constant')
            return x_ret, segm

    def random_perturb(self,num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        for i in range(num_segments):
            if i < num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        samples = np.floor(samples)
        return samples.astype(int)

class Ant13_SampleDataset:
    def __init__(self, args, mode="both",sampling='random'):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling=sampling
        self.num_segments_list = args.max_seqlen_list
        self.num_segments_single = args.max_seqlen_single # pretrain num segment
        self.feature_size = args.feature_size
        # self.path_to_features = os.path.join(args.path_dataset,self.dataset_name + "-I3D-JOINTFeatures.npy")
        self.train_features,self.test_features={},{}
        self.train_features['rgb']=self.preload_features(h5py.File(args.path_dataset+'feature_rgb_train.h5','r'))
        self.test_features['rgb']=self.preload_features(h5py.File(args.path_dataset+'feature_rgb_val.h5','r'))
        self.train_features['flow']=self.preload_features(h5py.File(args.path_dataset+'feature_flow_train.h5','r'))
        self.test_features['flow']=self.preload_features(h5py.File(args.path_dataset+'feature_flow_val.h5','r'))

        self.path_to_annotations = os.path.join(args.path_dataset,self.dataset_name + "-Annotations/")
        # self.features = np.load(
        #     self.path_to_features, encoding="bytes", allow_pickle=True
        # )
        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        # Specific to Thumos14

        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.durations=np.load(self.path_to_annotations+"duration.npy",allow_pickle=True)
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen_single
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]
        try:
            ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]
        except:
            ambilist = []
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024
        self.filter()

    def filter(self):
        new_testidx = []
        for idx in self.testidx:
            feat = self.test_features['rgb'][self.videonames[idx]][:]
            if len(feat)>10:
                new_testidx.append(idx)
        self.testidx = new_testidx

        new_trainidx = []
        for idx in self.trainidx:
            feat = self.train_features['rgb'][self.videonames[idx]][:]
            if len(feat)>10:
                new_trainidx.append(idx)
        self.trainidx = new_trainidx

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s== "train":  # Specific to Thumos14
                self.trainidx.append(i)
            elif s == "val":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                if self.train_features['rgb'][self.videonames[i]].shape[0]==0:
                    continue
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def preload_features(self,feat_dict):
        feature_dict={}
        for key in feat_dict.keys():
            feature_dict[key]=feat_dict[key][:]
        return feature_dict

    def load_data(self,is_single=True,is_training=True):
        if is_training:

            if is_single:
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])


                feat = []
                for i in idx:
                    # ifeat = self.features[i]
                    ifeat=np.concatenate([self.train_features['rgb'][self.videonames[i]],
                                        self.train_features['flow'][self.videonames[i]]],axis=-1)

                    if self.sampling == 'random':
                        sample_idx = self.random_perturb(self.num_segments_single,ifeat.shape[0])
                        ifeat = ifeat[sample_idx]
                    elif self.sampling == 'uniform':
                        sample_idx = self.uniform_sampling(self.num_segments_single,ifeat.shape[0])
                        ifeat = ifeat[sample_idx]
                    elif self.sampling == "all":
                        sample_idx = np.arange(ifeat.shape[0])
                        ifeat = ifeat[sample_idx]
                    elif self.sampling =='average':
                        ifeat=utils.average_to_fixed_length(ifeat,self.num_segments_single)
                    else:
                        raise AssertionError('Not supported sampling !')

                    feat.append(ifeat)

                feat = np.array(feat)
                labels = np.array([self.labels_multihot[i] for i in idx])
                if self.mode == "rgb":
                    feat = feat[..., : self.feature_size]
                elif self.mode == "flow":
                    feat = feat[..., self.feature_size :]

                return feat, labels,rand_sampleid,idx
            else:
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                ms_feat = []
                for segments in self.num_segments_list:
                    feat = []
                    for i in idx:
                        # ifeat = self.features[i]
                        ifeat=np.concatenate([self.train_features['rgb'][self.videonames[i]],
                                            self.train_features['flow'][self.videonames[i]]],axis=-1)

                        if self.sampling == 'random':
                            sample_idx = self.random_perturb(segments,ifeat.shape[0])
                            ifeat = ifeat[sample_idx]
                        elif self.sampling == 'uniform':
                            sample_idx = self.uniform_sampling(segments,ifeat.shape[0])
                            ifeat = ifeat[sample_idx]
                        elif self.sampling == "all":
                            sample_idx = np.arange(ifeat.shape[0])
                            ifeat = ifeat[sample_idx]
                        elif self.sampling =='average':
                            ifeat=utils.average_to_fixed_length(ifeat,segments)
                        else:
                            raise AssertionError('Not supported sampling !')

                        feat.append(ifeat)

                    feat = np.array(feat)
                    labels = np.array([self.labels_multihot[i] for i in idx])
                    if self.mode == "rgb":
                        feat = feat[..., : self.feature_size]
                    elif self.mode == "flow":
                        feat = feat[..., self.feature_size :]

                    ms_feat.append(feat)
                return ms_feat, labels,rand_sampleid,idx

        else:

            idx=self.testidx[self.currenttestidx]
            labs = self.labels_multihot[idx]
            # feat = self.features[self.testidx[self.currenttestidx]]
            vn = self.videonames[idx]
            feat = np.concatenate([self.test_features['rgb'][vn],
                                    self.test_features['flow'][vn]], axis=-1)
            dur=self.durations[idx]
            # feat = utils.process_feat(feat, normalize=self.normalize)
            # feature = feature[sample_idx]
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            output_dict={
                'feat':feat,
                'lab':np.array(labs),
                'vn':vn,
                'done':done,
                'mode':'All',
                'dur':dur
                # 'sample_idx':sample_idx,
            }

            return output_dict #feat, np.array(labs),vn, done    def random_avg(self, x, segm=None):

    def verify_dataset(self):
        not_exist_vns=[]
        for i in self.testidx:
            vn=self.videonames[i]

            try:
                a=self.test_features['rgb'][vn]
                b=self.test_features['flow'][vn]
            except:
                not_exist_vns.append(vn)
        print(not_exist_vns)

    def random_avg(self, x, segm=None):
        if len(x) < self.num_segments:
            ind = self.random_perturb(len(x))
            x_n = x[ind]
            segm = segm[ind] if segm is not None else None
            return x_n, segm
        else:
            inds = np.array_split(np.arange(len(x)), self.num_segments)
            x_n = np.zeros((self.num_segments, x.shape[-1])).astype(x.dtype)
            segm_n = np.zeros(
                (self.num_segments, segm.shape[-1])).astype(x.dtype)
            for i, ind in enumerate(inds):
                x_n[i] = np.mean(x[ind], axis=0)
                if segm is not None:
                    segm_n[i] = segm[(ind[0] + ind[-1]) // 2]
            return x_n, segm_n if segm is not None else None

    def random_pad(self, x, segm=None):
        length = self.num_segments
        if x.shape[0] > length:
            strt = np.random.randint(0, x.shape[0] - length)
            x_ret = x[strt:strt + length]
            if segm is not None:
                segm = segm[strt:strt + length]
                return x_ret, segm
        elif x.shape[0] == length:
            return x, segm
        else:
            pad_len = length - x.shape[0]
            x_ret = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            if segm is not None:
                segm = np.pad(segm, ((0, pad_len), (0, 0)), mode='constant')
            return x_ret, segm

    def random_perturb(self,num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        for i in range(num_segments):
            if i < num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        samples = np.floor(samples)
        return samples.astype(int)

class GTEA_SampleDataset:
    def __init__(self, args, mode="both",sampling='random'):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling=sampling
        self.num_segments_list = args.max_seqlen_list
        self.num_segments_single = args.max_seqlen_single 
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(args.path_dataset,self.dataset_name + "-{}-JOINTFeatures.npy".format(args.feature_type))
        self.path_to_annotations = os.path.join(args.path_dataset,self.dataset_name + "-Annotations/")
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )
        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        # Specific to Thumos14

        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.batch_size = args.batch_size
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]
        try:
            ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]
        except:
            ambilist = []
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode("utf-8") == "training":  # Specific to Thumos14
                self.trainidx.append(i)
            elif s.decode("utf-8") == "validation":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self,is_single=True,is_training=True):
        if is_training:
            
            if is_single: # pretrain num segment
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                feat = []
                for i in idx:
                    ifeat = self.features[i]
                    if self.sampling == 'random':
                        sample_idx = self.random_perturb(self.num_segments_single,ifeat.shape[0])
                    elif self.sampling == 'uniform':
                        sample_idx = self.uniform_sampling(self.num_segments_single,ifeat.shape[0])
                    elif self.sampling == "all":
                        sample_idx = np.arange(ifeat.shape[0])
                    else:
                        raise AssertionError('Not supported sampling !')
                    ifeat = ifeat[sample_idx]
                    feat.append(ifeat)
                feat = np.array(feat)
                labels = np.array([self.labels_multihot[i] for i in idx])
                if self.mode == "rgb":
                    feat = feat[..., : self.feature_size]
                elif self.mode == "flow":
                    feat = feat[..., self.feature_size :]
                
                return feat, labels,rand_sampleid,idx

            else: # multi-scale num segments
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                ms_feat = []
                for segments in self.num_segments_list:
                    feat = []
                    for i in idx:
                        ifeat = self.features[i]
                        if self.sampling == 'random':
                            sample_idx = self.random_perturb(segments,ifeat.shape[0])
                        elif self.sampling == 'uniform':
                            sample_idx = self.uniform_sampling(segments,ifeat.shape[0])
                        elif self.sampling == "all":
                            sample_idx = np.arange(ifeat.shape[0])
                        else:
                            raise AssertionError('Not supported sampling !')
                        ifeat = ifeat[sample_idx]
                        feat.append(ifeat)
                    feat = np.array(feat)
                    labels = np.array([self.labels_multihot[i] for i in idx])
                    if self.mode == "rgb":
                        feat = feat[..., : self.feature_size]
                    elif self.mode == "flow":
                        feat = feat[..., self.feature_size :]
                    
                    ms_feat.append(feat)

                return ms_feat, labels,rand_sampleid,idx

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            # feat = utils.process_feat(feat, normalize=self.normalize)
            # feature = feature[sample_idx]
            vn = self.videonames[self.testidx[self.currenttestidx]]
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            # return feat, np.array(labs),vn, done
            output_dict = {
                'feat': feat,
                'lab': np.array(labs),
                'vn': vn,
                'done': done,
                'mode': 'All'
                # 'sample_idx': sample_idx,
            }

            return output_dict  # feat, np.array(labs),vn, done

    def random_avg(self, x, segm=None):
        if len(x) < self.num_segments:
            ind = self.random_perturb(len(x))
            x_n = x[ind]
            segm = segm[ind] if segm is not None else None
            return x_n, segm
        else:
            inds = np.array_split(np.arange(len(x)), self.num_segments)
            x_n = np.zeros((self.num_segments, x.shape[-1])).astype(x.dtype)
            segm_n = np.zeros(
                (self.num_segments, segm.shape[-1])).astype(x.dtype)
            for i, ind in enumerate(inds):
                x_n[i] = np.mean(x[ind], axis=0)
                if segm is not None:
                    segm_n[i] = segm[(ind[0] + ind[-1]) // 2]
            return x_n, segm_n if segm is not None else None

    def random_pad(self, x, segm=None):
        length = self.num_segments
        if x.shape[0] > length:
            strt = np.random.randint(0, x.shape[0] - length)
            x_ret = x[strt:strt + length]
            if segm is not None:
                segm = segm[strt:strt + length]
                return x_ret, segm
        elif x.shape[0] == length:
            return x, segm
        else:
            pad_len = length - x.shape[0]
            x_ret = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            if segm is not None:
                segm = np.pad(segm, ((0, pad_len), (0, 0)), mode='constant')
            return x_ret, segm

    def random_perturb(self, num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        for i in range(num_segments):
            if i < num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        samples = np.floor(samples)
        return samples.astype(int)

class BEOID_SampleDataset:
    def __init__(self, args, mode="both",sampling='random'):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling=sampling
        self.num_segments_list = args.max_seqlen_list
        self.num_segments_single = args.max_seqlen_single 
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(args.path_dataset,self.dataset_name + "-{}-JOINTFeatures.npy".format(args.feature_type))
        self.path_to_annotations = os.path.join(args.path_dataset,self.dataset_name + "-Annotations/")
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )
        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        # Specific to Thumos14

        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.batch_size = args.batch_size
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]
        try:
            ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]
        except:
            ambilist = []
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode("utf-8") == "training":  # Specific to Thumos14
                self.trainidx.append(i)
            elif s.decode("utf-8") == "validation":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self,is_single=True,is_training=True):
        if is_training:
            
            if is_single: # pretrain num segment
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                feat = []
                for i in idx:
                    ifeat = self.features[i]
                    if self.sampling == 'random':
                        sample_idx = self.random_perturb(self.num_segments_single,ifeat.shape[0])
                    elif self.sampling == 'uniform':
                        sample_idx = self.uniform_sampling(self.num_segments_single,ifeat.shape[0])
                    elif self.sampling == "all":
                        sample_idx = np.arange(ifeat.shape[0])
                    else:
                        raise AssertionError('Not supported sampling !')
                    ifeat = ifeat[sample_idx]
                    feat.append(ifeat)
                feat = np.array(feat)
                labels = np.array([self.labels_multihot[i] for i in idx])
                if self.mode == "rgb":
                    feat = feat[..., : self.feature_size]
                elif self.mode == "flow":
                    feat = feat[..., self.feature_size :]
                
                return feat, labels,rand_sampleid,idx

            else: # multi-scale num segments
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                ms_feat = []
                for segments in self.num_segments_list:
                    feat = []
                    for i in idx:
                        ifeat = self.features[i]
                        if self.sampling == 'random':
                            sample_idx = self.random_perturb(segments,ifeat.shape[0])
                        elif self.sampling == 'uniform':
                            sample_idx = self.uniform_sampling(segments,ifeat.shape[0])
                        elif self.sampling == "all":
                            sample_idx = np.arange(ifeat.shape[0])
                        else:
                            raise AssertionError('Not supported sampling !')
                        ifeat = ifeat[sample_idx]
                        feat.append(ifeat)
                    feat = np.array(feat)
                    labels = np.array([self.labels_multihot[i] for i in idx])
                    if self.mode == "rgb":
                        feat = feat[..., : self.feature_size]
                    elif self.mode == "flow":
                        feat = feat[..., self.feature_size :]
                    
                    ms_feat.append(feat)

                return ms_feat, labels,rand_sampleid,idx

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            # feat = utils.process_feat(feat, normalize=self.normalize)
            # feature = feature[sample_idx]
            vn = self.videonames[self.testidx[self.currenttestidx]]
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            # return feat, np.array(labs),vn, done
            output_dict = {
                'feat': feat,
                'lab': np.array(labs),
                'vn': vn,
                'done': done,
                'mode': 'All'
                # 'sample_idx': sample_idx,
            }

            return output_dict  # feat, np.array(labs),vn, done

    def random_avg(self, x, segm=None):
        if len(x) < self.num_segments:
            ind = self.random_perturb(len(x))
            x_n = x[ind]
            segm = segm[ind] if segm is not None else None
            return x_n, segm
        else:
            inds = np.array_split(np.arange(len(x)), self.num_segments)
            x_n = np.zeros((self.num_segments, x.shape[-1])).astype(x.dtype)
            segm_n = np.zeros(
                (self.num_segments, segm.shape[-1])).astype(x.dtype)
            for i, ind in enumerate(inds):
                x_n[i] = np.mean(x[ind], axis=0)
                if segm is not None:
                    segm_n[i] = segm[(ind[0] + ind[-1]) // 2]
            return x_n, segm_n if segm is not None else None

    def random_pad(self, x, segm=None):
        length = self.num_segments
        if x.shape[0] > length:
            strt = np.random.randint(0, x.shape[0] - length)
            x_ret = x[strt:strt + length]
            if segm is not None:
                segm = segm[strt:strt + length]
                return x_ret, segm
        elif x.shape[0] == length:
            return x, segm
        else:
            pad_len = length - x.shape[0]
            x_ret = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            if segm is not None:
                segm = np.pad(segm, ((0, pad_len), (0, 0)), mode='constant')
            return x_ret, segm

    def random_perturb(self, num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        for i in range(num_segments):
            if i < num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, num_segments, length):
        if num_segments == length:
            return np.arange(num_segments).astype(int)
        samples = np.arange(num_segments) * length / num_segments
        samples = np.floor(samples)
        return samples.astype(int)

class SFNET_Dataset():
    def __init__(self, args, groundtruth_file=None, train_subset='validation', test_subset='test', preprocess_feat=False, mode='weakly', use_sf=False):
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.dataset_name = args.dataset_name
        #  self.num_class = args.num_class
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(
            args.path_dataset, self.dataset_name + '-I3D-JOINTFeatures.npy')
        self.path_to_annotations = os.path.join(
            args.path_dataset, self.dataset_name + '-Annotations')
        self.features = np.load(self.path_to_features, encoding='bytes',allow_pickle=True)
        self.segments = np.load(os.path.join(
            self.path_to_annotations, 'segments.npy'),allow_pickle=True)
        self.gtlabels = np.load(os.path.join(
            self.path_to_annotations, 'labels_sfnet.npy'),allow_pickle=True)
        self.labels = np.load(os.path.join(self.path_to_annotations,
                                           'labels_all_sfnet.npy'),allow_pickle=True)  # Specific to Thumos14
        self.fps = 25
        if groundtruth_file:
            with open(groundtruth_file, 'r') as fr:
                self.gt_info = json.load(fr)['database']
        else:
            self.gt_info = {}
        self.stride = 16
        if self.dataset_name == 'Thumos14':
            self.classlist20 = np.load(os.path.join(self.path_to_annotations,
                                                    'classlist_20classes.npy'),allow_pickle=True)
        self.classlist = np.load(os.path.join(
            self.path_to_annotations, 'classlist.npy'),allow_pickle=True)
        self.subset = np.load(os.path.join(
            self.path_to_annotations, 'subset.npy'),allow_pickle=True)
        self.duration = np.load(os.path.join(
            self.path_to_annotations, 'duration_sfnet.npy'),allow_pickle=True)
        self.videoname = np.load(os.path.join(
            self.path_to_annotations, 'videoname.npy'),allow_pickle=True)
        self.seed = args.seed
        self.lst_valid = None
        if preprocess_feat:
            lst_valid = []
            for i in range(self.features.shape[0]):
                feat = self.features[i]
                mxlen = np.sum(np.max(np.abs(feat), axis=1) > 0, axis=0)
                # Remove videos with less than 5 segments
                if mxlen > 5:
                    lst_valid.append(i)
            self.lst_valid = lst_valid
            if len(lst_valid) != self.features.shape[0]:
                self.features = self.features[lst_valid]
                self.subset = self.subset[lst_valid]
                self.videoname = self.videoname[lst_valid]
                self.duration = self.duration[lst_valid]
                self.gtlabels = self.gtlabels[lst_valid]
                self.labels = self.labels[lst_valid]
                self.segments = self.segments[lst_valid]

        self.batch_size = args.batch_size
        # self.t_max = args.max_seqlen
        self.num_segments_list = args.max_seqlen_list
        self.num_segments_single = args.max_seqlen_single 
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.currentvalidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist) for labs in self.labels
        ]
        self.train_test_idx()
        self.classwise_feature_mapping()
        self.labels101to20 = None
        if self.dataset_name == 'Thumos14':
            self.labels101to20 == np.array(self.classes101to20())
        self.class_order = self.get_class_id()
        self.count_labels = self.get_count()
        np.random.seed(self.seed)
        if mode == 'weakly' or mode == 'fully':
            self.init_frame_labels = self.get_all_frame_labels()
        elif mode == 'single':
            if use_sf:
                self.init_frame_labels = self.get_labeled_frame_labels(
                    os.path.join(self.path_to_annotations, 'single_frames'))
            else:
                self.init_frame_labels = self.get_rand_frame_labels()
        else:
            raise ValueError('wrong mode setting')

        #  self.init_frame_labels = self.get_mid_frame_labels()
        #  self.init_frame_labels = self.get_midc_frame_labels()
        #  self.init_frame_labels = self.get_bgmid_frame_labels()
        #  self.init_frame_labels  =self.get_frame_labels_custom_distribution()
        #  self.init_frame_labels = self.get_start_frame_labels(0.5)
        #  self.init_frame_labels = self.get_bgrand_frame_labels()
        #  self.init_frame_labels = self.get_bgall_frame_labels()
        self.frame_labels = copy.deepcopy(self.init_frame_labels)
        self.all_frame_labels = self.get_all_frame_labels()
        self.clusters = self.init_clusters()
        self.num_frames = np.sum([
            np.sum([len(p) for p in self.frame_labels[i] if len(p) > 0])
            for i in self.trainidx
        ])

    def get_labeled_frame_labels(self, annotation_dire):
        import pandas as pd

        def strip(text):
            try:
                return text.strip()
            except AttributeError:
                return text

        def make_float(text):
            return float(text.strip())
        datas = []
        for filename in os.listdir(annotation_dire):
            data = pd.read_csv(os.path.join(annotation_dire, filename), names=[
                               'vid', 'time', 'label'], converters={'vid': strip, 'time': make_float, 'label': strip})
            datas.append(data)
        labels = []
        classlist = self.get_classlist()
        for i in range(len(self.videoname)):
            data = datas[np.random.choice(range(len(datas)))]
            max_len = len(self.features[i])
            frame_label = [[] for _ in range(max_len)]
            if i not in self.trainidx:
                labels += [frame_label]
                continue
            vname = self.videoname[i].decode('utf-8')
            fps = self.get_fps(i)
            time_class = data[data.vid == vname][['time', 'label']].to_numpy()
            for t, c in time_class:
                pos = int(t * fps / self.stride)
                if pos >= max_len:
                    continue
                intl = utils.strlist2indlist([c], classlist)[0]
                frame_label[pos].append(intl)
            labels += [frame_label]
        return labels

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == self.train_subset and len(self.gtlabels[i]) > 0:
                #  if s.decode('utf-8') == train_str:
                self.trainidx.append(i)
            elif s.decode('utf-8') == self.test_subset:
                self.testidx.append(i)

    def classwise_feature_mapping(self):

        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode('utf-8'):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self, is_single=True,is_training=True):

        if is_training == True:
            if is_single: # pretrain num segment
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])
                feat = np.array(
                    [utils.process_feat(self.features[i], self.num_segments_single) for i in idx])
                # feat = []
                # for i in idx:
                #     ifeat = self.features[i]
                #     if self.sampling == 'random':
                #         sample_idx = self.random_perturb(self.num_segments_single,ifeat.shape[0])
                #     elif self.sampling == 'uniform':
                #         sample_idx = self.uniform_sampling(self.num_segments_single,ifeat.shape[0])
                #     elif self.sampling == "all":
                #         sample_idx = np.arange(ifeat.shape[0])
                #     else:
                #         raise AssertionError('Not supported sampling !')
                #     ifeat = ifeat[sample_idx]
                #     feat.append(ifeat)
                count_labels = np.array([self.count_labels[i] for i in idx])
                if self.labels101to20 is not None:
                    count_labels = count_labels[:, self.labels101to20]
                labels = np.array([self.labels_multihot[i] for i in idx])

                return feat, labels,rand_sampleid,idx

            else: # multi-scale num segments
                labels = []
                idx = []

                rand_sampleid = np.random.choice(
                    len(self.trainidx),
                    size=self.batch_size
                )

                for r in rand_sampleid:
                    idx.append(self.trainidx[r])

                ms_feat = []
                for segments in self.num_segments_list:
                    feat = np.array([utils.process_feat(self.features[i], segments) for i in idx])
                    # feat = []
                    # for i in idx:
                    #     ifeat = self.features[i]
                    #     if self.sampling == 'random':
                    #         sample_idx = self.random_perturb(segments,ifeat.shape[0])
                    #     elif self.sampling == 'uniform':
                    #         sample_idx = self.uniform_sampling(segments,ifeat.shape[0])
                    #     elif self.sampling == "all":
                    #         sample_idx = np.arange(ifeat.shape[0])
                    #     else:
                    #         raise AssertionError('Not supported sampling !')
                    #     ifeat = ifeat[sample_idx]
                    #     feat.append(ifeat)
                    # feat = np.array(feat)
                    labels = np.array([self.labels_multihot[i] for i in idx])

                    ms_feat.append(feat)

                return ms_feat, labels,rand_sampleid,idx


        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]

            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1

            return np.array([feat]), np.array(labs), done

    def get_feature(self, idx):
        return copy.deepcopy(self.features[idx])

    def get_vname(self, idx):
        return self.videoname[idx].decode('utf-8')

    def get_duration(self, idx):
        return self.duration[idx]

    def get_init_frame_label(self, idx):
        return copy.deepcopy(self.init_frame_labels[idx])

    def get_frame_data(self):
        features = []
        labels = []
        one_hots = np.eye(len(self.get_classlist()))
        for idx in self.trainidx:
            feature = self.get_feature(idx)
            frame_label = self.get_frame_label(idx)
            assert len(feature) == len(frame_label)
            for i, ps in enumerate(frame_label):
                if len(ps) < 1:
                    continue
                else:
                    for p in ps:
                        features += [feature[i]]
                        labels += [one_hots[p]]
        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    def get_frame_label(self, idx):
        return copy.deepcopy(self.frame_labels[idx])

    def get_fps(self, idx):
        vname = self.videoname[idx].decode('utf-8')
        try:
            fps = self.gt_info[vname].get('fps', self.fps)
        except:
            fps = self.fps
        return fps

    def get_video_label(self, idx, background=False):
        video_label = np.concatenate(self.all_frame_labels[idx]).astype(int)
        video_label = list(set(video_label))
        return video_label

    def get_gt_frame_label(self, idx):
        return copy.deepcopy(self.all_frame_labels[idx])

    def update_frame_label(self, idx, label):
        self.frame_labels[idx] = label

    def get_trainidx(self):
        return copy.deepcopy(self.trainidx)

    def get_testidx(self):
        return copy.deepcopy(self.testidx)

    def get_segment(self, idx):
        return self.segments[idx]

    def get_classlist(self):
        if self.dataset_name == 'Thumos14':
            return self.classlist20
        else:
            return self.classlist

    def get_frame_counts(self):
        #  counts = np.sum([len(np.where(self.frame_labels[i] != -1)[0]) for i in self.trainidx])
        return self.num_frames

    def update_num_frames(self):
        self.num_frames = np.sum([
            np.sum([1 for p in self.frame_labels[i] if len(p) > 0])
            for i in self.trainidx
        ])

    def classes101to20(self):

        classlist20 = np.array([c.decode('utf-8') for c in self.classlist20])
        classlist101 = np.array([c.decode('utf-8') for c in self.classlist])
        labelsidx = []
        for categoryname in classlist20:
            labelsidx.append([
                i for i in range(len(classlist101))
                if categoryname == classlist101[i]
            ][0])

        return labelsidx

    def get_class_id(self):
        # Dict of class names and their indices
        d = dict()
        for i in range(len(self.classlist)):
            k = self.classlist[i]
            d[k.decode('utf-8')] = i
        return d

    def get_count(self):
        # Count number of instances of each category present in the video
        count = []
        num_class = len(self.class_order)
        for i in range(len(self.gtlabels)):
            gtl = self.gtlabels[i]
            cnt = np.zeros(num_class)
            for j in gtl:
                cnt[self.class_order[j]] += 1
            count.append(cnt)
        count = np.array(count)
        return count

    def init_clusters(self):
        clusters = [[] for _ in range(len(self.frame_labels))]
        for idx in self.trainidx:
            frame_label = self.get_init_frame_label(idx)
            for jdx, pls in enumerate(frame_label):
                if len(pls) < 1:
                    continue
                for pl in pls:
                    clusters[idx].append([jdx, jdx, jdx, pl])
        return clusters

    def get_mid_frame_labels(self):
        classlist = self.get_classlist()
        labels = []
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            frame_label = [[] for _ in range(max_len)]
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = utils.strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg)
                    if start >= end:
                        continue
                    mid = (end + start) / 2 + np.random.randn()
                    mid = int(mid * fps / self.stride)
                    if mid < 0 or mid >= max_len:
                        continue
                    if intl not in frame_label[mid]:
                        frame_label[mid].append(intl)
                labels += [frame_label]
        return labels

    def get_all_frame_labels(self):
        classlist = self.get_classlist()
        labels = []
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(max_len)]
            mids = []
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = utils.strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg) * fps / self.stride
                    if end < start:
                        continue
                    elif end - start < 1.0:
                        end += 1
                    start = max(0, int(start))
                    end = min(max_len, int(end))
                    for pid in range(start, end):
                        if intl not in frame_label[pid]:
                            frame_label[pid].append(intl)
                labels += [frame_label]
        return labels

    def get_bgmid_frame_labels(self):
        classlist = self.get_classlist()
        labels = []
        for i, vid_seg in enumerate(self.segments):
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(len(self.features[i]))]
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                mid = (np.mean(vid_seg, axis=1) *
                       fps / self.stride).astype(int)
                effect_id = np.where(
                    np.logical_and(mid >= 0, mid <= len(self.features[i])))[0]
                mid = mid[effect_id]
                label = np.array(self.gtlabels[i])[effect_id]
                mid_label = np.array(utils.strlist2indlist(label,
                                                           classlist))

                [frame_label[i].append(l) for i, l in zip(mid, mid_label)]
                mid = sorted(mid)
                s = np.concatenate([np.zeros(1), mid], axis=0)
                e = np.concatenate(
                    [mid, np.array([len(self.features[i])])], axis=0)
                bg = ((e + s) / 2).astype(int)
                bg = list(set(bg) - set(mid))
                [frame_label[i].append(0) for i in bg]
                labels += [frame_label]
        return labels

    def get_rand_frame_labels(self):
        classlist = self.get_classlist()
        labels = []
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(max_len)]
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = utils.strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg) * fps / self.stride
                    start = max(0, int(np.ceil(start)))
                    end = min(max_len, int(end))
                    if end <= start:
                        continue
                    mid = np.random.choice(range(start, end), 1)[0]
                    if intl not in frame_label[mid]:
                        frame_label[mid].append(intl)
                labels += [frame_label]
        return labels

    def get_frame_labels_custom_distribution(self):
        custom_dist = [0.12, 0.16, 0.19, 0.16,
                       0.12, 0.11, 0.06, 0.03, 0.02, 0.03]
        np.random.seed(self.seed)
        classlist = self.get_classlist()
        labels = []
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(max_len)]
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = utils.strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg) * fps / self.stride
                    start = max(0, int(np.ceil(start)))
                    end = min(max_len, int(end))
                    if end <= start:
                        continue
                    rs = np.random.choice(range(10), p=custom_dist)
                    bias = np.random.uniform()
                    mid = int(start + (rs * 0.1 + bias) * (end - start))
                    mid = max(min(mid, end-1), start)
                    if intl not in frame_label[mid]:
                        frame_label[mid].append(intl)
                labels += [frame_label]
        return labels

    def get_start_frame_labels(self, ratio=1):
        np.random.seed(self.seed)
        classlist = self.get_classlist()
        labels = []
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(max_len)]
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = utils.strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg) * fps / self.stride
                    start = max(0, int(np.ceil(start)))
                    end = min(max_len, int(end))
                    if end <= start:
                        continue
                    length = max(int((end - start) * ratio), 1)
                    mid = np.random.choice(range(start, start+length), 1)[0]
                    if intl not in frame_label[mid]:
                        frame_label[mid].append(intl)
                labels += [frame_label]
        return labels

    def get_midc_frame_labels(self):
        labels = []
        classlist = self.get_classlist()
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(max_len)]
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = utils.strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg) * fps / self.stride
                    start = max(0, int(np.ceil(start)))
                    end = min(max_len, int(end))
                    if end <= start:
                        continue
                    mid = (start + end) // 2
                    bias = max(int((end - start) * 0.2), 1)
                    sign = np.random.choice([1, -1])
                    bias = np.random.choice(range(bias))
                    mid = mid + sign * bias
                    if intl not in frame_label[mid]:
                        frame_label[mid].append(intl)
                labels += [frame_label]
        return labels

    def get_bgrand_frame_labels(self):
        labels = []
        classlist = self.get_classlist()
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(max_len)]
            mids = []
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = utils.strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg) * fps / self.stride
                    start = max(0, int(np.ceil(start)))
                    end = min(max_len, int(end))
                    if end <= start:
                        continue
                    #  mid = (start+end) // 2
                    mid = np.random.choice(range(start, end), 1)[0]
                    if intl not in frame_label[mid]:
                        frame_label[mid].append(intl)
                    mids += [mid]
                mids = sorted(mids)
                s = np.concatenate([np.zeros(1), mids], axis=0)
                e = np.concatenate(
                    [mids, np.array([len(self.features[i])])], axis=0)
                bg = ((e + s) / 2).astype(int)
                bg = list(set(bg) - set(mids))
                [frame_label[i].append(0) for i in bg]
                labels += [frame_label]
        return labels

    def get_bgall_frame_labels(self):
        labels = []
        classlist = self.get_classlist()
        for i, vid_seg in enumerate(self.segments):
            max_len = len(self.features[i])
            fps = self.get_fps(i)
            assert len(vid_seg) == len(self.gtlabels[i])
            #  frame_label = np.array([-1] * len(self.features[i]))
            frame_label = [[] for _ in range(max_len)]
            mids = []
            if len(vid_seg) < 1:
                labels += [frame_label]
            else:
                for seg, l in zip(vid_seg, self.gtlabels[i]):
                    intl = utils.strlist2indlist([l], classlist)[0]
                    start, end = np.array(seg) * fps / self.stride
                    start = max(0, int(np.ceil(start)))
                    end = min(max_len, int(end))
                    if end <= start:
                        continue
                    for pid in range(start, end):
                        if intl not in frame_label[pid]:
                            frame_label[pid].append(intl)
                    for j in range(len(frame_label)):
                        if len(frame_label[j]) == 0:
                            frame_label[j] += [0]
                labels += [frame_label]
        return labels

    def load_frame_data(self):
        '''

        load frame data for training

        '''

        features = []
        labels = []
        inds = []
        count_labels = []
        video_labels = []
        frame_labels = []
        cent_labels = []
        frame_ids = []
        classlist = self.get_classlist()
        one_hots = np.eye(len(classlist))

        # random sampling
        rand_sampleid = np.arange(len(self.trainidx))
        np.random.shuffle(rand_sampleid)
        for i in rand_sampleid:
            inds.append(self.trainidx[i])
            idx = self.trainidx[i]
            feat = self.get_feature(idx)
            frame_label = self.get_frame_label(idx)
            #  count_label = np.zeros(len(classlist)+1)
            if len(feat) <= self.t_max:
                feat = np.pad(feat, ((0, self.t_max - len(feat)), (0, 0)),
                              mode='constant')
            else:
                r = np.random.randint(len(feat) - self.t_max)
                feat = feat[r:r + self.t_max]
                frame_label = frame_label[r:r + self.t_max]
            frame_id = [
                i for i in range(len(frame_label)) if len(frame_label[i]) > 0
            ]
            if len(frame_id) < 1:
                continue
            frame_label = [
                np.mean(one_hots[frame_label[i]], axis=0) for i in frame_id
            ]
            count_label = np.sum(frame_label, axis=0)
            video_label = (count_label > 0).astype(np.float32)
            #  video_label[0] = 1.0
            #  count_label[0] = 1
            #  count_label[0] = np.max(count_label)
            video_labels += [video_label]
            count_labels += [count_label]
            frame_labels += [np.array(frame_label)]
            features += [feat]
            frame_ids += [frame_id]
            if len(features) == self.batch_size:
                break

        frame_labels = np.concatenate(frame_labels, 0)
        return np.array(features), np.array(video_labels), np.array(
            count_labels), frame_labels, frame_ids

if __name__ == '__main__':
    args = options.parser.parse_args()
    dt = SampleDataset(args)
    data = dt.load_data()
    print(data[0][0].shape)
    # count=0
    # while(1):
    #     data = dt.load_data()
    #     # print(data)
    #     if (np.isnan(data[0]).any()==True):
    #         break
    #     print(count)
    #     count+=1

        # pdb.set_trace()
        # print(dt)