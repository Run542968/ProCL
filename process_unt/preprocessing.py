import numpy as np
import h5py as h5

import pdb

path_to_dataset='/home/jiachang/TAL_dataset/Thumos14/'
valid_h5=h5.File(path_to_dataset+'thumos14_valid_untrimmednet_features.hdf5','r')
test_h5=h5.File(path_to_dataset+'thumos14_test_untrimmednet_features.hdf5','r')

videonames = np.load(path_to_dataset+ "Thumos14reduced-Annotations/videoname.npy", allow_pickle=True).tolist()

feats_list=[]

for vn in videonames:
    vn=vn.decode('utf-8')
    if 'valid' in vn:
        feats_list.append(valid_h5[vn][:])
    else:
        feats_list.append(test_h5[vn][:])
#
# import pdb
# pdb.set_trace()
np.save('/home/jiachang/TAL_dataset/Thumos14/Thumos14reduced-UNT_ATC-JOINTFeatures.npy',feats_list,allow_pickle=True)
