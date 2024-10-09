import argparse

# from yaml import parse

parser = argparse.ArgumentParser(description='WTALC')

# model basic
parser.add_argument('--seed', type=int, default=3552, help='random seed (default: 1)')
parser.add_argument('--model-name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--use-model',type=str,help='model used to train the network')
parser.add_argument('--max-iter', type=int, default=10000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--delta',type=float,default=0.3,help="the fusion weight of RGB and Flow attention (default:1,RGB only)")
parser.add_argument('--interval', type=int, default=200,help='time interval of performing the test')
parser.add_argument('--show_log',type=bool,default=False)
parser.add_argument('--k',type=int,default=7)
parser.add_argument('--dropout_ratio',type=float,default=0.7)
parser.add_argument('--use_ms',action='store_true', default=False,help="whether use ms data") 

# for pooling kernel size calculate
parser.add_argument('--t',type=int,default=4)


# dataset
parser.add_argument('--dataset',type=str,default='SampleDataset')
parser.add_argument('--path-dataset', type=str, default='/home/share/fating/ProcessDataset/Thumos14', help='the path of data feature')
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--max_seqlen_list',type=int,nargs='+',default=[560], help='maximum sequence length during training (default: 750)')
parser.add_argument('--max_seqlen_single',type=int,default=560, help='maximum sequence length during training (default: 750)')
parser.add_argument('--num-class', type=int,default=20, help='number of classes (default: )')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on (default: )')
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used I3D or UNT (default: I3D)')

# loss
parser.add_argument('--lr', type=float, default=0.00003,help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument("--lambda_cls", type=float, default=1,help='weight of MIL loss')
parser.add_argument('--lambda_cll',type=float,default=0,help='weight of DCL loss')
parser.add_argument('--lambda_pcl',type=float,default=0,help='weight of PCL loss')
parser.add_argument('--alpha',type=float,default=0,help="the hyperparameters for uncertainty term")
parser.add_argument('--lambda_lpl',type=float,default=0,help='MPCL loss weight')
parser.add_argument('--SCL_method',type=str,default='after_pcl', choices=('no_pcl','after_pcl'))
parser.add_argument('--lambda_scl',type=float,default=0,help='the weight of FBD_loss')
parser.add_argument('--lambda_mscl',type=float,default=0,help='the weight of FBD_loss in finetune ms_criterion stage')
parser.add_argument('--SCL_align',type=str,default="BCE",help="choose the target of KL_divergence: ['BCE','symKL','MAE','MSE','Tanh','KL','JS,'Dexp']")
parser.add_argument('--SCL_alpha',type=float,default=0,help="the hyperparameters for uncertainty term")
parser.add_argument('--factor',type=float,default=0.5,help="the weight of SCL loss")
parser.add_argument('--lambda_pl',type=float,default=0,help='the weight of PL loss')
parser.add_argument('--rescale_mode',type=str,default="nearest",help="['linear','nearest','ares','nearest-exact']")
parser.add_argument('--ensemble_weight',type=float,nargs='+',default=[1])
parser.add_argument('--lpl_norm',type=str,default='none',choices=('none','min_max','positive'))
parser.add_argument('--multi_back',action='store_true', default=False,help="multi-scale mean backforward")



# for train pseudo complementary label
parser.add_argument('--pseudo_iter',type=int,default=30000)
parser.add_argument('--PLG_proposal_mode',type=str,default='atn',help='[atn,logits,atn_logits] for pseudo proposal generation')
parser.add_argument('--PLG_act_thres',type=float,nargs=3,default=[0.4, 0.925, 0.025]) # Follow ASM-loc
parser.add_argument('--PLG_method',type=str,default='Pseudo_Complementary_Label_Generation',choices=('Pseudo_Label_Generation',"Pseudo_Complementary_Label_Generation"))
parser.add_argument('--PLG_thres',type=float,default=0.8,help="the thres of choosing indices")
parser.add_argument('--PLG_logits_mode',type=str,default='norm',help="how to process logits ['norm','norm_softmax','softmax','none']")

# for test proposal genration
parser.add_argument('--scale',type=float,default=1)
parser.add_argument("--feature_fps", type=int, default=25)
parser.add_argument('--gamma-oic', type=float, default=0)
parser.add_argument('--att_thresh_params',type=float,nargs=3,default=[0.1,0.9,10],help="the act_thres of att,np.linspace for _v0 else np.arange")
parser.add_argument('--cam_thresh_params',type=float,nargs=3,default=[0.15, 0.25, 0.05],help="the act_thres of cam_logits,type is np.arange")
parser.add_argument('--test_proposal_method',type=str,default='multiple_threshold_hamnet')
parser.add_argument('--test_proposal_mode',type=str,default='att',help='[att,cam,both]')
parser.add_argument('--gamma',type=float,default=0.5)
parser.add_argument('--without_nms',action='store_true', default=False,help="whether use for multi-action") 
parser.add_argument('--nms_mode',type=str,default="soft_nms",help=["soft_nms","nms"]) 
parser.add_argument('--cls_thresh',type=float,default=0.2,help="thumos14:0.2,ant1.3:0.1")
parser.add_argument('--nms_thresh',type=float,default=0.75) 


if __name__=="__main__":
    args = parser.parse_args()
    print(args.max_seqlen_list)
    print(type(args.max_seqlen_list))