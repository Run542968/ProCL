# ablation studies


# ------------------------------------------ Thumos14 ---------------------------------------------------------- # 

## MIL :: 42.043
CUDA_VISIBLE_DEVICES=0 python main.py --use-model Model_TB_late_fusion_att_v4 --train_function 'train_v3' --dataset-name Thumos14reduced --path-dataset /home/jiarun/TAL_dataset/Thumos14 --model-name TCSVT_MIL --seed 355 --delta 0.2 --max_seqlen_single 560 --train_mode 'use_v4' --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --pseudo_iter -1 --interval 50

## MIL+CL :: 46.93
CUDA_VISIBLE_DEVICES=1 python main.py --use-model Model_TB_late_fusion_att_v4 --train_function 'train_v3' --dataset-name Thumos14reduced --path-dataset /home/jiarun/TAL_dataset/Thumos14 --model-name TCSVT_MIL_CL --seed 355 --delta 0.2 --max_seqlen_single 560 --train_mode 'use_v4' --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --lambda_cll 1 --pseudo_iter -1 --interval 50

## MIL+CL+PCL :: 47.1
CUDA_VISIBLE_DEVICES=2 python main.py --use-model Model_TB_late_fusion_att_v4 --train_function 'train_v3' --dataset-name Thumos14reduced --path-dataset /home/jiarun/TAL_dataset/Thumos14 --model-name TCSVT_MIL_CL_PCL --seed 355 --delta 0.2 --max_seqlen_single 560 --train_mode 'use_v4' --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --lambda_cll 1 --lambda_pcl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69  --pseudo_iter -1 --interval 50

## MIL+CL+PCL+FBD  :: 47.96
CUDA_VISIBLE_DEVICES=3 python main.py --use-model Model_TB_late_fusion_att_v4 --train_function 'train_v3' --dataset-name Thumos14reduced --path-dataset /home/jiarun/TAL_dataset/Thumos14 --model-name TCSVT_MIL_CL_PCL_FBD --seed 355 --delta 0.2 --max_seqlen_single 560 --train_mode 'use_v4' --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --lambda_cll 1 --lambda_pcl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69  --lambda_scl 1 --SCL_alpha 1 --pseudo_iter -1 --interval 50

## MIL+CL+FBD :: 46.8
CUDA_VISIBLE_DEVICES=3 python main.py --use-model Model_TB_late_fusion_att_v4 --train_function 'train_v3' --dataset-name Thumos14reduced --path-dataset /home/jiarun/TAL_dataset/Thumos14 --model-name TCSVT_MIL_CL_FBD --seed 355 --delta 0.2 --max_seqlen_single 560 --train_mode 'use_v4' --use_ms --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --lambda_cll 1 --lambda_scl 1 --SCL_alpha 1 --SCL_method 'no_pcl' --pseudo_iter -1 --interval 50

## MIL+CL+MPCL :: 46.5
CUDA_VISIBLE_DEVICES=3 python main.py --use-model Model_TB_late_fusion_att_v4 --train_function 'train_v3' --dataset-name Thumos14reduced --path-dataset /home/jiarun/TAL_dataset/Thumos14 --model-name TCSVT_MIL+CL+MPCL --seed 355 --delta 0.2 --max_seqlen_list 560 1120 280 --train_mode 'use_v4' --use_ms --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --lambda_cll 1 --lambda_lpl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69 --rescale_mode 'nearest' --ensemble_weight 0.33 0.33 0.33 --lpl_norm 'none' --alpha 0 --multi_back --pseudo_iter -1 --interval 50

## MIL+CL+MPCL+FBD --pseudo_iter -1 --interval 50 --alpha 0 --delta 0.2 :: 48.635
CUDA_VISIBLE_DEVICES=0 python main.py --use-model Model_TB_late_fusion_att_v4 --train_function 'train_v3' --dataset-name Thumos14reduced --path-dataset /home/jiarun/TAL_dataset/Thumos14 --model-name TCSVT_MIL_CL_MPCL_FBD --seed 355 --delta 0.2 --max_seqlen_list 560 1120 280 --train_mode 'use_v4' --use_ms --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --lambda_cll 1 --lambda_lpl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69 --rescale_mode 'nearest' --ensemble_weight 0.33 0.33 0.33 --lpl_norm 'none' --alpha 0 --multi_back --lambda_mscl 1 --SCL_alpha 1 --pseudo_iter -1 --interval 50


# --------------------- clean ---------------------------- # 
# reference cmd
CUDA_VISIBLE_DEVICES=0 python main.py --use-model Model_TB_late_fusion_att_v4 --train_function 'train_v3' --dataset-name Thumos14reduced --path-dataset /mnt/Datasets/TAL_dataset/Thumos14 --model-name TEST --seed 355 --delta 0.2 --max_seqlen_list 560 1120 280 --train_mode 'use_v4' --use_ms --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --lambda_cll 1 --lambda_lpl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69 --rescale_mode 'nearest' --ensemble_weight 0.33 0.33 0.33 --lpl_norm 'none' --alpha 0 --multi_back --lambda_mscl 1 --SCL_alpha 1 --pseudo_iter -1 --interval 50


# need to delete
1. --train_function
2. --train_mode

# clean full
CUDA_VISIBLE_DEVICES=0 python main.py --use-model Model_Thumos --dataset-name Thumos14reduced --path-dataset /mnt/Datasets/TAL_dataset/Thumos14 --model-name TEST --seed 355 --delta 0.2 --max_seqlen_list 560 1120 280 --use_ms --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --lambda_cll 1 --lambda_lpl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69 --rescale_mode 'nearest' --ensemble_weight 0.33 0.33 0.33 --lpl_norm 'none' --alpha 0 --multi_back --lambda_mscl 1 --SCL_alpha 1 --pseudo_iter -1 --interval 50

## test clean full
CUDA_VISIBLE_DEVICES=2 python test.py --use-model Model_Thumos --dataset-name Thumos14reduced --path-dataset /mnt/Datasets/TAL_dataset/Thumos14 --model-name TEST --seed 355 --delta 0.2 --max_seqlen_list 560 1120 280 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --PLG_logits_mode 'norm' --PLG_thres 0.69


# clean single-scale
CUDA_VISIBLE_DEVICES=1 python main.py --use-model Model_Thumos --dataset-name Thumos14reduced --path-dataset /mnt/Datasets/TAL_dataset/Thumos14 --model-name CLEAN_MIL_CL_PCL_FBD --seed 355 --delta 0.2 --max_seqlen_single 560 --k 7 --max-iter 20000 --att_thresh_params 0.1 0.925 0.025 --test_proposal_method 'multiple_threshold_hamnet_v3' --test_proposal_mode 'att' --lambda_cll 1 --lambda_pcl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69  --lambda_scl 1 --SCL_alpha 1 --pseudo_iter -1 --interval 50








# ------------------------------------------ ActivityNet1.3 ---------------------------------------------------------- # 
## MIL :: 21.9
CUDA_VISIBLE_DEVICES=1 python main.py --use-model Model_TB_late_fusion_att_ant_v4 --train_function 'train_v3' --path-dataset /home/jiarun/TAL_dataset/ActivityNet1.3/ --dataset-name ActivityNet1.3 --dataset Ant13_SampleDataset --model-name ant_Test_MIL --num-class 200 --seed 3552 --delta 0.3 --t 10 --max_seqlen_single 90 --k 10 --lr 1e-5 --max-iter 30000 --test_proposal_method 'multiple_threshold_hamnet_ant' 
## MIL+CL :: 25.7
CUDA_VISIBLE_DEVICES=1 python main.py --use-model Model_TB_late_fusion_att_ant_v4 --train_function 'train_v3' --path-dataset /home/jiarun/TAL_dataset/ActivityNet1.3/ --dataset-name ActivityNet1.3 --dataset Ant13_SampleDataset --model-name ant_Test_MIL_CL --num-class 200 --seed 3552 --delta 0.3 --t 10 --max_seqlen_single 90 --k 10 --lr 1e-5 --max-iter 30000 --test_proposal_method 'multiple_threshold_hamnet_ant' --lambda_cll 1 
## MIL+CL+PCL :: 25.81
CUDA_VISIBLE_DEVICES=1 python main.py --use-model Model_TB_late_fusion_att_ant_v4 --train_function 'train_v3' --path-dataset /home/jiarun/TAL_dataset/ActivityNet1.3/ --dataset-name ActivityNet1.3 --dataset Ant13_SampleDataset --model-name ant_Test_MIL_CL_PCL --num-class 200 --seed 3552 --delta 0.3 --t 10 --max_seqlen_single 90 --k 10 --lr 1e-5 --max-iter 30000 --test_proposal_method 'multiple_threshold_hamnet_ant' --lambda_cll 1 --lambda_pcl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69 --pseudo_iter 15000 
## MIL+CL+PCL+FBD :: 26.0
CUDA_VISIBLE_DEVICES=0 python main.py --use-model Model_TB_late_fusion_att_ant_v4 --train_function 'train_v3' --path-dataset /home/jiarun/TAL_dataset/ActivityNet1.3/ --dataset-name ActivityNet1.3 --dataset Ant13_SampleDataset --model-name ant_Test_MIL_CL_PCL_FBD --num-class 200 --seed 3552 --delta 0.3 --t 10 --max_seqlen_single 90 --k 10 --lr 1e-5 --max-iter 30000 --test_proposal_method 'multiple_threshold_hamnet_ant' --lambda_cll 1 --lambda_pcl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69 --lambda_scl 1 --SCL_alpha 1 --pseudo_iter 15000 
## MIL+CL+MPCL+FBD :: 26.13
CUDA_VISIBLE_DEVICES=0 python main.py --use-model Model_TB_late_fusion_att_ant_v4 --train_function 'train_v3' --path-dataset /data/jiachang/dataset/ActivityNet1.3/ --dataset-name ActivityNet1.3 --dataset Ant13_SampleDataset --model-name ant_Test_MIL_CL_MPCL_FBD --num-class 200 --seed 3552 --delta 0.3 --t 10 --max_seqlen_list 90 180 50 --use_ms --train_mode 'use_v4' --k 10 --lr 1e-5 --max-iter 30000 --test_proposal_method 'multiple_threshold_hamnet_ant' --lambda_cll 1 --lambda_lpl 1 --PLG_logits_mode 'norm' --PLG_thres 0.69 --rescale_mode 'nearest' --ensemble_weight 0.33 0.33 0.33 --lpl_norm 'none' --alpha 1 --multi_back --lambda_mscl 1 --SCL_alpha 1 --pseudo_iter 15000 
## MIL+CL+MPCL :: 26.0 (这个只跑了PLG_thres=0.685 pseudo_iter=20000的，和其他不能一起比，忽略这个吧)
CUDA_VISIBLE_DEVICES=0 python main.py --use-model Model_TB_late_fusion_att_ant_v4 --train_function 'train_v3' --path-dataset /data/jiachang/dataset/ActivityNet1.3/ --dataset-name ActivityNet1.3 --dataset Ant13_SampleDataset --model-name ant_Test_MIL_CL_MPCL --num-class 200 --seed 3552 --delta 0.3 --t 10 --max_seqlen_list 90 180 50 --use_ms --train_mode 'use_v4' --k 10 --lr 1e-5 --max-iter 30000 --test_proposal_method 'multiple_threshold_hamnet_ant' --lambda_cll 1 --lambda_lpl 1 --PLG_logits_mode 'norm' --PLG_thres 0.685 --rescale_mode 'nearest' --ensemble_weight 0.33 0.33 0.33 --lpl_norm 'none' --alpha 1 --multi_back --pseudo_iter 20000 


# --------------------- clean ---------------------------- # 

# clean full
CUDA_VISIBLE_DEVICES=2 python main.py --use-model Model_Ant --path-dataset /mnt/Datasets/TAL_dataset/ActivityNet1.3/ --dataset-name ActivityNet1.3 --dataset Ant13_SampleDataset --model-name Ant_Clean_Full --num-class 200 --seed 3552 --delta 0.3 --t 10 --max_seqlen_list 90 180 50 --use_ms --k 10 --lr 1e-5 --max-iter 30000 --test_proposal_method 'multiple_threshold_hamnet_ant' --lambda_cll 1 --lambda_lpl 1 --PLG_logits_mode 'norm' --PLG_thres 0.685 --rescale_mode 'nearest' --ensemble_weight 0.33 0.33 0.33 --lpl_norm 'none' --alpha 1 --multi_back --pseudo_iter 20000 

## test clean full
CUDA_VISIBLE_DEVICES=2 python test.py --use-model Model_Ant --path-dataset /mnt/Datasets/TAL_dataset/ActivityNet1.3/ --dataset-name ActivityNet1.3 --dataset Ant13_SampleDataset --model-name Ant_Clean_Full --num-class 200 --seed 3552 --delta 0.3 --t 10 --max_seqlen_list 90 180 50 --test_proposal_method 'multiple_threshold_hamnet_ant' --PLG_logits_mode 'norm' --PLG_thres 0.685

### delete
--test_multi_action
--test_long_action
--test_sparse_action

# clean single-scale
CUDA_VISIBLE_DEVICES=3 python main.py --use-model Model_Ant --path-dataset /mnt/Datasets/TAL_dataset/ActivityNet1.3/ --dataset-name ActivityNet1.3 --dataset Ant13_SampleDataset --model-name Ant_Clean_Single_Scale --num-class 200 --seed 3552 --delta 0.3 --t 10 --max_seqlen_single 90 --k 10 --lr 1e-5 --max-iter 30000 --test_proposal_method 'multiple_threshold_hamnet_ant'  


