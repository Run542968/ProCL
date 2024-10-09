import h5py as h5
import os
import numpy as np

# class Model_TBSM(torch.nn.Module):
#     def __init__(self, n_feature, n_class,**args):
#         super().__init__()
#         embed_dim=2048
#         dropout_ratio=args['opt'].dropout_ratio
#         self.feat_encoder = nn.Sequential(
#             nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
#
#         self.classifier = nn.Sequential(
#             nn.Conv1d(embed_dim, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),
#             nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class+1, 1))
#
#         self.modality_dim=1024
#         self.rgb_attention = nn.Sequential(
#             nn.Conv1d(self.modality_dim, self.modality_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio),nn.Conv1d(self.modality_dim,1,1))
#         self.flow_attention = nn.Sequential(
#             nn.Conv1d(self.modality_dim, self.modality_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio),nn.Conv1d(self.modality_dim,1,1))
#         self.Sigmoid=nn.Sigmoid()
#         self.apply(weights_init)
#
#     def forward(self, inputs, is_training=True, **args):
#         x = inputs.transpose(-1, -2)
#         feat = self.feat_encoder(x)
#         x_cls = self.classifier(feat)
#
#         rgb_atn = self.rgb_attention(x[:,:self.modality_dim])
#         flow_atn=self.flow_attention(x[:,self.modality_dim:])
#
#         x_atn=(torch.sigmoid(rgb_atn)+torch.sigmoid(flow_atn))
#
#         # atn_supp, atn_drop = self.adl(x_cls, x_atn, include_min=True)
#         return feat.transpose(-1, -2), x_cls.transpose(-1, -2), x_atn.transpose(-1, -2), rgb_atn.transpose(-1, -2), flow_atn.transpose(-1, -2),x_atn.transpose(-1,-2)
#
#         # return att_sigmoid,att_logit, feat_emb, bag_logit, instance_logit
#
#     def criterion(self, outputs, labels, **args):
#
#         feat,element_logits, atn_supp,rgb_atn_logit, flow_atn_logit,_ = outputs
#         b,n,c = element_logits.shape
#
#         element_logits_supp = self._multiply(element_logits, atn_supp,include_min=True)
#
#         rgb_atn=torch.sigmoid(rgb_atn_logit)
#         flow_atn=torch.sigmoid(flow_atn_logit)
#
#         loss_1_orig, _ = self.topkloss(element_logits,
#                                        labels,
#                                        is_back=True,
#                                        rat=args['opt'].k,
#                                        reduce=None)
#         # SAL
#         loss_2_orig_supp, _ = self.topkloss(element_logits_supp,
#                                             labels,
#                                             is_back=False,
#                                             rat=args['opt'].k,
#                                             reduce=None)
#
#         # actually, we do not have real attention weights after suppress, so it is not suitable to suppress the logits_supp
#         loss_3_orig_negative = self.negative_loss(element_logits,labels,is_back=True)
#         loss_3_supp_negative = self.negative_loss(element_logits_supp,labels,is_back=True)
#
#         loss_3_orig_Contrastive = self.Contrastive(feat,element_logits,labels,is_back=False)
#         loss_3_supp_Contrastive = self.Contrastive(feat,element_logits_supp,labels,is_back=False)
#
#         atts=torch.cat([rgb_atn,flow_atn],dim=-1)
#         max_att=torch.max(atts,dim=-1,keepdim=True)[0]
#         min_att=torch.min(atts,dim=-1,keepdim=True)[0]
#         element_logits_semi_hard=self._multiply(element_logits,(max_att-min_att),include_min=True)
#         loss_2_semi_hard=self.noisy_topkloss(element_logits_semi_hard,labels,is_back=False,rat=args['opt'].k,loss_type=args['opt'].noisy_loss_type)
#         wt = 0.2
#
#
#         loss_norm = atn_supp.mean()
#
#         # guide loss
#         # loss_guide = (1 - element_atn -
#         #               element_logits.softmax(-1)[..., [-1]]).abs().mean()
#         # loss_guide=(self.Guide_Loss(rgb_atn_logit,element_logits[...,[-1]])+self.Guide_Loss(flow_atn_logit,element_logits[...,[-1]]))/2.
#         loss_guide=self.Guide_Loss(rgb_atn_logit+flow_atn_logit,cls_logit=element_logits[...,[-1]])
#         # loss_guide=self.MAE_Guide_Loss((rgb_atn+flow_atn)/2.,cls_score=torch.softmax(element_logits,dim=-1)[:,:,[-1]])
#         # loss_guide=(self.KL_Loss(-x_atn_logit,element_logits[...,[-1]])+self.KL_Loss(element_logits[...,[-1]],-x_atn_logit))/2
#
#         # total loss
#         # total_loss = (1* loss_1 + 1 * loss_2 +
#         #               0.8* loss_norm +
#         #               0.8* loss_guide)
#         # loss_cycle = losses.Cycle_Cosine_Loss(feat, element_logits_supp, args['opt'].num_similar, labels, args=args['opt'],
#         #                                device=feat.device, is_back=False)
#         #
#         # loss_distill = losses.distill_loss(element_logits, element_logits_supp, labels, feat.device, args['opt'],
#         #                                    is_back=False)
#
#         total_loss = args['opt'].lambda1* (loss_1_orig.mean() +loss_2_orig_supp.mean() )+\
#                       args['opt'].lambda2*(loss_3_orig_negative.mean()) +\
#                      args['opt'].lambda3*(loss_3_supp_Contrastive)+\
#                       args['opt'].lambda4* loss_norm +\
#                       args['opt'].lambda5* loss_guide+\
#                      args['opt'].lambda7*loss_2_semi_hard
#         # +loss_3_supp_negative.mean())+\
#
#         # total_loss = args['opt'].lambda1* (loss_1_orig.mean() +loss_2_orig_supp.mean() )+\
#         #               args['opt'].lambda2*(loss_3_orig_negative.mean() +loss_3_supp_negative.mean())+\
#         #              args['opt'].lambda3*(loss_3_supp_Contrastive.mean()+loss_3_orig_Contrastive.mean())+\
#         #               args['opt'].lambda4* loss_norm +\
#         #               args['opt'].lambda5* loss_guide
#
#         return total_loss
#
#     def _multiply(self, x, atn, dim=-1, include_min=False):
#         if include_min:
#             _min = x.min(dim=dim, keepdim=True)[0]
#         else:
#             _min = 0
#         return atn * (x - _min) + _min
#
#     def _multiply_with_att(self,x,atn,dim=-1,include_min=False):
#
#         if include_min:
#             _min = x.min(dim=dim, keepdim=True)[0]
#         else:
#             _min = 0
#
#         tmp_x=atn*(x-_min)
#     def KL_Loss(self,p_logit,q_logit):
#         p_log=F.log_softmax(p_logit)
#         q_output=F.softmax(q_logit)
#         return F.kl_div(p_log,q_output)
#
#     def Guide_Loss(self,att_logit,cls_logit):
#         p_output=F.softmax(-att_logit)
#         q_output=F.softmax(cls_logit)
#         log_mean_output=((p_output+q_output)/2).log()
#
#         return (F.kl_div(log_mean_output,p_output)+F.kl_div(log_mean_output,q_output))/2
#
#     def MAE_Guide_Loss(self,att_score,cls_score):
#         return torch.abs(att_score-cls_score).mean()
#
#
#     def Contrastive(self,x,element_logits,labels,is_back=False):
#         if is_back:
#             labels = torch.cat(
#                 (labels, torch.ones_like(labels[:, [0]])), dim=-1)
#         else:
#             labels = torch.cat(
#                 (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
#         sim_loss = 0.
#         n_tmp = 0.
#         _, n, c = element_logits.shape
#         for i in range(0, 3*2, 2):
#             atn1 = F.softmax(element_logits[i], dim=0)
#             atn2 = F.softmax(element_logits[i+1], dim=0)
#
#             n1 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
#             n2 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
#             Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)      # (n_feature, n_class)
#             Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
#             Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
#             Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)
#
#             d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
#             d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
#             d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
#             sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
#             sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
#             n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
#         sim_loss = sim_loss / n_tmp
#         return sim_loss
#
#     def negative_loss(self,element_logits,labels,is_back=False):
#         if is_back:
#             labels_with_back = torch.cat(
#                 (labels, torch.ones_like(labels[:, [0]])), dim=-1)
#         else:
#             labels_with_back = torch.cat(
#                 (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
#         b,n,c = element_logits.shape
#         negative_loss = 0
#         element_prob = F.softmax(element_logits,-1)
#         # inverse_prob = 1 - element_prob
#         labels_with_back = labels_with_back.unsqueeze(1).expand(b,n,c)
#         s_loss = -((1-labels_with_back)*torch.log(1 - element_prob+1e-8)).sum(-1).mean(-1)
#         return s_loss
#     def topkloss(self,
#                  element_logits,
#                  labels,
#                  is_back=True,
#                  lab_rand=None,
#                  rat=8,
#                  reduce=None):
#         if is_back:
#             labels_with_back = torch.cat(
#                 (labels, torch.ones_like(labels[:, [0]])), dim=-1)
#         else:
#             labels_with_back = torch.cat(
#                 (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
#
#         if lab_rand is not None:
#             labels_with_back = torch.cat((labels, lab_rand), dim=-1)
#
#         topk_val, topk_ind = torch.topk(
#             element_logits,
#             k=max(1, int(element_logits.shape[-2] // rat)),
#             dim=-2)
#
#         instance_logits = torch.mean(
#             topk_val,
#             dim=-2,
#         )
#
#         labels_with_back = labels_with_back / (
#             torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
#         milloss = (-(labels_with_back *
#                      F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
#
#         if reduce is not None:
#             milloss = milloss.mean()
#         return milloss, topk_ind
#
#     def noisy_topkloss(self,
#                  element_logits,
#                  labels,
#                  is_back=True,
#                  lab_rand=None,
#                  rat=8,
#                  loss_type='MAE'):
#         if is_back:
#             labels_with_back = torch.cat(
#                 (labels, torch.ones_like(labels[:, [0]])), dim=-1)
#         else:
#             labels_with_back = torch.cat(
#                 (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
#
#         if lab_rand is not None:
#             labels_with_back = torch.cat((labels, lab_rand), dim=-1)
#
#         topk_val, topk_ind = torch.topk(
#             element_logits,
#             k=max(1, int(element_logits.shape[-2] // rat)),
#             dim=-2)
#
#         instance_logits = torch.mean(
#             topk_val,
#             dim=-2,
#         )
#
#         labels_with_back = labels_with_back / (
#             torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
#
#         cls_scores=torch.softmax(instance_logits,dim=-1)
#         if loss_type=='MAE':
#             loss= torch.abs(labels_with_back-cls_scores).mean()
#             return loss
#         elif loss_type=='CE':
#             loss=(-(labels_with_back*torch.log(cls_scores+1e-8)).sum(dim=-1)).mean()
#             return loss
#         elif loss_type=='MSE':
#             loss= ((labels_with_back-cls_scores)**2).mean()
#             return loss
#
#     def decompose(self, outputs, **args):
#         feat, instance_logit, atn_supp, atn_drop, element_atn,x_atn_logit  = outputs
#         return instance_logit

class Model_TBSM(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim=2048
        dropout_ratio=args['opt'].dropout_ratio

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class+1, 1))

        self.modality_dim=1024
        self.rgb_attention = nn.Sequential(
            nn.Conv1d(self.modality_dim, self.modality_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio),nn.Conv1d(self.modality_dim,1,1))
        self.flow_attention = nn.Sequential(
            nn.Conv1d(self.modality_dim, self.modality_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio),nn.Conv1d(self.modality_dim,1,1))

        self.apply(weights_init)

        self.Sigmoid=nn.Sigmoid()

    def forward(self, inputs, is_training=True, **args):
        x = inputs.transpose(-1, -2)
        feat = self.feat_encoder(x)
        x_cls = self.classifier(feat)

        rgb_atn_logit = self.rgb_attention(x[:,:self.modality_dim])
        flow_atn_logit=self.flow_attention(x[:,self.modality_dim:])

        rgb_atn=self.Sigmoid(rgb_atn_logit)
        flow_atn=self.Sigmoid(flow_atn_logit)
        x_atn=(rgb_atn+flow_atn)/2.

        # vid_feat=(x_atn*feat).sum(dim=-1,keepdim=True)/torch.sum(x_atn,dim=-1,keepdim=True)
        # vid_logits=self.classifier(vid_feat)

        # atn_supp, atn_drop = self.adl(x_cls, x_atn, include_min=True)
        return feat.transpose(-1, -2), x_cls.transpose(-1, -2), x_atn.transpose(-1, -2), rgb_atn_logit.transpose(-1, -2), flow_atn_logit.transpose(-1, -2),x_atn.transpose(-1,-2)

        # return att_sigmoid,att_logit, feat_emb, bag_logit, instance_logit

    def criterion(self, outputs, labels, **args):

        feat,element_logits, atn_supp, rgb_atn_logit, flow_atn_logit,_ = outputs
        b,n,c = element_logits.shape

        rgb_atn=self.Sigmoid(rgb_atn_logit)
        flow_atn=self.Sigmoid(flow_atn_logit)

        element_logits_supp = self._multiply(element_logits, atn_supp,include_min=True)

        # element_logits_drop = self._multiply(
        #     element_logits, (atn_drop > 0).type_as(element_logits),
        #     include_min=True)
        # element_logits_drop_supp = self._multiply(element_logits,
        #                                           atn_drop,
        #                                           include_min=True)
       # BCL
        loss_1_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=args['opt'].k,
                                       reduce=None)
        # SAL
        loss_2_orig_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)

        # actually, we do not have real attention weights after suppress, so it is not suitable to suppress the logits_supp
        loss_3_orig_negative = self.negative_loss(element_logits,labels,is_back=True)
        loss_3_supp_negative = self.negative_loss(element_logits_supp,labels,is_back=True)

        loss_3_orig_Contrastive = self.Contrastive(feat,element_logits,labels,is_back=False)
        loss_3_supp_Contrastive = self.Contrastive(feat,element_logits_supp,labels,is_back=False)

        # atts=torch.cat([rgb_atn,flow_atn],dim=-1)
        # max_att=torch.max(atts,dim=-1,keepdim=True)[0]
        # min_att=torch.min(atts,dim=-1,keepdim=True)[0]
        # element_logits_semi_hard=self._multiply(element_logits,(max_att-min_att).detach(),include_min=True)
        # loss_2_semi_hard=self.noisy_topkloss(element_logits_semi_hard,labels,is_back=False,rat=args['opt'].k,loss_type=args['opt'].noisy_loss_type)
        # wt = 0.2
        # loss_norm = atn_supp.mean()
        # guide loss
        # loss_guide = (1 - element_atn -
        #               element_logits.softmax(-1)[..., [-1]]).abs().mean()
        # loss_guide=(self.Batch_Guide_Loss(rgb_atn_logit,element_logits[...,[-1]])+self.Batch_Guide_Loss(flow_atn_logit,element_logits[...,[-1]]))/2
        # # loss_guide=self.Guide_Loss(x_atn_logit,element_logits[...,[-1]])
        # #loss_guide=self.Co_Guide_Loss(rgb_atn_logit,flow_atn_logit,element_logits[...,[-1]],lambd=args['opt'].lambd,loss_type='Mutual')
        # # loss_guide=(self.KL_Loss(-x_atn_logit,element_logits[...,[-1]])+self.KL_Loss(element_logits[...,[-1]],-x_atn_logit))/2
        # # loss_guide=self.MAE_Guide_Loss(rgb_atn_logit,flow_atn_logit)
        # # total loss
        # # total_loss = (1* loss_1 + 1 * loss_2 +
        # #               0.8* loss_norm +
        # #               0.8* loss_guide)
        # loss_cycle = losses.Cycle_Score_Loss(feat, element_logits_supp, args['opt'].num_similar, labels, args=args['opt'],
        #                                device=feat.device, is_back=False)

        # loss_vid=self.BCE_Loss(vid_logit,labels,is_back=False)

        # loss_distill = losses.distill_loss(element_logits, element_logits_supp, labels, feat.device, args['opt'],
        #                                    is_back=False)
        # loss_var=self.Variance_Loss(rgb_atn_logit,flow_atn_logit,element_logits[...,[-1]])


        total_loss = args['opt'].lambda1* (loss_1_orig.mean() +loss_2_orig_supp.mean() )+\
                      args['opt'].lambda2*(loss_3_orig_negative.mean()) +\
                     args['opt'].lambda3*(loss_3_supp_Contrastive)#+\
                     #  args['opt'].lambda4* loss_norm +\
                     #  args['opt'].lambda5* loss_guide+\
                     # args['opt'].lambda6*loss_var+\
                     # args['opt'].lambda7*loss_2_semi_hard
        # +loss_3_supp_negative.mean())+\

        # total_loss = args['opt'].lambda1* (loss_1_orig.mean() +loss_2_orig_supp.mean() )+\
        #               args['opt'].lambda2*(loss_3_orig_negative.mean() +loss_3_supp_negative.mean())+\
        #              args['opt'].lambda3*(loss_3_supp_Contrastive.mean()+loss_3_orig_Contrastive.mean())+\
        #               args['opt'].lambda4* loss_norm +\
        #               args['opt'].lambda5* loss_guide

        return total_loss

    def BCE_Loss(self,logits,labels,is_back=False):
        # select the foreground out for the learning
        # pred logits is with shape [B,C]
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        # fg_pred=logits[:,:,0]
        # cls_pred=logits[:,:,1:]
        # fg_pred=torch.softmax(fg_pred,dim=-1)
        # bg_pred=torch.softmax(bg_pred,dim=-1)
        # labels is with shape [B,C]
        # loss=(-labels*torch.log(fg_pred+1e-8)).sum(-1).mean()-(bg_labels*torch.log(bg_pred+1e-8)).sum(-1).mean()
        logits=logits.squeeze(-1)
        loss=(-labels*torch.log_softmax(logits,dim=-1)).sum(-1).mean()
        return loss

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def _multiply_with_att(self,x,atn,dim=-1,include_min=False):

        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0

        tmp_x=atn*(x-_min)
    def KL_Loss(self,p_logit,q_logit):
        p_log=F.log_softmax(p_logit,dim=1)
        q_output=F.softmax(q_logit,dim=1)
        return F.kl_div(p_log,q_output)

    def Guide_Loss(self,att_logit,cls_logit):
        p_output=F.softmax(-att_logit,dim=1)
        q_output=F.softmax(cls_logit,dim=1)
        log_mean_output=((p_output+q_output)/2).log()

        return (F.kl_div(log_mean_output,p_output)+F.kl_div(log_mean_output,q_output))/2

    def Batch_Guide_Loss(self,att_logit,cls_logit):
        p_output=F.softmax(-att_logit,dim=0)
        q_output=F.softmax(cls_logit,dim=0)
        log_mean_output=((p_output+q_output)/2).log()

        return (F.kl_div(log_mean_output,p_output)+F.kl_div(log_mean_output,q_output))/2

    def Co_Guide_Loss(self,rgb_logit,flow_logit,class_logit,loss_type='MAE',lambd=1):
        loss=0
        # rgb_weights=torch.sigmoid(rgb_logit)
        # flow_weights=torch.sigmoid(flow_logit)
        # cls_weights=1-torch.softmax(class_logit,dim=-1)[...,[-1]]
        #
        # weights=torch.cat([rgb_weights,flow_weights,cls_weights],dim=-1)
        # var,mean=torch.var_mean(weights,dim=-1,keepdim=True,unbiased=False)
        # # to generate the variance mask
        # # make a temporal variance mask, it may need a lambd (temperature) to avoid the center problem
        # # [B,T,C]
        # weighted_map=torch.softmax(var*lambd,dim=1)
        if loss_type=='MAE':
            # dis = torch.abs(weights - mean)
            # weighted_dis=((1-weighted_map)*dis)/(weighted_map.shape[1]-1)
            # weighted_dis=weighted_dis.mean()
            # return weighted_dis
            rgb_weights=torch.sigmoid(rgb_logit)
            flow_weights=torch.sigmoid(flow_logit)
            return torch.abs(rgb_weights-flow_weights).mean()

        elif loss_type=='JS':

            return self.Guide_Loss(rgb_logit,flow_logit).mean()
        elif loss_type=='Mutual':
            return (self.KL_Loss(rgb_logit,class_logit)+self.KL_Loss(flow_logit,class_logit))/2.

    def MAE_Guide_Loss(self,rgb_logit,flow_logit,class_logit):
        rgb_weights=torch.sigmoid(rgb_logit)
        flow_weights=torch.sigmoid(flow_logit)
        class_weights=torch.softmax(class_logit,dim=-1)[...,[-1]]

        return (torch.abs(class_weights.detach()-rgb_weights).mean()+torch.abs(class_weights.detach_()-flow_weights).mean())/2.

    def Variance_Loss(self,rgb_logit,flow_logit,class_logit):

        rgb_weights=torch.softmax(-rgb_logit,dim=1)
        flow_weights=torch.softmax(-flow_logit,dim=1)
        cls_weights=torch.softmax(class_logit,dim=1)
        weights=torch.cat([rgb_weights,flow_weights,cls_weights],dim=-1)

        std=torch.std(weights,dim=-1,keepdim=True)
        return std.mean()

    def Contrastive(self,x,element_logits,labels,is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3*2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i+1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)      # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def negative_loss(self,element_logits,labels,is_back=False):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        b,n,c = element_logits.shape
        negative_loss = 0
        element_prob = F.softmax(element_logits,-1)
        # inverse_prob = 1 - element_prob
        labels_with_back = labels_with_back.unsqueeze(1).expand(b,n,c)
        s_loss = -((1-labels_with_back)*torch.log(1 - element_prob+1e-8)).sum(-1).mean(-1)
        return s_loss

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

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

        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def noisy_topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 loss_type='MAE'):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

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

        cls_scores=torch.softmax(instance_logits,dim=-1)
        if loss_type=='MAE':
            loss= torch.abs(labels_with_back-cls_scores).mean()
            return loss
        elif loss_type=='CE':
            loss=(-(labels_with_back*torch.log(cls_scores+1e-8)).sum(dim=-1)).mean()
            return loss
        elif loss_type=='MSE':
            loss= ((labels_with_back-cls_scores)**2).mean()
            return loss

    def decompose(self, outputs, **args):
        feat, instance_logit, atn_supp, atn_drop, element_atn,x_atn_logit  = outputs
        return instance_logit
