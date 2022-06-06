from .bbox_iou_evaluation import match_bboxes
import torch
from torch import nn
import torch.nn.functional as F

class ContextRefinementBranch(torch.nn.Module):

    def __init__(self, num_box_classes, num_context_classes, loss_alpha=1.0, start_epoch=0):
        super(ContextRefinementBranch, self).__init__()

        self.top_K = 10
        self.in_channels = 0
        self.out_channels = num_box_classes
        self.num_box_class = num_box_classes
        self.num_context_classes = num_context_classes
        self.loss = nn.CrossEntropyLoss()
        self.remove_bg = True
        print("REMOVING BG: {}".format(self.remove_bg))

        self.use_box_features = True
        print("USE BOX FEATURES: {}".format(self.use_box_features))
        if self.use_box_features:
            # kernel_size = 8
            # padding = 0
            # stride = kernel_size
            # dilation = 1
            # self.box_feat_pool = nn.MaxPool1d(kernel_size, stride, padding, dilation)
            box_feature_dim = 1024
            print("BOX FEATURE DIM HARD CODED AS: {}".format(box_feature_dim))
            # pool_out_channels = (box_feature_dim + 2 * padding - dilation * (kernel_size-1) - 1) / stride
            # pool_out_channels += 1
            # self.in_channels = pool_out_channels
            self.in_channels += box_feature_dim
        
        self.use_global_pool_features = True
        print("USE GLOBAL POOL FEATURES {}".format(self.use_global_pool_features))
        if self.use_global_pool_features:
            self.pool_channels = 256
            self.pool_h = 6
            self.pool_w = 21
            self.pool_feature_dim = self.pool_channels * self.pool_w * self.pool_h
            print("POOL FEATURE_DIM_HARD CODED AS {}".format(self.pool_feature_dim))

            self.conv1d_out_channels = 2
            print("CONV1D OUT C: {}".format(self.conv1d_out_channels))
            self.conv1d_kernel_size = 3
            self.conv1d = nn.Conv1d(self.pool_channels, self.conv1d_out_channels, self.conv1d_kernel_size)
            # average pool?
            self.in_channels += self.conv1d_out_channels * self.pool_w * self.pool_h - 2*self.conv1d_out_channels

        # self.in_channels = num_box_classes + num_context_classes
        # if self.remove_bg:
        #     self.in_channels -= 1
        #     self.out_channels -= 1

        self.linear = nn.Linear(self.in_channels, self.out_channels)
        # self.init_with_zeros = True
        # print('INIT CTX BRANCH WITH ZEROS, ADD LABS LATER: {}'.format(self.init_with_zeros))
        # if self.init_with_zeros:
        #     # initialize the linear layer with 0s and add an identity like matrix before loss calc
        #     linear_shape = self.linear.weight.data.shape
        #     self.identity_like = torch.cat(
        #         [torch.eye(linear_shape[0]), torch.zeros([linear_shape[1]-linear_shape[0], linear_shape[0]])]
        #     ).T
        #     self.linear.weight.data.copy_(torch.zeros(self.identity_like.shape))
        #     self.linear.bias.data.copy_(torch.zeros(self.linear.bias.shape))
        self.loss_alpha = loss_alpha
        print("CRB_LOSS_ALPHA: {}".format(self.loss_alpha))
        print("NO RELU HERE")
        self.use_gt_hab = True
        print("USE GT HAB: {}".format(self.use_gt_hab))
        self.use_logit_softmax = False
        print("USE SOFTMAX ON LOGITS: {}".format(self.use_logit_softmax))
        self.start_epoch = start_epoch
        print("WAIT UNTIL EPOCH {} BEFORE ACTIVATING CRB".format(self.start_epoch))

    def one_hot(self, x, class_count):
        return torch.eye(class_count)[x,:]

    def forward(self, features, detections, im_sizes, hab_preds, targets=None):
        # print('targets is None: {}'.format(targets is None))
        device = detections[0]['boxes'].device
        loss = None
        idxs_true, idxs_pred, match_ious, match_labels = None, None, None, None
        # class_logits, box_regression, proposals, image_shapes = det_feats

        batch_size, pool_channels, pool_h, pool_w = features['pool'].shape
        assert(pool_channels == self.pool_channels and self.pool_h == pool_h and pool_w == self.pool_w)
        global_feats = self.conv1d(features['pool'].reshape(batch_size, pool_channels, pool_h * pool_w))

        mlp_in = torch.zeros((len(detections), self.top_K, self.in_channels), device=device)
        det_labels = torch.zeros((len(detections), self.top_K, self.num_box_class), device=device)
        det_labels[:,:,0] = 1 # assume a box is bg, will be replaced if not
        loss_idxs = []
        for i, det in enumerate(detections):
            preds = det['boxes']

            if targets is not None:
                gt_boxes = targets[i]['boxes']
                idxs_true, idxs_pred, match_ious, match_labels = match_bboxes(gt_boxes, preds, 0.25)
                for j in range(len(idxs_true)):
                    if idxs_pred[j] >= self.top_K:
                        continue
                    one_hot_gt_label = self.one_hot(targets[i]['labels'][idxs_true[j]], self.num_box_class)
                    det_labels[i][idxs_pred[j]] = one_hot_gt_label

            # scores = det['scores']
            # labels = det['labels']
            # one_hot_labels = self.one_hot(labels, self.num_box_class).to(device)
            # logits = det['logits']
            # if self.remove_bg:
            #     logits = det['logits'][:,1:]
            # else:
            #     logits = det['logits']   
            # if self.use_logit_softmax:
            #     logits = F.softmax(logits, -1)
            # hab = hab_preds[i]
            # if targets is not None and self.use_gt_hab:
            #     h_tar = targets[i]['habs_hot']
            #     hab_repeat = h_tar.repeat(len(logits), 1).to(device)
            # else:
            #     hab_repeat = hab.repeat(len(logits), 1).to(device)
            # # must have one_hot_labels first to accommodate identity initializiation
            # new_feat = torch.cat([logits, hab_repeat], 1)[:self.top_K]
            # mlp_in[i][:new_feat.shape[0]] = new_feat

            g_feats = global_feats[i].flatten() # may want an avg pool..
            g_feats = g_feats.repeat(self.top_K, 1).to(device)

            box_feats = det['box_features'][:self.top_K]
            g_feats = g_feats[:box_feats.shape[0]]

            new_feat = torch.cat([box_feats[:self.top_K], g_feats], 1)
            mlp_in[i][:new_feat.shape[0]] = new_feat
        
        new_labels = self.linear(mlp_in)
        # if self.init_with_zeros:
        #     assert(logits.shape[1] == self.out_channels)
        #     new_labels = new_labels + mlp_in[:,:,:self.out_channels]
        # if self.remove_bg:
        #     mz = torch.zeros(det_labels.shape).to(device)
        #     mz[:,:,1:] = new_labels
        #     new_labels = mz
        new_labels_for_loss = new_labels.reshape([batch_size * self.top_K, self.num_box_class])
        new_det_labels_for_loss = det_labels.reshape([batch_size * self.top_K, self.num_box_class]).type(torch.long)
        _, new_det_labels_for_loss = torch.where(new_det_labels_for_loss == 1)
        loss = self.loss(new_labels_for_loss, new_det_labels_for_loss)
        loss *= self.loss_alpha

        not_hot_labels = torch.argmax(new_labels, 2, keepdim=False)
        for i, det in enumerate(detections):
            det['labels'][:self.top_K] = not_hot_labels[i][:len(det['labels'])]

        return detections, {"loss_ctx_branch": loss}



        # feat = x['pool'].reshape(x['pool'].shape[0], -1)
        # logits = self.fc(feat)
        # probs = F.softmax(logits)

        # if targets is not None:
        #     labels = torch.zeros(logits.shape)
        #     rows_to_keep = []
        #     for i,target in enumerate(targets):
        #         # some targets may have no / bad habitat labels
        #         # keep only those with habitat labels for loss calc
        #         if not torch.all(target['habs_hot'] == torch.zeros(logits.shape).to(logits.device)):
        #             rows_to_keep.append(i)
        #         labels[i] = target['habs_hot']
        #     labels = labels.to(logits.device)

        #     if len(rows_to_keep) == len(labels):
        #         loss = self.loss(logits, labels)
        #     elif len(rows_to_keep) == 0:
        #         loss = 0
        #     else:
        #         logits = logits[rows_to_keep]
        #         labels = labels[rows_to_keep]
        #         loss = self.loss(logits, labels)

        #     loss *= self.alpha
        #     ldict = {'context_loss': loss}
        #     return probs, ldict
        # else: 
        #     return probs, {}
