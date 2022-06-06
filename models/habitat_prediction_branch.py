import torch
from torch import nn
import torch.nn.functional as F


class HabitatPredictionBranch(torch.nn.Module):

    def __init__(self, in_channels, num_habitats, alpha=0.1, global_to_predictor=False, global_feats_for_logits=False, use_hpb_ctx_rep_only=False):
        super(HabitatPredictionBranch, self).__init__()
        # self.cls_logits = nn.Conv2d(in_channels, num_habitats, kernel_size=1, stride=1)
        self.alpha = alpha
        self.global_to_predictor = global_to_predictor
        self.global_feats_for_logits = global_feats_for_logits

        self.pool_channels = 256
        self.pool_h = 6
        self.pool_w = 21
        self.pool_feature_dim = self.pool_channels * self.pool_w * self.pool_h
        print("POOL FEATURE_DIM_HARD CODED AS {}".format(self.pool_feature_dim))

        self.conv1d_out_channels = 2
        print("HPB CONV1D OUT C: {}".format(self.conv1d_out_channels))
        self.conv1d_kernel_size = 3
        self.conv1d = nn.Conv1d(self.pool_channels, self.conv1d_out_channels, self.conv1d_kernel_size)
        self.conv1d_out_dim = self.conv1d_out_channels * self.pool_w * self.pool_h - 2*self.conv1d_out_channels
        print("HPB CONV1D OUT DIM: {}".format(self.conv1d_out_dim))

        self.use_hpb_ctx_rep_only = use_hpb_ctx_rep_only
        if use_hpb_ctx_rep_only:
            assert(global_to_predictor)
        else:
            self.fc = nn.Linear(in_channels, num_habitats)
            self.loss = nn.BCEWithLogitsLoss()

        '''
        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        '''

    def forward(self, x, targets=None):
        if self.global_to_predictor or self.global_feats_for_logits:
            batch_size, pool_channels, pool_h, pool_w = x['pool'].shape
            assert(pool_channels == self.pool_channels and self.pool_h == pool_h and pool_w == self.pool_w)
            hab_rep = self.conv1d(x['pool'].reshape(batch_size, pool_channels, pool_h * pool_w))
            hab_rep = hab_rep.reshape(batch_size, self.conv1d_out_dim)
            if self.global_feats_for_logits:
                logits = self.fc(hab_rep)
                probs = F.softmax(logits)
        
        if not self.global_to_predictor:
            hab_rep = None

        if not self.global_feats_for_logits and not self.use_hpb_ctx_rep_only:
            feat = x['pool'].reshape(x['pool'].shape[0], -1)
            logits = self.fc(feat)
            probs = F.softmax(logits)
        elif self.use_hpb_ctx_rep_only:
            logits = None
            probs = None


        if targets is not None and logits is not None:
            labels = torch.zeros(logits.shape)
            rows_to_keep = []
            for i,target in enumerate(targets):
                # some targets may have no / bad habitat labels
                # keep only those with habitat labels for loss calc
                if not torch.all(target['habs_hot'] == torch.zeros(logits.shape).to(logits.device)):
                    rows_to_keep.append(i)
                labels[i] = target['habs_hot']
            labels = labels.to(logits.device)

            if len(rows_to_keep) == len(labels):
                loss = self.loss(logits, labels)
            elif len(rows_to_keep) == 0:
                loss = 0
            else:
                logits = logits[rows_to_keep]
                labels = labels[rows_to_keep]
                loss = self.loss(logits, labels)

            loss *= self.alpha
            ldict = {'context_loss': loss}
            return probs, hab_rep, ldict
        else: 
            return probs, hab_rep, {}

'''
        print('FEATURE SHAPE', x.keys())
        import pdb; pdb.set_trace()
        if self.training and targets is None:
                print('hpb in training mode, must pass targets')
        logits = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            print(k, v.shape)

        losses = {}
        if self.training:
            labels = targets['labels']
            losses['cls_loss'] = F.cross_entropy(logits, labels)
        return logits, losses
'''

'''
        
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # type: (List[Tensor])
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


'''
