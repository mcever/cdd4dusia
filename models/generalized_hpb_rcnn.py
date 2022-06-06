# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import numpy as np
import torch.nn.functional as F


class GeneralizedHPBRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform, hpb=None, ctx_branch=None):
        super(GeneralizedHPBRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.hpb = hpb
        self.ctx_branch = ctx_branch
        self.SCORE_THRESH = 0.8
        if self.ctx_branch is not None:
            assert(hpb is not None, "ctx_branch requires hpb")
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections, hab_preds):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections, hab_preds

    def forward(self, images, targets=None, use_crb=False):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        batch_has_boxes = False
        batch_has_counts = False

        if targets:
            for t in targets:
                if len(t['boxes']) > 0:
                    batch_has_boxes = True
                    break

            for t in targets:
                if sum(t['count_hot']) > 0:
                    batch_has_counts = True
                    break

            if batch_has_counts:
                pass

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        if self.hpb is not None:
            hab_preds, ctx_rep, hab_losses = self.hpb(features, targets)
            # hab_preds = hab_preds.unsqueeze(1).unsqueeze(3).repeat((1, 256, 1, 2))
            # features['context'] = hab_preds
        else:
            hab_preds = None
            ctx_rep = None

        ctx_branch_losses = None
        proposals, proposal_losses = self.rpn(images, features, targets)
        need_dets = use_crb or batch_has_counts
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets, need_dets, ctx_rep)

        count_losses = {}
        if targets and False:
            # compute count loss, TO BE REMOVED
            if len(detections) > 0:
                pred_count_hot = torch.zeros(len(detections), targets[0]['count_hot'].shape[0])
                gt_count_hot = torch.zeros(len(detections), targets[0]['count_hot'].shape[0])
                for i,d in enumerate(detections): 
                    if len(targets[i]['boxes']) > 0:
                        continue
                    gt_count_hot[i] = targets[i]['count_hot']

                    # score_lst = list(np.around(d['scores'].cpu().detach().numpy(),2))
                    # print(score_lst)

                    for j,score in enumerate(d['scores']):
                        if score > self.SCORE_THRESH:
                            pred_count_hot[i][d['labels'][j]] += 1
                count_loss = F.mse_loss(gt_count_hot, pred_count_hot)*0.0001
                count_losses = {'loss_count': count_loss}
        if use_crb:
            detections, ctx_branch_losses = self.ctx_branch(features, detections, images.image_sizes, hab_preds, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(proposal_losses)
        # if batch_has_boxes:
        losses.update(detector_losses)
        if ctx_branch_losses is not None:
            losses.update(ctx_branch_losses)
        if self.hpb is not None:
            losses.update(hab_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections, hab_preds)
        else:
            return self.eager_outputs(losses, detections, hab_preds)
