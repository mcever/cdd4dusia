import math
import sys
sys.path.append("..")

import os
import time
import pickle
import torch
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import average_precision_score as aps

import torchvision.models.detection.mask_rcnn
from torchvision.ops import boxes as box_ops

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import cdd_utils
import imageio
import bbox_visualizer as bbv

from cdd_utils.vis_cvat_annos import draw_fasterrcnn_boxes
from torchvision import transforms
import matplotlib.pyplot as plt

lab_to_spec = {0: 'bg', 1: 'fpu', 2: 'gg', 3: 'sl'}
lab_2_name = {0: "background", 1: 'fragile pink urchin', 2: 'gray gorgonian', 3: 'squat lobster'}
lab_2_col = {'fragile pink urchin': (0,0,255), 'gray gorgonian': (0,255,0), 'squat lobster': (255,0,0)}

def vis_boxes_w_score(visd, boxes, labels, scores=None, use_name=True, use_score=False, use_ind=False):
    '''for figures and visualization against gt'''
    if len(boxes.shape) == 1:
        labels = np.expand_dims(labels, 0)
        boxes = np.expand_dims(boxes, 0)
        if scores is not None:
            scores = np.expand_dims(scores, 0)

    names = [lab_2_name[lab] for lab in np.asarray(labels)]
    box_labels = []
    for j in reversed(range(len(boxes))):
        if scores is not None:
            if scores[j] < 0.2:
                continue
        box = boxes[j]
        col = lab_2_col[names[j]]
        visd = bbv.draw_rectangle(visd, box, col)

        if use_score:
            assert(scores is not None)

        if use_score and use_name:
            box_label = '{}: {:.2f}'.format(names[j], scores[j])
        elif use_name:
            box_label = names[j]
        elif use_score:
            box_label = '{:.2f}'.format(scores[j])
        else:
            box_label = ''

        if use_ind:
            box_label = '({})'.format(j) + box_label
            
        if box_label != '':
            visd = bbv.add_label(visd, box_label, box)
        box_labels.append(box_label)

    return visd

def vis_boxes(img, boxes, labels, fid='', sv_dir='figs'):
    ''' for debugging '''

    # img = inverse_normalize(img, mean, std)
    if type(img) != np.ndarray:
        t = transforms.ToPILImage()
        img = t(img.cpu()) #now a pil image in rgb

    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    frame, _ = draw_fasterrcnn_boxes(np.array(img), boxes, labels)
    
    plt.imshow(frame)
    plt.title(fid)
    vid, fnum = fid.split('_')
    os.makedirs('{}/{}'.format(sv_dir, vid), exist_ok=True)
    plt.savefig('{}/{}/{}.png'.format(sv_dir, vid, fnum), dpi=500)
    # plt.show()

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = cdd_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', cdd_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if model.ctx_branch is not None:
        use_crb = epoch >= model.ctx_branch.start_epoch
    else:
        use_crb = False

    nimgs = 0

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = cdd_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        assert(len(targets) == len(images))
        '''
        vid = data_loader.dataset.id_lst[int(targets[0]['id'][0])]
        print('0: ', vid)
        '''

        vis_debug = False
        if vis_debug:
            sv_dir = 'figs_topoff'
            for i in range(len(images)):
                vid = data_loader.dataset.id_lst[int(targets[i]['id'][0])]
                vis_boxes(images[i], targets[i]['boxes'], targets[i]['labels'], vid, sv_dir)
                nimgs += 1
            if nimgs > 2500:
                sys.exit()

        loss_dict = model(images, targets, use_crb=use_crb)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = cdd_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, use_crb=False, vis_dir=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = cdd_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    hab_prediction_on = False
    hab_labs = []
    h_preds = []
    img_counter = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs, hab_preds = model(images, use_crb=use_crb)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        if vis_dir is not None:
            use_name = False
            use_score = False
            use_ind = True
            for i in range(len(images)):
                # _, tar = data_loader.dataset[img_counter+i]
                id_lst_idx = targets[i]['image_id'].item()
                im_id = data_loader.dataset.id_lst[id_lst_idx]
                im_path = os.path.join(data_loader.dataset.image_home, "{}.npy".format(im_id))
                np_im = np.load(im_path)
                h, _, _ = np_im.shape
                h_off = int(0.5*h)
                im_crop = np_im[h_off:, :,:]

                out_dict = outputs[i]

                keep = box_ops.batched_nms(out_dict['boxes'].cpu(), out_dict['scores'].cpu(), out_dict['labels'].cpu(), 0.3)
                
                boxes = np.array(out_dict['boxes'].cpu(), dtype=int)[keep]
                box_feats_pre = np.array(out_dict['box_features_pre'].cpu(), dtype=int)[keep]
                box_feats_post = np.array(out_dict['box_features_post'].cpu(), dtype=int)[keep]
                scores = np.array(out_dict['scores'].cpu())[keep]
                labels = np.array(out_dict['labels'].cpu().numpy())[keep]

                im_w_preds = vis_boxes_w_score(im_crop.copy(), boxes, labels, scores, use_name, use_score, use_ind)

                gt_boxes = np.array(targets[i]['boxes'].cpu(), dtype=int)
                gt_labels = targets[i]['labels'].cpu().numpy()
                im_w_gts = vis_boxes_w_score(im_crop.copy(), gt_boxes, gt_labels, None, use_name, False, use_ind)

                vid, fnum = im_id.split('_')
                os.makedirs('{}/{}/pngs'.format(vis_dir, vid), exist_ok=True)
                os.makedirs('{}/{}/pickles'.format(vis_dir, vid), exist_ok=True)
                # imageio.imwrite('{}/{}/{}.png'.format(vis_dir, vid, fnum))
                np_im[:h_off,:,:] = im_w_preds
                np_im[h_off:,:,:] = im_w_gts
                imageio.imwrite('{}/{}/pngs/{}_{}.png'.format(vis_dir, vid, vid, fnum), np_im)

                if use_ind:
                    with open('{}/{}/pickles/{}_{}.p'.format(vis_dir, vid, vid, fnum), 'wb') as f:
                        to_dump = (boxes, box_feats_pre, box_feats_post, scores, labels)
                        pickle.dump(to_dump, f)

        if hab_preds is not None:
            hab_prediction_on = True
            APs = []
            for i,target in enumerate(targets):
                htar = target['habs_hot'].numpy()
                if np.all(htar == np.zeros(htar.shape)):
                    # TODO fix ap calc
                    # for now take out 0 vecs
                    continue
                hab_labs.append(htar)
                h_preds.append(hab_preds[i].cpu().numpy())

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        img_counter += len(images)

    if hab_prediction_on:
        #TODO fix ap calc
        # for now, just take out sand
        hab_mse = mse(hab_labs, h_preds)
        hab_labs = np.array(hab_labs)[:, :-1]
        h_preds = np.array(h_preds)[:, :-1]
        # for each substrate, compute ap
        h_aps = []
        for i in range(hab_labs.shape[1]):
            h_aps.append(aps(hab_labs[:,i], h_preds[:,i]))
        print('per class context APs: {}'.format(h_aps))
        print('context mAP: {}'.format(np.mean(h_aps)))
        print('Context MSE: {}'.format(hab_mse))
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_str = 'Averaged stats: ' + str(metric_logger)
    print(metric_str)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    coco_evaluator.metric_str = metric_str
    torch.set_num_threads(n_threads)

    cebb = coco_evaluator.coco_eval['bbox']
    tps = cebb.eval['all_tps'][0,:,0,2]
    fps = cebb.eval['all_fps'][0,:,0,2]
    pos = cebb.eval['all_pos'][0,:,0,2]

    tpfpStr = ' TP/FP/Pos @ [IoU=0.5 | area=all | maxDets=100]\n\tTPs: {} {}\n\tFPs: {} {}\n\tPos: {} {}'.format(tps, np.sum(tps), fps, np.sum(fps), pos, np.sum(pos))
    print(tpfpStr)

    return coco_evaluator
