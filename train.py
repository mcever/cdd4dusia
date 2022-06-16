r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""

import datetime
import os
import sys
sys.path.append('training')
import time
import pickle
import yaml
import random

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa 

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection

from cdd_utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from training.engine import train_one_epoch, evaluate

import training.presets
import cdd_utils.cdd_utils as cdd_utils

import mymodels
from dataset.MareSpeciesDataset import MareSpeciesDataset
from cdd_utils import read_train_pickles

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataset(dpath, name, image_set, transform, data_path, short=False, use_ia=False, set_name='conspec', aug_pct=0.5, ann_type='cvat', mix_pad_pct=0.3):

    if ann_type not in ['cvat', 'count', 'mixed']:
        print('invalid ann_type ', ann_type)
        sys.exit() 

    my_mode = 'full'

    split = image_set

    id_lst_pth = os.path.join(dpath, 'idd_lsts', '{}_{}_{}_frames.txt'.format(split, set_name, ann_type))
    fpath = os.path.join(dpath, 'frames', '{}_{}'.format(split, set_name))
    
    if short:
        print('using {} short'.format(split))
        id_lst_pth = os.path.join(dpath, 'idd_lsts', '{}_{}_cvat_frames_short.txt'.format(split, set_name))

    print(id_lst_pth)
    # should probably make these arguments
    
    ppath = os.path.join(dpath, 'pickle_annos')
    cvpath = os.path.join(dpath, 'cvat_annos')

    if use_ia:
        ia.seed(0)
        my_augs = [ 
            iaa.MultiplyAndAddToBrightness(mul=(0.5,1.5), add=(-30,30)),
            iaa.AdditiveGaussianNoise(scale=(0, 25)), 
            iaa.GaussianBlur(sigma=(0,2.0)),
            iaa.Affine(scale=(0.9,1.1), translate_percent=(0, 0.05)),
            iaa.GammaContrast((0.7, 1.5)),
            iaa.HorizontalFlip(1)]
        # seq = iaa.Sequential(my_augs, random_order=True)
        aug_factory= iaa.SomeOf(3, my_augs)
    else:
        aug_factory = None

    dset = MareSpeciesDataset(id_lst_pth, ppath, fpath, my_mode, transform, split, aug_factory, cvpath, aug_pct, mix_pad_pct)

    return dset, len(dset.species)

def get_transform(train):
    return training.presets.DetectionPresetTrain() if train else training.presets.DetectionPresetEval()


def main(args):

    seed = 1
    # torch.use_deterministic_algorithms(True) # , in beta....
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # ia.seed is done in get_dataset if using imgaugs

    if args['distributed']:
        cdd_utils.init_distributed_mode(args)
    else:
        print('not using distributed mode')
    print(args)

    device = torch.device(args['device'])

    # Data loading code
    print("Loading data")

    set_name = args['data']['set_name']
    dataset, num_classes = get_dataset(args['data']['data_path'], args['data']['dataset'], args['data']['trainsplit'], get_transform(train=False), args['data']['data_path'], args['short'], args['data']['use_ia'], set_name, args['data']['aug_pct'], args['data']['traintype'], args['data']['mix_pad_pct'])
    # valfull is the fully annotated version
    dataset_test, _ = get_dataset(args['data']['data_path'], args['data']['dataset'], args['data']['valsplit'], get_transform(train=False), args['data']['data_path'], args['short'], set_name=set_name)

    want_to_val_on_testsplit = False
    if want_to_val_on_testsplit:
        dataset_test2, _ = get_dataset(args['data']['data_path'], args['data']['dataset'], args['data']['testsplit'], get_transform(train=False), args['data']['data_path'], args['short'], set_name=set_name)
    else:
        dataset_test2 = None
        print('not doing val on testsplit')

    print("Creating data loaders")
    if args['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        if want_to_val_on_testsplit:
            test_sampler2 = torch.utils.data.distributed.DistributedSampler(dataset_test2)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        if want_to_val_on_testsplit:
            test_sampler2 = torch.utils.data.SequentialSampler(dataset_test2)

    if args['data']['aspect_ratio_group_factor'] >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args['data']['aspect_ratio_group_factor'])
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args['batch_size'])
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args['batch_size'], drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args['workers'],
        collate_fn=cdd_utils.collate_fn,
        worker_init_fn=seed_worker)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args['workers'],
        collate_fn=cdd_utils.collate_fn,
        worker_init_fn=seed_worker)

    if want_to_val_on_testsplit:
        data_loader_test2 = torch.utils.data.DataLoader(
            dataset_test2, batch_size=1,
            sampler=test_sampler2, num_workers=args['workers'],
            collate_fn=cdd_utils.collate_fn,
            worker_init_fn=seed_worker)
    else:
        data_loader_test2 = None
    test_metric_strs = []
    test_cebbs = []

    kwargs = {}
    kwargs['use_hpb'] = args['model']['context_branch']['use_hpb']
    kwargs['hpb_alpha'] = args['model']['context_branch']['hpb_alpha']
    kwargs['use_crb'] = args['model']['context_branch']['use_crb']
    kwargs['crb_alpha'] = args['model']['context_branch']['crb_alpha']
    kwargs['crb_start_ep'] = args['model']['context_branch']['crb_start_epoch']
    kwargs['global_to_predictor'] = args['model']['context_branch']['global_to_predictor']
    kwargs['global_feats_for_logits'] = args['model']['context_branch']['global_feats_for_logits']
    kwargs['use_hpb_ctx_rep_only'] = args['model']['context_branch']['use_hpb_ctx_rep_only']
    kwargs['count_cls_loss_scale'] = args['data']['count_cls_loss_scale']
    kwargs['global_rep_scalar'] = args['model']['hyper_params']['global_rep_scalar']
    
    '''
    if "rcnn" in args['model']['model']:
        if args['model']['rpn_score_thresh'] is not None:
            kwargs["rpn_score_thresh"] = args['model']['rpn_score_thresh']
    '''

    print("Creating model")
    # model = torchvision.models.detection.__dict__[args['model']['model']](num_classes=num_classes, pretrained=args['model']['pretrained'], **kwargs)
    model = mymodels.__dict__[args['model']['model']](num_classes=num_classes, 
            pretrained=args['model']['pretrained'], roi_drop_pct=args['data']['roi_drop_pct'], **kwargs)
    model.to(device)

    model_without_ddp = model
    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args['model']['hyper_params']['lr'], momentum=args['model']['hyper_params']['momentum'], weight_decay=args['model']['hyper_params']['weight_decay'])

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['model']['hyper_params']['lr_step_size'], gamma=args['model']['hyper_params']['lr_gamma'])
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['model']['hyper_params']['lr_steps'], gamma=args['model']['hyper_params']['lr_gamma'])
    print('using ExponentialLR')
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args['model']['hyper_params']['lr_gamma'])

    if args['model']['resume']:
        checkpoint = torch.load(args['model']['resume'], map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args['model']['start_epoch'] = checkpoint['epoch'] + 1

        model_num = int(checkpoint['epoch'])
        with open(os.path.join(args['output_dir'], 'pickles/{}.p'.format(model_num))) as f:
            pickd = pickle.load(f)
        train_loss_strs, train_metric_strs, train_cebbs, val_metric_strs, val_cebbs, lrs = pickd
    else:
        train_loss_strs = []
        val_metric_strs = []
        val_cebbs = []
        train_metric_strs = []
        train_cebbs = []
        lrs = []

    if args['model']['start_weights'] != '':
        sd_pth = args['model']['start_weights']
        print("loading args['model']['start_weights'] {}".format(sd_pth))
        model.load_state_dict(torch.load(sd_pth)['model'])

    if args["model"]["test_only_weights"] != '':
        sd_pth = args["model"]["test_only_weights"]
        print('loading args["model"]["test_only_weights"] {}'.format(sd_pth))
        model.load_state_dict(torch.load(sd_pth)['model'])
        print('Running test_only on {}'.format(data_loader_test.dataset.set_split))
        out_dir = None
        if args['save_detections']:
            out_dir = args['output_dir']
        evaluate(model, data_loader_test, device=device, use_crb=args['model']['context_branch']['use_crb'], vis_dir=out_dir)
        return

    print("Start training")
    best_val_AP = -1.0
    last_val_model_path = ''
    best_train_AP = -1.0
    train_AP = -1.0
    last_train_model_path = ''
    save_now = False
    start_time = time.time()
    for epoch in range(args['model']['start_epoch'], args['epochs']):
        if args['distributed']:
            train_sampler.set_epoch(epoch)

        if model.ctx_branch:
            use_crb = epoch >= model.ctx_branch.start_epoch
        else:
            use_crb = False
        
        train_metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, args['print_freq'])
        train_loss_strs.append(train_metric_logger.log_msgs)
        lrs.append(lr_scheduler.get_last_lr())
        lr_scheduler.step()

        # evaluate after every epoch
        train_eval_freq = 1
        want_to_val_on_train = True
        if not (epoch % train_eval_freq) and want_to_val_on_train:
            print('EVALUATING TRAINSPLIT')
            train_coco_evaluator = evaluate(model, data_loader, device=device, use_crb=use_crb) 
            train_metric_strs.append(train_coco_evaluator.metric_str)
            train_cebbs.append(train_coco_evaluator.coco_eval['bbox'])
            train_AP = train_coco_evaluator.coco_eval['bbox'].stats[1]
        else:
            print('NOT EVALUATING TRAINSPLIT')

        print('EVALUATING VALSPLIT')
        val_coco_evaluator = evaluate(model, data_loader_test, device=device, use_crb=use_crb)
        val_metric_strs.append(val_coco_evaluator.metric_str)
        val_cebbs.append(val_coco_evaluator.coco_eval['bbox'])

        if want_to_val_on_testsplit:
            print('EVALUATING TESTSPLIT')
            test_coco_evaluator = evaluate(model, data_loader_test2, device=device, use_crb=use_crb)
            test_metric_strs.append(test_coco_evaluator.metric_str)
            test_cebbs.append(test_coco_evaluator.coco_eval['bbox'])
            test_AP = test_coco_evaluator.coco_eval['bbox'].stats[1]
        else:
            test_AP = -1.0

        val_AP = val_coco_evaluator.coco_eval['bbox'].stats[1]
        
        print('train: {}, val: {}, test: {} AP'.format(train_AP, val_AP, test_AP))
        
        if val_AP > best_val_AP:
            save_now = True
            best_val_AP = val_AP
            print('best val AP so far achieved {}'.format(best_val_AP))
        if train_AP > best_train_AP:
            # save_now = True # do you care about best train weights? not anymore, save space
            best_train_AP = train_AP
            print('best train AP so far achieved {}'.format(best_train_AP))

        if args['output_dir'] and save_now:
            cdd_utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args['output_dir'], 'model_{}.pth'.format(epoch)))

        # only want to keep the most recent best weights
        # and remove the old ones to save storage
        if val_AP > best_val_AP:
            if os.path.isfile(last_val_model_path):
                os.remove(last_val_model_path)
            last_val_model_path = os.path.join(args['output_dir'], 'model_{}.pth'.format(epoch))
        if train_AP > best_train_AP:
            if os.path.isfile(last_train_model_path):
                os.remove(last_train_model_path)
            last_train_model_path = os.path.join(args['output_dir'], 'model_{}.pth'.format(epoch))

        save_now = False

        if args['output_dir']:
            pickle_dir = os.path.join(args['output_dir'], 'pickles')
            if data_loader_test2 is None:
                to_pickle = (train_loss_strs, train_metric_strs, train_cebbs, 
                        val_metric_strs, val_cebbs, lrs)
            else:
                to_pickle = (train_loss_strs, train_metric_strs, train_cebbs, 
                        val_metric_strs, val_cebbs, test_metric_strs, 
                        test_cebbs, lrs)

            with open(os.path.join(pickle_dir, '{}.p'.format(epoch)), 'wb') as f:
                pickle.dump(to_pickle, f)

            if epoch > 1:
                to_del_ep = epoch - 2 
                if to_del_ep%10 != 0:
                    todp = os.path.join(pickle_dir, '{}.p'.format(to_del_ep))
                    if os.path.isfile(todp):
                        os.remove(todp)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    print('now calling read_train_pickles to summarize')
    read_train_pickles.main(args['output_dir'], epoch, False)

    end_txt_pth = os.path.join(args['output_dir'], 'finished.txt')
    text = 'done'
    with open(end_txt_pth, 'w') as f:
        f.write(text)

if __name__ == "__main__":

    with open('config.yaml') as f:
        args = yaml.load(f, Loader=yaml.Loader)

    if args['output_dir']:
        cdd_utils.mkdir(args['output_dir'])
        cdd_utils.mkdir(os.path.join(args['output_dir'], 'pickles'))
        
    main(args)
