from __future__ import print_function, division
import os
import pickle
import random
import sys
from cdd_utils.Annos import Annos as Annos
from cdd_utils.process_cvat_xml import process_cvat_xml
from cdd_utils.translate_boxes import translate_boxes
from cdd_utils.cvat_boxes_to_fasterrcnn import cvat_boxes_to_fasterrcnn
from cdd_utils.cvat_boxes_to_fasterrcnn import cvat_boxes_to_BBs
from cdd_utils.cvat_boxes_to_fasterrcnn import BBOI_to_fasterrcnn
sys.modules['Annos'] = sys.modules['cdd_utils.Annos'] # for pickle compatibility

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


class MareSpeciesDataset(Dataset):
    """PyTorch Dataset for MARE's Oceana videos, to be represented as frames
       includes invertebrate counts in frames"""

    def __init__(self, id_lst_pth, pickle_home, image_home, mode, transforms, set_split, ia=None, cvat_xml_home=None, aug_pct=0.5, mix_pad_pct=0.3, crop_mode='top_off', count_offset=-15):
        self.image_home = image_home
        self.cvat_xml_home = cvat_xml_home
        self.pickle_home = pickle_home
        self.transforms = transforms
        self.ia = ia
        self.aug_pct = aug_pct
        if ia is not None and transforms is not None:
            print('warning: ia is not None and transforms is not None')
        self.crop_mode = crop_mode
        self.count_offset = count_offset
        self.set_split = set_split


        # self.species = ['bg', 'fragile pink urchin', 'gray gorgonian', 'squat lobster']
        self.species = ['aabg', 'fragile pink urchin', 'gray gorgonian', 'squat lobster', 'yellow gorgonian', 'white slipper sea cucumber', 'ui laced sponge', 'basket star', 'white spine sea cucumber', 'long legged sunflower star', 'red swiftia gorgonian']
        self.species.sort()
        print('ordered species are : {}'.format(self.species))
        self.substrates = ['boulder', 'cobble', 'mud', 'rock', 'sand']
        self.substrates.sort()
        print('ordered substrates are : {}'.format(self.substrates))
        with open(id_lst_pth, 'r') as f:
            lines = f.readlines()
            id_lst = [l.strip() for l in lines]
        self.id_lst = id_lst
        self.id_to_cvat_annos = {}

        self.mix_pad_pct = mix_pad_pct

        self.vid_to_habs = {}
        self.vid_to_specs = {}
        for fid in id_lst:
            vid, fnum = fid.split('_')
            if vid not in self.vid_to_habs:
                with open(os.path.join(pickle_home, vid + '_habs.p'), 'rb') as f: 
                    hannos = pickle.load(f)
                self.vid_to_habs[vid] = hannos
            if vid not in self.vid_to_specs:
                with open(os.path.join(pickle_home, vid + '_specs.p'), 'rb') as f: 
                    specs = pickle.load(f)
                self.vid_to_specs[vid] = specs

        if 'count' == mode:
            self.mode = 'count'
        elif 'full' == mode:
            self.mode = 'full'
            assert(self.cvat_xml_home is not None)
        elif 'mixed' == mode:
            self.mode = 'mixed'
        else:
            print('mode must be count, mixed, or full')
            sys.exit()
        return
 
    def get_height_and_width(self,idx):
        # all these frames are from 1080p video 1080,1920
        # if the top is cropped off, it's 540
        # TODO: figure out what this is about
        return 540,1920

    def __getitem__(self, idx):
        ide = self.id_lst[idx]
        orig_id, frame = ide.split('_')
        count_only = False
        if frame[0] == 'c':
            # this is a count only frame
            count_only = True
            frame = frame[1:]

        hannos = self.vid_to_habs[orig_id][int(frame)]
        hannos = list(hannos)
        hannos.sort()
        hannos = [h.lower() for h in hannos]
        habs_hot = torch.zeros(len(self.substrates))
        for h in hannos:
            # if h not in self.substrates:
            #     print(h, ' not in substrates')
            if h in self.substrates:
                habs_hot[self.substrates.index(h)] = 1

        count_hot = torch.zeros(len(self.species))
        try:
            found_name = False
            count_annos = self.vid_to_specs[orig_id][int(frame)-self.count_offset]
            for i, name in enumerate(count_annos['common_name']):
                if name.lower() in self.species:
                    ind = self.species.index(name.lower())
                    count_hot[ind] = count_annos['count'][i]
                    found_name = True
            if not found_name:
                # print('no relevant name for ', ide)
                pass
            else:
                # print('count_hot for ', ide, ' ', count_hot)
                pass
        except KeyError:
            # print('no specs for ', ide)
            pass

        arr = np.load(os.path.join(self.image_home, ide + '.npy'))

        # load the image, crop, and apply transforms??
        h,w,c = arr.shape
        zero_padded = 0
        want_to_double = False
        want_to_zero_pad = False
        if self.crop_mode == 'top_off':
            PCT_OFF = 0.5
            h_off = int(PCT_OFF*h)
            w_off = 0
            h2 = None # only used when padding with zeros
            arr_crop = arr[h_off:, :,:]
            if count_only and (want_to_zero_pad or want_to_double):
                h2 = int(0.5*h_off)
                if want_to_zero_pad:
                    # replace with some zeros
                    arr_crop[:h2,:,:] = np.zeros(arr_crop[h2:,:,:].shape)
                    zero_padded = 1
                if want_to_double:
                    arr_crop[:h2,:,:] = arr_crop[h2:,:,:]
                    zero_padded = 2
                    count_hot *= 2

        else:
            print('not cropping mode: {}'.format(self.crop_mode))
            arr_crop = arr


        # need to get boxes for ful
        # /partially box supervised task
        if 'full' in self.set_split:
            shapes_only = True
        else:
            shapes_only = False

        if self.set_split == 'tkfvf':
            shapes_only = False
            if ide in self.valfull:
                shapes_only = True
        

        if orig_id not in self.id_to_cvat_annos:
            fnum_to_boxes = process_cvat_xml(os.path.join(self.cvat_xml_home, orig_id+'.xml'), shapes_only)
            self.id_to_cvat_annos[orig_id] = fnum_to_boxes
        else:
            fnum_to_boxes = self.id_to_cvat_annos[orig_id]
        one_hot = torch.zeros(len(self.species))
        imid = torch.tensor([idx])

        if not count_only:
            try:
                boxes = fnum_to_boxes[int(frame)]
                if self.crop_mode == 'top_off':
                    if self.mode == 'mixed' and random.random() < self.mix_pad_pct:
                        h2 = int(0.5*h_off)
                        h2_boxes = translate_boxes(h_off, w_off, boxes, h2)
                        if len(h2_boxes) > 1:
                            arr_crop[:h2,:,:] = np.zeros(arr_crop[h2:,:,:].shape)
                            zero_padded = 1
                            boxes = h2_boxes
                        else:
                          boxes = translate_boxes(h_off, w_off, boxes)  
                    else:
                        boxes = translate_boxes(h_off, w_off, boxes)


                if self.ia is not None and torch.rand(1).item() < self.aug_pct:
                    # if we want to do imgaug augmentations, need to edit boxes now
                    # some pct of the time, do no augmentations
                    BoundingBoxes = cvat_boxes_to_BBs(boxes, self.species)
                    BBOI = BoundingBoxesOnImage(BoundingBoxes, shape=arr_crop.shape)
                    augd, BBOId = self.ia(image=arr_crop, bounding_boxes=BBOI)
                    BBOId = BBOId.remove_out_of_image().clip_out_of_image()
                    aug_boxes, aug_labels = BBOI_to_fasterrcnn(BBOId)
                    if len(aug_boxes) > 0:
                        # ensure we didn't clip all the boxes
                        boxes, labels = aug_boxes, aug_labels
                        arr_crop = augd
                        # print('augs performed')
                    else:
                        # this function also filters out boxes from non-self.species
                        boxes, labels = cvat_boxes_to_fasterrcnn(boxes, self.species)
                else:
                    boxes, labels = cvat_boxes_to_fasterrcnn(boxes, self.species)

                for j,lab in enumerate(labels):
                    one_hot[lab] = 1
                    box = boxes[j]
                    xtl, ytl, xbr, ybr = box
                    if ybr > (int(PCT_OFF*h)- 5):
                        # this box is very close to bottom of frame
                        count_hot[lab] += 1
                    # count_hot[lab] += 1
                iscrowd = torch.zeros(len(boxes)) # nonsense to appease coco eval stuff
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                if len(boxes) > 0:
                    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                    target = {'one_hot': one_hot, 'count_hot': count_hot, 'id': imid, 'boxes': boxes, 'labels': labels, 'image_id': imid, 'area': area, 'iscrowd': iscrowd, 'habs_hot': habs_hot, 'count_only': torch.tensor(count_only, dtype=torch.int64), 'is_padded': torch.tensor(zero_padded, dtype=torch.int64)}

                else:
                    iscrowd = torch.zeros(0)
                    area = torch.tensor(0.0)
                    
                    print('no boxes after crop on {}'.format(ide))
                    target = {'one_hot': one_hot, 'count_hot': count_hot, 'id': imid, 'boxes': torch.zeros((0, 4), dtype=torch.float32), 'area': area, 'iscrowd': iscrowd,
                              'labels': torch.zeros((1, 1), dtype=torch.int64), 'habs_hot': habs_hot, 'image_id': imid, 
                              'count_only': torch.tensor(count_only, dtype=torch.int64), 'is_padded': torch.tensor(zero_padded, dtype=torch.int64)}

            except KeyError:
                area = torch.tensor(0.0)
                iscrowd = torch.zeros(0)
                print('KeyError: no boxes on {}'.format(ide))
                target = {'one_hot': one_hot, 'count_hot': count_hot, 'id': imid, 'area': area, 'iscrowd': iscrowd, 'boxes': torch.zeros((0, 4), dtype=torch.float32), 'labels': torch.zeros((1, 1), dtype=torch.int64), 'habs_hot': habs_hot, 'image_id': imid, 'count_only': torch.tensor(count_only, dtype=torch.int64), 'is_padded': torch.tensor(zero_padded, dtype=torch.int64)}
        else:
            # get counts / labels
            target = {'one_hot': one_hot, 'count_hot': count_hot, 'id': imid, 'boxes': torch.zeros((0, 4), dtype=torch.float32), 'labels': torch.zeros((1, 1), dtype=torch.int64), 'habs_hot': habs_hot, 'image_id': imid, 'count_only': torch.tensor(count_only, dtype=torch.int64), 'is_padded': torch.tensor(zero_padded, dtype=torch.int64)}

        # use transforms
        if self.ia:
            # can i give this tensors
            pass
        if self.transforms:
            img = Image.fromarray(arr_crop)
            img, target = self.transforms(img, target)
        else:
            # print('no transforms')
            img = torch.tensor(arr_crop)

        return img, target


    def __len__(self):
        return len(self.id_lst)
