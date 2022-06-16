'''
if your run times out, you can just try running again as
reader may get hung, otherwise
TIMEOUT_MAX may need to be increased if your disk is slow
'''

from process_cvat_xml import process_cvat_xml
from timeout import timeout
from translate_boxes import translate_boxes
import os
import time
import yaml

import numpy as np
import av
from PIL import Image

with open('../config.yaml') as f:
        args = yaml.load(f, Loader=yaml.Loader)

data_path = args['data']['data_path']
set_name = args['data']['set_name']
cvat_xml_home = os.path.join(data_path, 'cvat_annos')
VID_HOME = args['data']['vids_path']
FRAME_HOME = os.path.join(data_path, 'frames')
SPLITS = [args['data']['trainsplit'], args['data']['valsplit'], args['data']['testsplit']]

TIMEOUT_MAX = 10 # in seconds
h_off = 540 # half 
w_off = 0

last_vid = ''
for split in SPLITS:
    id_lst = os.path.join(data_path, 'idd_lsts','{}_{}_cvat_frames.txt'.format(split, set_name))
    assert(os.path.isfile(id_lst))
    split_home = os.path.join(FRAME_HOME, '{}_{}'.format(split, set_name))
    os.makedirs(split_home, exist_ok=True)

    with open(id_lst, 'r') as f:
        ids = f.readlines()
        ids = [ide.strip() for ide in ids]

    vid_to_fnums = {}
    for ide in ids:
        vid, fnum = ide.split('_')
        if vid not in vid_to_fnums:
            vid_to_fnums[vid] = [int(fnum)]
        else:
            vid_to_fnums[vid].append(int(fnum))

    split_write = ''
    for vid,fnums in vid_to_fnums.items():
        saved_fnums = []
        fnums = sorted(fnums)
        vpath = os.path.join(VID_HOME, vid+'.mp4')
        container = av.open(vpath)

        # see if there are any unsaved in this vid
        # maybe we can skip the whole video
        never_saved_fnums = []
        for fnum in fnums:
            sv_name = os.path.join(split_home, '{}_{:07d}'.format(vid, fnum))
            if os.path.isfile(sv_name+'.npy'):
                saved_fnums.append('{}_{:07d}'.format(vid, fnum))
            else:
                never_saved_fnums.append(fnum)
                break
        if len(never_saved_fnums) == 0:
            print('skipping ', vid, ' all frames saved')
            continue


        print('-------- opening {} at {} --------'.format(vid, time.ctime()))
        for fnum, frame in enumerate(container.decode(video=0)):
            if fnum not in fnums:
                continue

            sv_name = os.path.join(split_home, '{}_{:07d}'.format(vid, fnum))
            if os.path.isfile(sv_name+'.npy'):
                saved_fnums.append('{}_{:07d}'.format(vid, fnum))
                continue

            print('{}: saving {} frame {}'.format(time.ctime(), vid, fnum))
            img = frame.to_image()
            arr = np.array(img)
            np.save(sv_name, arr)
            saved_fnums.append('{}_{:07d}'.format(vid, fnum))

        container.close()
