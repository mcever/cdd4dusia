'''
ensure all the HOME directories match yours
ensure id_lst is correct
adjust "for split in" line for the splits you're interested in
if your run times out, you can just try running again as
reader may get hung, otherwise
TIMEOUT_MAX may need to be increased if your disk is slow
'''

from .utils.process_cvat_xml import process_cvat_xml
from .utils.timeout import timeout
from .utils.translate_boxes import translate_boxes
import os
import time
import numpy as np
import av
from PIL import Image

cvat_xml_home = 'data/idd/cvat_annos'

set_name = 'auspec' # ifthab, conspec, auspec, etc
TIMEOUT_MAX = 10 # in seconds
h_off = 540 # half 
w_off = 0
VID_HOME = 'data/idd/vids/' # where raw videos are stored
FRAME_HOME = 'data/MARE/cvat/' # where to output frames
# ids = ['00002016080720454000']

last_vid = ''
# for split in ['train', 'val', 'test']:
#for split in ['trainkf', 'valfull', 'testfull']:
for split in ['testfull']:
    id_lst = 'data/idd_lsts/{}_{}_cvat_frames.txt'.format(split, set_name)
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
