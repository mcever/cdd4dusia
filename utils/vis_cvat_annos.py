'''
draw boxes on frames to ensure 
cvat xml is read correctly
'''

import copy
import psutil
import os
import time
from lxml import etree
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
from .process_cvat_xml import process_cvat_xml


WRITE_HOME = 'write_vids'
MEM_MAX = 20


def draw_fasterrcnn_boxes(frame, boxes, labels, show=False, text_label=None):
    '''
    frame should be acceptable by cv2 (e.g. np array)
    boxes is fasterrcnn style (FloatTensor[N, 4]) [xmin, ymin, xmax, ymax]
    labels is number corresponding to species
    show is bool whether to show or not from here
    '''
    lab_2_name = {0: 'bg', 1: 'fragile pink urchin', 2: 'gray gorgonian', 3: 'squat lobster'}
    lab_2_col = {'fragile pink urchin': (0,0,255), 'gray gorgonian': (0,255,0), 'squat lobster': (255,0,0)}
    boxes = np.array(boxes)
    for i,box in enumerate(boxes):
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        try:
            label = lab_2_name[int(labels[i])]
        except Exception as e:
            import pdb; pdb.set_trace()
            print(e)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), lab_2_col[label])
        if text_label:
            xcord = xmin
            ycord = ymin-10 if ymin-10 > 0 else 0
            cv2.putText(frame, text_label, (xcord, ycord), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lab_2_col[label], 2)


    if show:
        cv2.imshow('box_frame', frame)
        # if hit q, turn off showing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            show = False
    return frame, show

def draw_boxes(frame, boxes, show=False, solid_kf=False, title=''):
    '''
    frame should be acceptable by cv2 (e.g. np array)
    boxes is dict of frameid to cvat box dict
    show is bool whether to show or not from here
    '''
    lab_2_col = {'fragile pink urchin': (0,0,255), 'gray gorgonian': (0,255,0), 'squat lobster': (255,0,0)}
    for trackid,box in boxes.items():
        xtl, ytl, xbr, ybr = int(box['xtl']), int(box['ytl']), int(box['xbr']), int(box['ybr'])
        if xtl == 415: continue
        label = box['label']
        keyf = box['keyframe']
        th = 4
        if solid_kf and keyf:
            th = -1
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), lab_2_col[label], th)

    if show:
        '''
        cv2.imshow('box_frame', frame)
        # if hit q, turn off showing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            show = False
        '''
        plt.imshow(frame)
        plt.title(title)
        plt.show()
    return frame, show

def draw_on_frames_and_save(f2b, v_path, start=0):
    process = psutil.Process(os.getpid())
    cap = cv2.VideoCapture(v_path)
    show = False
    new_vid = []
    the_start = copy.deepcopy(start)
    fnum = start
    # loop to get to starting fnum
    for i in range(fnum):
        cap.read()
    while True:
        # get the next frame
        ret, frame = cap.read()
        if not ret:
            break
        if fnum in f2b:
            # draw the box
            boxes = f2b[fnum]
            add_frame, show = draw_boxes(frame, boxes, show)
        else:
            add_frame = frame

        new_vid.append(add_frame)
        
        # check up on RAM
        if not fnum % 100:
            mem = process.memory_info().rss / 1e9
            if mem > MEM_MAX:
                print('{}: writing . . .'.format(time.ctime()))
                nv = np.array(new_vid)
                nv = nv[:,:,:,[2,1,0]] # BGR to RGB
                imageio.mimwrite(os.path.join(WRITE_HOME, 'writ_early_{}-{}.mp4'.format(the_start, fnum)), nv, fps=30.0)
                break
            print('{}: {} GB'.format(fnum, mem))

        fnum += 1

    cap.release()
    cv2.destroyAllWindows()
    return fnum

def main():
    ide = '00002016080720454000'
    xml_path = '../{}.xml'.format(ide)
    v_path = '/media/ssd2/marine/idd/vids/{}.mp4'.format(ide)
    fnum_to_boxes = process_cvat_xml(xml_path)

    ending_fnum = draw_on_frames_and_save(fnum_to_boxes, v_path)
    print('ended on fnum: {}'.format(ending_fnum))

if __name__ == '__main__':
    # warning: probably don't run this in utils dir
    pass

