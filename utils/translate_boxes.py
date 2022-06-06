'''
translate cvat format dictionary of boxes
for cropped images
throw out boxes of 0 h or w
'''

def translate_boxes(h_off, w_off, boxes, pad_h=None):
    new_boxes = {}
    for ide,bdict in boxes.items():
        xtl = bdict['xtl'] - w_off
        if xtl < 0:
            xtl = 0
        ytl = bdict['ytl'] - h_off
        if ytl < 0:
            ytl = 0
        xbr = bdict['xbr'] - w_off
        if xbr < 0:
            xbr = 0
        ybr = bdict['ybr'] - h_off
        if ybr < 0:
            ybr = 0

        if pad_h is not None:
            if ytl < pad_h:
                ytl = pad_h

        # throw out boxes of 0 height or width
        if xbr - xtl <= 0:
            continue
        if ybr - ytl <= 0:
            continue

        new_box = {}
        new_box['xtl'] = xtl
        new_box['ytl'] = ytl
        new_box['xbr'] = xbr
        new_box['ybr'] = ybr
        new_box['label'] = bdict['label']
        new_box['keyframe'] = bdict['keyframe']

        new_boxes[ide] = new_box

    return new_boxes


