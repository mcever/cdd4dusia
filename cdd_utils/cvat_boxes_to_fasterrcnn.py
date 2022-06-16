from imgaug.augmentables.bbs import BoundingBox

def cvat_boxes_to_fasterrcnn(boxes, specs):
    ''' also filters out irrelvant species boxes'''
    new_boxes = []
    new_labels = []

    for i,bdict in boxes.items():
        lab = bdict['label']
        cname = bdict['label'].lower()
        if cname in specs:
            ind = specs.index(cname)
        else:
            # remove this box...
            continue

        assert(bdict['xtl'] < bdict['xbr'])
        assert(bdict['ytl'] < bdict['ybr'])

        box = [bdict['xtl'], bdict['ytl'], bdict['xbr'], bdict['ybr']]
        new_boxes.append(box)
        new_labels.append(ind)

    return new_boxes, new_labels

def cvat_boxes_to_BBs(boxes, specs):
    BoundingBoxes = []
    for tid, box in boxes.items():
        box['label'] = box['label'].lower()
        if box['label'] in specs:
            lab = specs.index(box['label'])
            BoundingBoxes.append(BoundingBox(x1=box['xtl'], y1=box['ytl'],
                x2=box['xbr'], y2=box['ybr'], label=lab))
    return BoundingBoxes


def BBOI_to_fasterrcnn(boxes):
    new_boxes = []
    new_labels = []

    for box in boxes.bounding_boxes:
        new_boxes.append([box.x1, box.y1, box.x2, box.y2])
        new_labels.append(box.label)

    return new_boxes, new_labels
