'''
see process_cvat_xml docstring
'''

import os
from lxml import etree


def process_cvat_xml(xml_file, shapes_only=False, kf_only=False):
    """
    Transforms a single XML in CVAT format to Python readable format
    shapes_only=True will ignore tracks and only return shapes
    shapes_only=False will ignore shapes and return only tracks

    :return: frames dictionary maps frame number to boxes
    """
    cvat_xml = etree.parse(xml_file)

    basename = os.path.splitext( os.path.basename( xml_file ) )[0]

    tracks= cvat_xml.findall( './/track' )
    # Build up a list of each bounding box per frame

    frames = {}

    if kf_only:
        frameid_to_kf = {}
        for track in tracks:
            trackid = int(track.get("id"))
            label = track.get("label")
            boxes = track.findall( './box' )
            for box in boxes:
                shape_or_track = 'track'
                parent_track = box.getparent()
                nchildren = len(parent_track.getchildren())
                if nchildren == 2:
                    assert(parent_track.getchildren()[1].get('outside') == '1')
                    # its a shape not a track
                    shape_or_track = 'shape'
                if nchildren == 1:
                    # print('1 child???')
                    pass

                if shape_or_track == 'track' and shapes_only:
                    continue
                if shape_or_track == 'shape' and not shapes_only:
                    continue


                frameid  = int(box.get('frame'))
                if frameid not in frameid_to_kf:
                    frameid_to_kf[frameid] = False
                outside  = int(box.get('outside'))
                occluded = int(box.get('occluded'))
                keyframe = int(box.get('keyframe'))
                if keyframe and not outside and not occluded:
                    frameid_to_kf[frameid] = True

    for track in tracks:
        trackid = int(track.get("id"))
        label = track.get("label")
        boxes = track.findall( './box' )
        for box in boxes:
            shape_or_track = 'track'
            parent_track = box.getparent()
            nchildren = len(parent_track.getchildren())
            if nchildren == 2:
                assert(parent_track.getchildren()[1].get('outside') == '1')
                # its a shape not a track
                shape_or_track = 'shape'
            if nchildren == 1:
                # print('1 child???')
                pass

            if shape_or_track == 'track' and shapes_only:
                continue
            if shape_or_track == 'shape' and not shapes_only:
                continue


            frameid  = int(box.get('frame'))
            if kf_only:
                if not frameid_to_kf[frameid]:
                    continue
            outside  = int(box.get('outside'))
            occluded = int(box.get('occluded'))
            keyframe = int(box.get('keyframe'))
            xtl      = float(box.get('xtl'))
            ytl      = float(box.get('ytl'))
            xbr      = float(box.get('xbr'))
            ybr      = float(box.get('ybr'))
        
            frame = frames.get( frameid, {} )

            if outside == 0:
                frame[ trackid ] = { 'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr, 'label': label, 'keyframe': keyframe }

            frames[ frameid ] = frame

    return frames


if __name__ == "__main__":
    val_file = '/media/ssd1/mcever/invert_counter/data/idd/cvat_annos/00002016080800061300.xml'

    frames = process_cvat_xml(val_file, True)
