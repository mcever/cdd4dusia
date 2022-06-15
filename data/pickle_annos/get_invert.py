import os
import pickle
import sys
from PIL import Image
import imageio
path_to_annos = '/media/ssd2/marine/idd/pickle_annos/Annos.py'
sys.path.append(path_to_annos)
from Annos import Annos


P_HOME = '/media/ssd2/marine/idd/pickle_annos'
V_HOME = '/media/ssd2/marine/idd/vids'
frame_home = '/media/ssd2/marine/idd/frames'

im_cnt = 0
IM_MAX = 10

if __name__ == "__main__":

    piks = os.listdir(P_HOME)

    pname_to_obj = {}

    for pname in piks:
        if 'habs' in pname:
            with open(os.path.join(P_HOME, pname), 'rb') as f:
                habs = pickle.load(f)
            pname_to_obj[pname] = habs
        elif 'specs' in pname:
            with open(os.path.join(P_HOME, pname), 'rb') as f:
                specs = pickle.load(f)
            ide = pname.split('_')[0]

            reader = imageio.get_reader(os.path.join(V_HOME, ide+'.mp4'))

            fnums = list(specs.keys())
            fnums.sort()

            for fnum in fnums:
                di = specs[fnum]
                if len(di['count']) < 3: continue
                print('---- {} ----'.format(fnum))
                print(di)
                # with timeout(seconds=5)j
                frame = reader.get_data(fnum)
                new_id = ide + '_{:07d}'.format(fnum)
                sv_name = os.path.join(frame_home, new_id+'.png')
                img = Image.fromarray(frame)
                img.save(sv_name)
                reader.close()
                im_cnt += 1
                if im_cnt >= IM_MAX:
                    sys.exit()
            pname_to_obj[pname] = specs



