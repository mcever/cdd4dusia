import os
import pickle
from Annos import Annos

def subs_to_hbts(subs):
    hard = ['Boulder', 'Rock']
    soft = ['Mud', 'Sand', 'Cobble']
    hbt = []
    for sub in subs:
        if sub in hard:
            hbt.append('h')
        elif sub in soft:
            hbt.append('s')
        elif sub == 'No data':
            return 'No data'
        elif sub == 'Off Transect':
            hbt.append('O')
            continue
        else:
            print(sub)

    hbt = set(hbt)
    if 'O' in hbt:
        hbt.remove('O')
        if len(hbt) == 0:
            return None

    if len(hbt) == 2:
        return 'Mixed'

    assert(len(hbt) == 1)
        
    ch = hbt.pop()
    if ch == 'h':
        return 'Hard'
    else:
        assert(ch == 's')
        return 'Soft'


for fname in os.listdir('.'):

    if '_habs' in fname:
        with open(fname, 'rb') as f:
            subs_file = pickle.load(f)

        last_hbt = None
        bounds = []
        substrates = []
        for i in range(len(subs_file)):
            sub = subs_file.substrates[i]
            hbt = subs_to_hbts(sub)
            if hbt == None:
                # Off Transect is only anno, treat as No data
                hbt = 'No data'
            if hbt == last_hbt:
                # extend 
                extend_to = subs_file.bounds[i]
                bounds[-1][1] = subs_file.bounds[i][1]

            else:
                bounds.append(subs_file.bounds[i])
                substrates.append(set([hbt]))
                last_hbt = hbt

        # remove the bounds with 'No data'
        new_bs = []
        new_ss = []
        for i in range(len(bounds)):
            if 'No data' not in substrates[i]:
                new_bs.append(bounds[i])
                new_ss.append(substrates[i])

        # now save it
        annos = Annos()
        annos.bounds = new_bs
        annos.substrates = new_ss
        annos.beg = annos.bounds[0][0]
        annos.end = annos.bounds[-1][1]

        save_name = fname.replace('habs', 'hbts')
        with open(save_name, 'wb') as f:
            pickle.dump(annos, f)

