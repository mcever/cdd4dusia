import pickle
import os

def adj_fnums(fnum, fnums):
    # print adjacent anno fnums
    try:
        ind = fnums.index(fnum)
        new_fnum = fnum
    except ValueError:
        new_fnum = min(fnums, key=lambda x:abs(x-fnum))
        print('nearest to {} is {}'.format(fnum, new_fnum))
        ind = fnums.index(new_fnum)

    for i in range(ind-2, ind+3):
        try:
            print(fnums[i])
        except IndexError:
            print('end')
            break
    return new_fnum

def get_anno_n_adj(sd, fnum):
    fnums = sorted(list(sd.keys()))

    new_fnum = adj_fnums(fnum, fnums)
    print(sd[new_fnum])

id_to_sd = {}
id_to_fnums = {}

for fname in os.listdir('.'):
    if 'specs' not in fname:
        continue

    ide, _ = fname.split('_')

    with open(fname, 'rb') as f:
        sd = pickle.load(f)

    fnums = sorted(list(sd.keys()))

    id_to_sd[ide] = sd
    id_to_fnums[ide] = fnums



