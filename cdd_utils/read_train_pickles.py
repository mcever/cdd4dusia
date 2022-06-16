import os
import pickle
import numpy as np

import tkinter
import matplotlib

import argparse


def main(pickles_dir, pnum, should_plot):
    
    if should_plot:
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        # plot and clf here to throw DISPLAY error
        # before doing all the heavy lifting..
        plt.plot([0], [0], label='train_losses')
        plt.clf()
    else:
        matplotlib.use('Agg') # non GUI, good if only showing plots
        import matplotlib.pyplot as plt

    out_subdir = 'from_pickles'
    start_epoch = -1
    ppath = os.path.join(pickles_dir, 'pickles/{}.p'.format(pnum))
    OUT_DIR = os.path.join(pickles_dir, out_subdir)
    print('out_dir: {}'.format(OUT_DIR))

    with open(ppath, 'rb') as f:
        stuff = pickle.load(f)
        if len(stuff) == 5:
            (train_loss_strs, train_metric_strs, train_cebbs, val_metric_strs, val_cebbs) = stuff
            lrs = None
            test_metric_strs, test_cebbs = None, None
        elif len(stuff) == 6:
            (train_loss_strs, train_metric_strs, train_cebbs, val_metric_strs, val_cebbs, lrs) = stuff
            test_metric_strs, test_cebbs = None, None
        elif len(stuff) == 8:
            (train_loss_strs, train_metric_strs, train_cebbs, val_metric_strs, val_cebbs, test_metric_strs, test_cebbs, lrs) = stuff
        else: 
            print('len(stuff) = {}'.format(len(stuff)))
            sys.exit()

    tls = train_loss_strs
    tms = train_metric_strs
    vms = val_metric_strs

    # extract train losses
    train_losses = []
    seen_epochs = []
    epoch_train_losses = []
    for i,emsgs in enumerate(tls):
        for msg in emsgs:
            try:
                s_msg = msg.split(' ') # split the epoch mss
                enum = int(s_msg[1][1:-1])
                
                parts = msg.split(':')
                assert('[' in parts[1])
                ep_num = int(parts[1].split(' ')[1][1:-1])
                if ep_num not in seen_epochs:
                    seen_epochs.append(ep_num)
                    epoch_train_losses.append(-1)
                for pi,part in enumerate(parts):
                    
                    if ('loss' in part) and ('_' not in part):
                        med_loss = float(parts[pi+1].split(' ')[1])
                        train_losses.append(med_loss)
                        epoch_train_losses[-1] = med_loss
                    '''
                    if ('lr' in part) and ('_' not in part):
                        lr = float(parts[pi+1].split(' ')[1])
                    '''


                '''
                loss_cls = float(s_msg[19])
                loss_box_reg = float(s_msg[23])
                loss_obj = float(s_msg[27])
                '''
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()

    if test_cebbs is not None:
        all_cebbs = [train_cebbs, val_cebbs, test_cebbs]
    else:
        all_cebbs = [train_cebbs, val_cebbs]
    best_AP = [-1.0]*len(all_cebbs)
    best_AR = [-1.0]*len(all_cebbs)
    best_AR_ep = [-1]*len(all_cebbs)
    best_AP_ep = [-1]*len(all_cebbs)
    APss = []
    ARss = []
    tpss, fpss, posss = [], [], []
    for i,cebbs in enumerate(all_cebbs):
        APs = []
        ARs = []
        tps, fps, poss = [], [], []
        for j,cebb in enumerate(cebbs):
            # assume cebb.eval['params'] never changes...
            # so looking at [IoU=0.5 | area=all | maxDets=100]
            APs.append(cebb.stats[1])
            ARs.append(cebb.stats[8])

            if cebb.stats[1] > best_AP[i]:
                best_AP[i] = cebb.stats[1]
                best_AP_ep[i] = j
            if cebb.stats[8] > best_AR[i]:
                best_AR[i] = cebb.stats[8]
                best_AR_ep[i] = j

            tps.append(cebb.eval['all_tps'][0,:,0,2])
            fps.append(cebb.eval['all_fps'][0,:,0,2])
            poss.append(cebb.eval['all_pos'][0,:,0,2])

        APss.append(APs)
        ARss.append(ARs)
        tpss.append(tps)
        fpss.append(fps)
        posss.append(poss)

    os.makedirs(OUT_DIR, exist_ok=True)

    splts = ['train', 'val', 'test']

    to_plot = train_losses
    plt.plot(list(range(len(to_plot))), to_plot, label='train_losses')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'train_losses.png'))
    plt.title('{} {}'.format(pickles_dir, pnum))
    plt.clf()

    to_plot = epoch_train_losses
    plt.plot(list(range(len(to_plot))), to_plot, label='train_losses')
    for i in range(len(APss)):
        to_plot = APss[i]
        plt.plot(list(range(len(to_plot))), to_plot, label='{}_APs'.format(splts[i]))
        to_plot = ARss[i]
        plt.plot(list(range(len(to_plot))), to_plot, label='{}_ARs'.format(splts[i]))
    '''
    to_plot = APss[1]
    plt.plot(list(range(len(to_plot))), to_plot, label='val_APs')
    to_plot = ARss[1]
    plt.plot(list(range(len(to_plot))), to_plot, label='val_ARs')
    '''
    if lrs is not None:
        to_plot = lrs
        plt.plot(list(range(len(to_plot))), to_plot, label='lrs')
    plt.legend()
    plt.title('{} {}'.format(pickles_dir, pnum))
    plt.savefig(os.path.join(OUT_DIR, 'summaries_{}.png'.format(pnum)))
    if should_plot:
        plt.show()
    plt.clf()

    tp_sumss = []
    fp_sumss = []
    for i in range(len(tpss)):
        splt = splts[i]
        try: 
            tsum = np.sum(np.array(tpss[i]), axis=1)
        except:
            # import pdb; pdb.set_trace()
            break
        to_plot = tsum
        plt.plot(list(range(len(to_plot))), to_plot, label='{}_tps'.format(splt))

        fsum = np.sum(np.array(fpss[i]), axis=1)
        to_plot = fsum
        plt.plot(list(range(len(to_plot))), to_plot, label='{}_fps'.format(splt))

        psum = np.sum(np.array(posss[i]), axis=1)
        to_plot = psum
        plt.plot(list(range(len(to_plot))), to_plot, label='{}_poss'.format(splt))

    plt.legend()
    plt.title('{} {}'.format(pickles_dir, pnum))
    plt.savefig(os.path.join(OUT_DIR, 'tp_fp_pos.png'))
    plt.clf()

    if len(best_AP) == 2:
        print('[train, val]')
    elif len(best_AP) == 3:
        print('[train, val, test]')
    else:
        print('len(best_AP): {}'.format(len(best_AP)))

    print('best APs {} from eps {}'.format(best_AP, best_AP_ep))
    print('best ARs {} from eps {}'.format(best_AR, best_AR_ep))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pdir', type=str)
    parser.add_argument('pnum', type=int)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    pickles_dir = args.pdir
    pnum = args.pnum
    main(args.pdir, args.pnum, args.plot)


