import os
import argparse
def send_mail(message):
    print('mail not sent')
try:
    try:
        from send_mail import send_mail
    except:
        from mare_utils.send_mail import send_mail
except:
    print('WARNING you may need to write your own send_mail')

def parse_log(log_pth):
    message = ''
    with open(log_pth, 'r') as f:
        lines = f.readlines()

    current_ep = -1
    loss_nan_count = 0
    start_time_line = 'start time line not generated'
    best_AP_line = 'na'
    best_AR_line = 'na'
    ep_to_APs = {}
    ep_to_val_APs = {}
    ep_to_per_class_AP_strs = {}
    ep_to_context_mAPs = {}
    ep_to_context_mses = {}
    ep_to_context_ap_strs = {}
    splits = ['train', 'val', 'test']
    for line in lines:
        try:
            if 'Loss is nan' in line:
                loss_nan_count += 1
            if 'Namespace(' in line:
                args_line = line
            if 'entering main at' in line:
                start_time_line = line
            if 'Epoch: [' in line:
                current_ep = int(line.split('[')[1].split(']')[0])
            if 'EVALUATING TRAIN' in line:
                evaling = 'train'
            if 'EVALUATING VAL' in line:
                evaling = 'val'
            if 'EVALUATING TEST' in line:
                evaling = 'test'
            if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] =' in line:
                ind = splits.index(evaling)
                the_ap = float(line.strip().split(' ')[-1])
                if current_ep not in ep_to_APs:
                    aps = [0.0, 0.0, 0.0]
                    aps[ind] = the_ap
                    ep_to_APs[current_ep] = aps
                else:
                    # note: if loss nans, this will overwrite the broken runs
                    # so dont need to do more to accomodate a successful run
                    # after a nand run
                    ep_to_APs[current_ep][ind] = the_ap
            if 'Per class APs' in line:
                _, class_aps_str = line.split(':')
                class_aps_str = class_aps_str.strip()
                ind = splits.index(evaling)
                if current_ep not in ep_to_per_class_AP_strs:
                    per_class_AP_strs = ['', '', '']
                    per_class_AP_strs[ind]= class_aps_str
                    ep_to_per_class_AP_strs[current_ep] = per_class_AP_strs
                else:
                    ep_to_per_class_AP_strs[current_ep][ind] = class_aps_str

            if 'context mAP:' in line:
                _, cmap = line.split(':')
                cmap = float(cmap.strip())
                ind = splits.index(evaling)
                if current_ep not in ep_to_context_mAPs:
                    cmAPs = [-1, -1, -1]
                    cmAPs[ind] = cmap
                    ep_to_context_mAPs[current_ep] = cmAPs
                else:
                    ep_to_context_mAPs[current_ep][ind] = cmap

            if 'Context MSE:' in line:
                _, cmse = line.split(':')
                cmse = float(cmse.strip())
                ind = splits.index(evaling)
                if current_ep not in ep_to_context_mses:
                    cmses = [-1.0, -1.0, -1.0]
                    cmses[ind] = cmse
                    ep_to_context_mses[current_ep] = cmses
                else:
                    ep_to_context_mses[current_ep][ind] = cmse

            if 'per class context APs:' in line:
                _, aps = line.split(':')
                aps = aps.replace(']', '')
                aps = aps.replace('[', '')
                aps = aps.strip()
                aps = [ap.strip() for ap in aps.split(',')]
                ctx_ap_str = ','.join(aps)
                ind = splits.index(evaling)
                if current_ep not in ep_to_context_ap_strs:
                    ap_strs = ['', '', '']
                    ap_strs[ind] = ctx_ap_str
                    ep_to_context_ap_strs[current_ep] = ap_strs
                else:
                    ep_to_context_ap_strs[current_ep][ind] = ctx_ap_str

            if 'best val AP so far achieved' in line:
                words = line.split(' ')
                ep_to_val_APs[current_ep] = float(words[-1].strip())
            if 'best APs [' in line:
                if best_AP_line is not 'na':
                    print('new best AP line...')
                best_AP_line = line
            if 'best ARs [' in line:
                if best_AR_line is not 'na':
                    print('new best AR line...')
                best_AR_line = line
        except Exception as e:
            message += 'problem line gives error ' + str(e)
            message += line
            message += '\n'
            print(message)

    best_train_AP, best_val_AP, best_test_AP = 'no', 'no', 'no'
    best_val_AP_ep = -1
    best_test_AP_ep = -1
    best_test_AP_at_val_ep = -1.0
    if best_AP_line is not 'na':
        best_AP_split = best_AP_line.split(' ')
        if len(best_AP_split) == 8:
            best_train_AP = float(best_AP_split[2][1:-1])
            best_val_AP = float(best_AP_split[3][:-1])
            best_test_AP = -1
            best_train_AP_ep = int(best_AP_split[6][1:-1])
            best_val_AP_ep = int(best_AP_split[7].strip()[:-1])
            best_test_AP_ep = -1
            best_test_AP_at_val_ep = -1
        else:
            best_train_AP = float(best_AP_split[2].split('[')[1][:-1])
            best_val_AP = float(best_AP_split[3][:-1])
            best_test_AP = float(best_AP_split[4][:-1])
            best_val_AP_ep = int(best_AP_split[8][:-1])
            best_train_AP_ep = -1
            best_test_AP_ep = int(best_AP_split[9].strip()[:-1])
            best_test_AP_at_val_ep = ep_to_APs[best_val_AP_ep][2]
    
    best_train_AR, best_val_AR, best_test_AR = 'no', 'no', 'no'
    best_val_AR_ep = -1
    if best_AR_line is not 'na':
        best_AR_split = best_AR_line.split(' ')
        if len(best_AR_split) == 8:
            best_train_AR = float(best_AR_split[2][1:-1])
            best_val_AR = float(best_AR_split[3][:-1])
            best_test_AR = -1
            best_val_AR_ep = int(best_AR_split[-1].strip()[:-1])
        else:
            best_train_AR = float(best_AR_line.split(' ')[2].split('[')[1][:-1])
            best_val_AR = float(best_AR_line.split(' ')[3][:-1])
            best_test_AR = float(best_AR_line.split(' ')[4][:-1])
            best_val_AR_ep = int(best_AR_line.split(' ')[8][:-1])


    # add per class ap stuff in to csl
    per_class_aps_at_best_val = '-0.1,'*10
    if best_val_AP_ep in ep_to_per_class_AP_strs:
        ind = splits.index('test')
        per_class_aps_at_best_val = ep_to_per_class_AP_strs[best_val_AP_ep][ind]
        if len(per_class_aps_at_best_val.split(',')) < 10:
            ind = splits.index('val')
            per_class_aps_at_best_val = ep_to_per_class_AP_strs[best_val_AP_ep][ind]
            if len(per_class_aps_at_best_val.split(',')) < 10:
                print('not enough classes in val class APS????')
                message += 'not enough classes in val class APS????\n'

    # add context stuff into csl
    context_ap_str = '-0.1,'*4
    if best_val_AP_ep in ep_to_context_mAPs:
        ind = splits.index('test')
        context_ap_str = ep_to_context_ap_strs[best_val_AP_ep][ind] + ','
    context_map_str = '-0.1,'
    if best_val_AP_ep in ep_to_context_mAPs:
        ind = splits.index('test')
        context_map_str = str(ep_to_context_mAPs[best_val_AP_ep][ind]) + ','
         


    csl = '{},{},{},{},{},{},{},'.format(best_val_AP, best_val_AP_ep, best_val_AR, 
            best_val_AR_ep, best_test_AP, best_test_AP_ep, best_test_AP_at_val_ep)
    csl += per_class_aps_at_best_val
    csl += context_ap_str
    csl += context_map_str
    if csl[-1] == ',':
        csl = csl[:-1]
    message += 'start time: ' + start_time_line
    message += 'loss_nan_count: ' + str(loss_nan_count) + '\n'
    message += best_AP_line + best_AR_line
    return message, csl

if __name__ == "__main__":
    # log_pth = '../detection/exps/out041522_02/train.log'
    log_pth = '../detection/exps/out041522_03/train.log'
    log_pth = '../detection/exps/kep041522_90/train.log'

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_mail', default=False, action='store_true')
    parser.add_argument('log_pth', type=str)
    args = parser.parse_args()
    assert(os.path.isfile(args.log_pth))

    message, csl = parse_log(args.log_pth)
    print(message)
    print(csl)

    msg = 'SUBJECT: ' + args.log_pth + '\n\n'
    msg += message
    msg += csl

    if not args.no_mail:
        send_mail(msg)
    else:
        print('no mail')



