# parse xml gathering habitat information for each frame
import math
import pandas as pd
from copy import copy

class Annos:
    def __init__(self):
        self.bounds = []
        self.substrates = []
        self.beg = None
        self.end = None

    def __getitem__(self, key):
        # binary search to return substrates
        # for a given frame number
        L = 0
        R = len(self.bounds) - 1
        while L <= R:
            m = math.floor( (L+R) / 2 )
            b = self.bounds[m]
            if b[1] < key:
                L = m + 1
            elif b[0] > key:
                R = m - 1
            else:
                return self.substrates[m]
        return {'No data'}

    def __len__(self):
        return len(self.bounds)

    def add_habitat_anno(self, habs, beg, end):
        # want to keep ordered for better lookups
        if self.beg is None: self.beg = beg
        if self.end is None: self.end = end
        if beg < self.beg: self.beg = beg
        if end > self.end: self.end = end

        habs = set(habs)
        assert(end >= beg)
        if len(self.bounds) == 0:
            self.bounds = [[beg, end]] # assume no overlap for now
            self.substrates.append(habs)
            return

        # for i, b in enumerate(self.bounds):
        i = 0
        while i < len(self.bounds):
            b = self.bounds[i]
            # asssume bounds are in order [(10, 20), (30, 40), (1000, 1003)...]
            if beg < b[0]:
                if end <= b[0]:
                    # case A: insert before and exit
                    # ideally would create another, but whatever
                    if end == b[0]: end -= 1
                    self.bounds.insert(i, [beg, end])
                    self.substrates.insert(i, habs)
                    return
                elif end <= b[1]:
                    # case B: create new intervals, insert, and exit
                    if end == b[1]:
                        # don't need to create 2
                        beg2 = -1
                    else:
                        beg2 = end+1
                        end2 = copy(b[1])
                        habs2 = copy(self.substrates[i])

                    beg3 = beg
                    end3 = b[0] - 1

                    # edit overlapped area
                    self.substrates[i] = self.substrates[i].union(habs)
                    self.bounds[i][1] = end

                    # if needed, insert after
                    if beg2 != -1:
                        self.bounds.insert(i+1, [beg2, end2])
                        self.substrates.insert(i+1, habs2)

                    # insert before
                    self.bounds.insert(i, [beg3, end3])
                    self.substrates.insert(i, habs)
                    return
                else:
                    # case C: create new intervals, insert, set beg = b[1], 
                    beg2 = beg
                    end2 = b[0]-1

                    # edit overlapped area
                    self.substrates[i] = self.substrates[i].union(habs)

                    self.bounds.insert(i, [beg2, end2])
                    self.substrates.insert(i, habs)

                    # set beginning to end of bound, and keep searching..
                    i+=1
                    beg = b[1] + 1
                    # now essentially case F
            elif beg <= b[1]: 
                if end <= b[1]: 
                    # case D: create new invertvals, insert, and exit
                    if end == b[1]:
                        beg2 = -1
                    else:
                        beg2 = end+1
                        end2 = b[1]
                        habs2 = self.substrates[i]
                    if beg == b[0]:
                        beg3 = -1
                    else:
                        beg3 = b[0]
                        end3 = beg-1
                        habs3 = self.substrates[i]

                    self.substrates[i] = self.substrates[i].union(habs)
                    self.bounds[i] = [beg, end]

                    if beg2 != -1:
                        self.bounds.insert(i+1, [beg2, end2])
                        self.substrates.insert(i+1, habs2)

                    if beg3 != -1:
                        self.bounds.insert(i, [beg3, end3])
                        self.substrates.insert(i, habs3)
                    return
                else:
                    # case E: create new intervals, insert, set beg = b[1]
                    if beg == b[0]:
                        beg2 = -1
                    else:
                        beg2 = b[0]
                        end2 = beg-1
                        habs2 = copy(self.substrates[i])

                    self.substrates[i] = self.substrates[i].union(habs)
                    self.bounds[i][0] = beg

                    if beg2 != -1:
                        self.bounds.insert(i, [beg2, end2])
                        self.substrates.insert(i, habs2)

                    # set beginning to end of bound, and keep searching..
                    i += 1
                    beg = b[1] + 1
                    # now essentially case F
            else: 
                assert(beg > b[1])
                i += 1
                l = len(self.bounds)
                if i == l:
                    # at end of list, just insert at end
                    self.bounds.insert(l, [beg, end])
                    self.substrates.insert(l, habs)
                    return

lid_to_offset = {'517_1160': -1851045, '520_1140': -2007187, '524_1170': -1615301, '532_620': -46485}


def get_frame_num(lid, tc):
    # this will fail if we go to next day or something
    FPS = 30

    offset = lid_to_offset[lid]
    hour = tc.hour
    minute = tc.minute
    second = tc.second
    seconds = 3600*hour + 60*minute + second

    return (seconds*FPS + offset)

if __name__ == "__main__":

    xlsx_name = '/media/ssd1/mcever/datasets/MARE/MARE/Fish_Invert_Habitat_Data.xlsx'
    dfs = pd.read_excel(xlsx_name, sheet_name=None)

    dfkey = 'Habitat'
    df = dfs[dfkey]
    lineID_to_annos = {}
    for i in range(len(df)):
        #print(i)
        lineID = df['LineID'][i]
        bframe_num = get_frame_num(lineID, df['BTC'][i])
        eframe_num = get_frame_num(lineID, df['ETC'][i])

        if lineID not in lineID_to_annos:
            lineID_to_annos[lineID] = Annos()
        annos = lineID_to_annos[lineID]

        annos.add_habitat_anno([df['Substrate'][i]], bframe_num, eframe_num)
    # now want to be able to say
    # annos = lineID_to_annos[lineID]
    # annos_at_time_n = annos[n]
    # substrat_at_n = annos_at_time_n['Substrate']
        
    #import pdb;pdb.set_trace()
                                                                                                                                        



