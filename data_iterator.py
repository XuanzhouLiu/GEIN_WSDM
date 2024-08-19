# -*- coding: utf-8 -*-
import numpy
import json
import pickle as pkl
import random
import gzip
from dataset_process import shuffle
#import shuffle
import time
import pandas as pd

def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)
            # return unicode_to_utf8(pkl.load(f))

def fopen(filename, mode='r'):
    if str(filename).endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def sampling_behavior_negatives_odps(mid_list, pos_list, mid_list_for_random, meta_id_map):
    #count = 0
    noclk_mid_list = []
    noclk_cat_list = []
    noclk_pos_list = []
    for i, pos_mid in enumerate(mid_list):
        noclk_tmp_mid = []
        noclk_tmp_cat = []
        noclk_tmp_pos = []
        noclk_index = 0
        while True:

            noclk_mid_indx = random.randint(0, len(mid_list_for_random) - 1)
            noclk_mid = mid_list_for_random[noclk_mid_indx]
            if noclk_mid == pos_mid:
                continue
            noclk_tmp_mid.append(noclk_mid)
            noclk_tmp_cat.append(meta_id_map[noclk_mid] if noclk_mid in meta_id_map else 0)
            noclk_tmp_pos.append(pos_list[i])
            noclk_index += 1
            if noclk_index >= 5:
                break
        noclk_mid_list.append(noclk_tmp_mid)
        noclk_cat_list.append(noclk_tmp_cat)
        noclk_pos_list.append(noclk_tmp_pos)

    return noclk_mid_list, noclk_cat_list, noclk_pos_list


class DataIterator:
    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 item_info,
                 reviews_info,
                 batch_size=128,
                 maxlen=20,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=False,
                 max_batch_size=20,
                 minlen=None,
                 max_time=60*24*30,
                 graph_prod_num= 3,
                 seed = 0,
                 distance = None,
                 causal = False,
                 target=False):
        # shuffle the input file
        self.graph_prod_num = graph_prod_num
        self.seed = seed
        self.rng = numpy.random.default_rng(seed = seed)
        self.source_orig = source
        self.batch_size = batch_size
        self.distance = distance
        self.target = target

        if distance is not None:
            self.distance_begin = 0
            self.data_buffer = []

        if shuffle_each_epoch:
            self.source = shuffle.main(self.source_orig, temporary=False, seed=self.rng.integers(100000))
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc]:
            self.source_dicts.append(load_dict(source_dict))

        # Mapping Dict: {item:category}
        f_meta = open(item_info, "r")
        meta_map = {}
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                if len(arr) == 3:
                    meta_map[arr[0]] = arr[2]
                else:
                    meta_map[arr[0]] = arr[1]
        self.meta_id_map = {}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx

        # Get all the interacted items
        f_review = open(reviews_info, "r")  # [user, item, rating, timestamp]
        self.mid_list_for_random = []
        for line in f_review:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[1]:  # if the item exsist,
                tmp_idx = self.source_dicts[1][arr[1]]  # get item's ID
            self.mid_list_for_random.append(tmp_idx)  # list of all the interacted items

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.max_time = max_time
        self.skip_empty = skip_empty
        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def get_n(self):
        return self.n_uid+1, self.n_mid+1, self.n_cat+1

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source = shuffle.main(self.source_orig, temporary=False, seed=self.rng.integers(100000))
        else:
            self.source.seek(0)

        if self.distance is not None:
            self.distance_begin = 0

    def __next__(self):
        beg_time = time.time()
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # Buffer: ss is one line of local_train_splitByUser/local_test_splitByUser
        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))
            # sort by history behavior length
            if self.sort_by_length:
                his_length = numpy.array([len(s[4].split("")) for s in self.source_buffer])
                tidx = his_length.argsort()
                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

            if self.distance is not None:
                try:
                    self.data_buffer = pd.read_csv(self.distance, skiprows=self.distance_begin, nrows=self.k, header=None)
                except:
                    self.data_buffer = pd.DataFrame()

                self.current_begin = 0
                self.distance_begin+=self.k
                if self.sort_by_length:
                    self.data_buffer = self.data_buffer.iloc[tidx[-1::-1]]
                    tidx = []

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            '''
            each ss, [label, user, item, category, [item list], [item cate list], [item decay list],
            [item graph list], [item graph score list] [item graph cate list]
            [['0',
              'AZPJ9LUT0FEPY',
              'B00AMNNTIA',
              'Literature & Fiction',
              '0307744434\x020062248391\x020470530707\x020978924622\x021590516400',
              'Books\x02Books\x02Books\x02Books\x02Books']]
            '''
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0
                # ------------------------------ History list ------------------------------#
                tmp = []
                for fea in ss[4].split(""):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                mid_list = tmp

                tmp1 = []
                for fea in ss[5].split(""):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                cat_list = tmp1

                #decay time
                tmp2 = []
                for fea in ss[6].split(""):
                    t = min(int(fea)//60, self.max_time-1)
                    tmp2.append(t)
                pos_list = tmp2

                #graph prod
                tmp3 = []
                hist_len = len(tmp2)
                graph_len = len(ss[7].split(""))
                score = list(map(float, ss[8].split("")))
                sep = graph_len//hist_len
                if sep!=self.graph_prod_num:
                    cnt = 0
                    for i, fea in enumerate(ss[7].split("")):
                        m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                        if score[i]<=1 and score[i]>0 and cnt < self.graph_prod_num:
                            tmp3.append(m)
                            cnt+=1
                        if i%sep==sep-1:
                            while cnt < self.graph_prod_num:
                                tmp3.append(0)
                                cnt+=1
                            cnt = 0

                    graph_list = tmp3
                    if len(tmp3)>self.graph_prod_num*hist_len:
                        print(tmp3)
                    tmp4 = []
                    for i,fea in enumerate(ss[9].split("")):
                        c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                        if score[i]<=1 and score[i]>0 and cnt < self.graph_prod_num:
                            tmp4.append(c)
                            cnt+=1
                        if i%sep==sep-1:
                            while cnt < self.graph_prod_num:
                                tmp4.append(0)
                                cnt+=1
                            cnt = 0
                    if len(tmp4)>self.graph_prod_num*hist_len:
                        print(tmp4)
                    graph_cate_list = tmp4
                else:
                    tmp3 = []
                    for i, fea in enumerate(ss[7].split("")):
                        m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                        tmp3.append(m)
                    graph_list = tmp3

                    tmp4 = []
                    for i,fea in enumerate(ss[9].split("")):
                        c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                        tmp4.append(c)
                    graph_cate_list = tmp4

                tmp5 = []
                for i,fea in enumerate(ss[10].split("")):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp5.append(c)
                c2c_graph_cate_list = tmp5

                if self.target:
                    target_i2i_mid = [self.source_dicts[1][v] if v in self.source_dicts[1] else 0 for v in ss[11].split("")]
                    target_i2i_cat = [self.source_dicts[2][c] if c in self.source_dicts[2] else 0 for c in ss[12].split("")]

                if self.minlen != None:
                    if len(mid_list) <= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue

                # -------------------------------- Negative sample -------------------------------#
                noclk_mid_list, noclk_cat_list, noclk_pos_list = sampling_behavior_negatives_odps(mid_list, pos_list,
                                                                             self.mid_list_for_random,
                                                                             self.meta_id_map)
                tmp_source = [uid, mid, cat, mid_list, cat_list, pos_list, graph_list, graph_cate_list, c2c_graph_cate_list, noclk_mid_list, noclk_cat_list, noclk_pos_list]
                if self.target:
                    tmp_source.append(target_i2i_mid)
                    tmp_source.append(target_i2i_cat)
                source.append(tmp_source)
                target.append([float(ss[0]), 1 - float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        if self.distance is not None:
            distance_data = self.data_buffer[self.current_begin:self.current_begin+self.batch_size]
            self.current_begin+=self.batch_size
            return source, target, distance_data
        return source, target
