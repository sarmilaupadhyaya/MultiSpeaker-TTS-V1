# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np
import torch

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed


class TextMelMultispeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, text, sid, eid = filepath_and_text[0], filepath_and_text[1], filepath_and_text[2], filepath_and_text[3]
        text = self.get_text(text, add_blank=self.add_blank)
        sid = self.get_sid(sid)
        eid = self.get_eid(eid)
        mel = self.get_mel(filepath)
        return (text, mel, sid, eid)

    def get_mel(self, filepath):
    
        melspec = np.load(filepath, allow_pickle=True).all()['melspec_feat']

        return melspec
        
    def get_sid(self, sid):
        sid = torch.IntTensor([int(sid)])
        return sid
        
    def get_eid(self, eid):
        eid = torch.IntTensor([int(eid)])
        return eid

    def get_text(self, text, add_blank=True):
        text_norm = torch.from_numpy(np.asanyarray(text.strip().split(','), dtype=np.int)).type(torch.int32)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, mel, sid, eid = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text, 'sid': sid, 'eid': eid}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelMultispeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        sid = torch.zeros((B, 1), dtype=torch.long)
        eid = torch.zeros((B, 1), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_, sid_, eid_ = item['y'], item['x'], item['sid'], item['eid']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            sid[i,:1] = sid_
            eid[i,:1] = eid_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'sid': sid, 'eid': eid}
