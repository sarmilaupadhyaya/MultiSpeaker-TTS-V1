# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = 'resources/filelists/final_merged_train.txt'
valid_filelist_path = 'resources/filelists/final_merged_val.txt'
test_filelist_path = 'resources/filelists/final_merged_test.txt'
cmudict_path = 'resources/cmu_dictionary'
n_feats = 80
add_blank = False

# multispeaker
n_speakers =12 #n_Speakers+1
gin_channels_spk = 80
# expressive
#n_emotions = 5# n_emotions + 1, 
#gin_channels_emotion = 80
n_langs = 2
gin_channels_langs = 80

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = '/srv/storage/multispeechedu@talc-data2.nancy.grid5000.fr/software_project/akriukova/gradtts_model/logs/' # model dir path

test_size = 4
n_epochs = 10000
batch_size = 16
learning_rate = 1e-4
seed = 37
save_every = 1
out_size = fix_len_compatibility(2*22050//256)
language_id = {0:"fr", 1:"en"}
