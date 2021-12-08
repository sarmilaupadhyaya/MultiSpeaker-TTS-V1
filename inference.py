# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
import os
import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
import pandas as pd

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
from text import text_to_sequence

df = pd.read_csv("feature_extraction/phonemes.csv", header=None)
df.columns=["phoneme", "id"]
dictionary = {row["phoneme"]:row["id"] for index, row in df.iterrows()}


def get_text(text, language,add_blank=True):

    seq = [str(each) for each in text_to_sequence(text, dictionary=dictionary, language=language)]
    text_norm = torch.from_numpy(np.asanyarray(seq, dtype=np.int)).type(torch.int32)
    if add_blank:
        text_norm = intersperse(text_norm, 200)  # add a blank token, whose id number is len(symbols)
    text_norm = torch.IntTensor(text_norm)
    return text_norm


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = 1
   # if 'iteration' in checkpoint_dict.keys():
    #    iteration = checkpoint_dict['iteration']
    #if 'learning_rate' in checkpoint_dict.keys():
    #    learning_rate = checkpoint_dict['learning_rate']
    #if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
    #    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict= {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("Loaded checkpoint '{}' (iteration {})" .format(
                      checkpoint_path, iteration))
    return model

def get_id(id_):

    id_ = torch.LongTensor([int(id_)])
    return id_


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'
grad_checkpoint = '/srv/storage/multispeechedu@talc-data2.nancy.grid5000.fr/software_project/akriukova/gradtts_model/logs/speaker_id_lang_id/G_229.pth'
outpath = '/srv/storage/multispeechedu@talc-data2.nancy.grid5000.fr/software_project/akriukova/gradtts_model/baseline_v1'

if os.path.exists(outpath) == False:
    os.mkdir(outpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    #parser.add_argument('-s', '--speaker', type=str, required=True, help='path to a file with texts to synthesize')
    #parser.add_argument('-e', '--emotion', type=str, required=True, help='path to a file with texts to synthesize')
    #parser.add_argument('-o', '--outpath', type=str, required=True, help='path to a file with texts to synthesize')
    #parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=1000, help='number of timesteps of reverse diffusion')
    args = parser.parse_args()

    nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)
    
    print('Initializing Grad-TTS...')
    generator = GradTTS(nsymbols, params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale, params.n_speakers, params.n_langs, params.gin_channels_spk, params.gin_channels_langs, "id", "id")
    #generator.load_state_dict(torch.load(grad_checkpoint, map_location=lambda loc, storage: loc))
    load_checkpoint(grad_checkpoint, generator)
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    
    fnames = [os.path.basename(line.strip().split('|')[0]).replace('.pt.npy', '.wav') for line in open(args.file, 'r')]
    texts = [line.strip().split('|')[1] for line in open(args.file, 'r')]
    sids = [line.strip().split('|')[3] for line in open(args.file, 'r')]
    lids = [line.strip().split('|')[4] for line in open(args.file, 'r')]
    languages = [line.strip().split('|')[2] for line in open(args.file ,'r')]
    
    #cmu = cmudict.CMUDict('./resources/cmu_dictionary')
        
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            
            speaker_tag=sids[i]
            lang_tag=lids[i]
            language = languages[i]
            print(language)
            x = get_text(text,language,params.add_blank).unsqueeze(0).cuda()
            
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            
            g1 = torch.LongTensor(get_id(int(sids[i]))).unsqueeze(0).cuda()
            g2 = torch.LongTensor(get_id(int(lids[i]))).unsqueeze(0).cuda()
            #g2 = torch.LongTensor(get_id(int(6))).unsqueeze(0).cuda()
            print(x.shape, x_lengths, g1.shape, g2.shape)
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, length_scale=0.91, g1=g1, g2=g2)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            import pdb
            pdb.set_trace()
            
            write(os.path.join(outpath, speaker_tag+'_'+lang_tag+'_'+fnames[i]), 22050, audio)

    print('Done. Check out `out` folder for samples.')
