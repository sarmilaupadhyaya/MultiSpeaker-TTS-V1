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
from model.utils import fix_len_compatibility

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
from text import text_to_sequence
import logging as logger
sys.path.append('./')


df = pd.read_csv("feature_extraction/phonemes.csv", header=None)
df.columns=["phoneme", "id"]
dictionary = {row["phoneme"]:row["id"] for index, row in df.iterrows()}




HIFIGAN_CONFIG = 'checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = 'checkpts/hifigan.pt'
sample_lang={0:"sample/audio_lang_0.npy", 1:"sample/audio_lang_1.npy"}
sample_speaker = "sample/audio_speaker_"

def get_mel( filepath):

    mel = np.load(filepath, allow_pickle=True)
    mel = np.expand_dims(mel, axis=0)
    B = len(mel)
    y_max_length = max([item.shape[-1] for item in mel])
    y_max_length = fix_len_compatibility(y_max_length)
    n_feats = mel[0].shape[-2]
    y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
    return y

def get_mel_speaker(i):
    return get_mel(sample_speaker+str(i)+".npy")

def get_mel_language(i):
    return get_mel(sample_lang[i])

def get_text(text, language,add_blank=True):
    print(language)
    seq = [str(each) for each in text_to_sequence(text, dictionary=dictionary, language=language)]
    text_norm = torch.from_numpy(np.asanyarray(seq, dtype=np.int)).type(torch.int32)
    if add_blank:
        text_norm = intersperse(text_norm, 200)  # add a blank token, whose id number is len(symbols)
    text_norm = torch.IntTensor(text_norm)
    return text_norm

def load_checkpoint(checkpoint_path, model, optimizer=None):
    print(checkpoint_path)
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

def load_grad_tts(checkpoint, nsymbols, speaker_rep, lang_rep):
    print('Initializing Grad-TTS...')
    generator = GradTTS(nsymbols, params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale, params.n_speakers, params.n_langs, params.gin_channels_spk, params.gin_channels_langs, speaker_rep, lang_rep)
    load_checkpoint(checkpoint, generator)
    _ = generator.eval()
    print(f'Number of parameters: {generator.nparams}')
    return generator

def load_hifi():
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder

def get_id(id_):
    id_ = torch.LongTensor([int(id_)])
    return id_

    
#if using checkpts from flashdrive, mount the D drive first
def main(text, checkpts="models", timesteps=50, speaker_id=2, lang_id=1, language="en", rep="id", outpath="out/", out_f=None):
    nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)
    chosen = "G_1000_model1_speaker2.pth"
    params.n_speakers = 2
    if rep == "emb":
        chosen = "G_1000_model4_6speakers.pth"
        params.n_speakers = 6
    checkpt = os.path.join(checkpts, chosen)
    generator = load_grad_tts(checkpt, nsymbols, rep, rep)
    vocoder = load_hifi()
    cmu = cmudict.CMUDict('resources/cmu_dictionary')
    texts = [text]
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing...', end=' ')
            print(language)
            x = get_text(text,language,params.add_blank).unsqueeze(0)
            x_lengths = torch.LongTensor([x.shape[-1]])
            # switching the model
            if rep == "id":
                g1 = torch.LongTensor(get_id(int(speaker_id))).unsqueeze(0)
                g2 = torch.LongTensor(get_id(int(lang_id))).unsqueeze(0)
            else:
                g1 =get_mel_speaker(speaker_id)
                g2 = get_mel_language(lang_id)
            print(x.shape, x_lengths, g1.shape, g2.shape)
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=timesteps, temperature=1.5,
                                                   stoc=False, length_scale=0.91, g1=g1, g2=g2)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            if out_f:
                write(os.path.join(outpath, out_f), 22050, audio)
            else:
                write(os.path.join(outpath, str(speaker_id)+'_'+str(lang_id) +".wav"), 22050, audio)
                write(os.path.join(outpath, "latest" +".wav"), 22050, audio)
    print('Done. Check out `out` folder for samples.')
if __name__ == '__main__':
    for i in range(6):
        main("This is a test of our text to speech system.", lang_id=1, language="en", speaker_id=i, rep="emb")

