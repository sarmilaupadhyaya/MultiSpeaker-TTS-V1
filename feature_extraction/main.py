import sys
from scipy.io.wavfile import read
import numpy as np
import librosa
import hparams
import commons
import pandas as pd

import torch
import pipeline
sys.path.append('../')
from text import *



df = pd.read_csv("../feature_extraction/phonemes.csv", header=None)
df.columns=["phoneme", "id"]
dictionary = {row["phoneme"]:row["id"] for index, row in df.iterrows()}

stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

def get_mel(filename):
    #print(filename)
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, stft.sampling_rate))
    if max(audio) < 1.0:
        audio_norm = audio
    else:
        audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.clip(audio_norm, -1.0, 1.0)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    
    return melspec
    
def load_wav_to_torch(full_path, target_sampling_rate=22050):
    sampling_rate, data = read(full_path)
    if sampling_rate != target_sampling_rate:
        data = librosa.resample(data.astype(np.float32), sampling_rate, target_sampling_rate)
    data = torch.FloatTensor(data.astype(np.float32))
    return data, target_sampling_rate
    
def main(filepath, language, speakerid, langid):
    print(language, speakerid, langid)
    f = open("../resources/filelists/final_" +filepath, "w")
    for line in open(filelists_path, 'r'):
        fpath = line.strip().split('|')[0]
        text = line.strip().split('|')[1]
        melspec = get_mel(fpath)
        print(fpath.replace('.wav', '.npy'))
        np.save(fpath.replace('.wav', '.npy'), melspec)
        #p = pipeline.Preprocessing(text, dictionary, language="en")
        seq = [str(each) for each in text_to_sequence(text, dictionary=dictionary, language=language)]
        print(seq)
        if seq is None:
            import pdb
            pdb.set_trace()
        speaker_id = speakerid
        emotion_id = 0
        lang_id = langid
        f.write(fpath.replace('.wav', '.npy') + "|" + ",".join(seq) + "|" + str(speaker_id) + "|" + str(emotion_id) + "|" + str(lang_id))
        f.write("\n")


        
        
        
if __name__ == "__main__":
    filelists_path = sys.argv[1]
    language = sys.argv[2]
    speakerid = sys.argv[3]
    langid = sys.argv[4]
    main(filelists_path, language, speakerid, langid)      
        
        
        
        
        
        

        
        
        

