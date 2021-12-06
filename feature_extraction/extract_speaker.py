from collections import defaultdict
import os
import itertools

txt_path = "/srv/storage/multispeechedu@talc-data2.nancy.grid5000.fr/software_project/corpus/all_text.txt"
audio_path = "/srv/storage/multispeechedu@talc-data2.nancy.grid5000.fr/software_project/corpus/synpaflex/wavs_16bit/"
f = open(txt_path, "r")

data = f.readlines()
speaker_data = defaultdict(list)
for x in data:
    f = x.strip().split(" ")
    filename = f[0]
    txt = " ".join(f[1:])
    filepath = audio_path + filename + ".wav"
    speaker = filename.split("_")[0]
    if os.path.isfile(filepath):
        speaker_data[speaker].append(filepath + "|" + txt)
   
len_speaker_data = {k:len(v) for k,v in speaker_data.items()}

dd = dict(sorted(len_speaker_data.items(), key=lambda item: item[1], reverse=True))
dd = dict(itertools.islice(dd.items(), 5))

for k,v in dd.items():
    v = speaker_data[k]
    f = open("text_"+k+ ".txt", "w")
    for each in v:
        f.write(each)
        f.write("\n")
