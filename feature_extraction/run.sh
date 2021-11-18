#!/bin/sh

#export PATH=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/akulkarni/new_workspace/anaconda/bin:$PATH

source ../newenv/bin/activate

python main.py siwis_val.txt fr 0 0
python main.py siwis_test.txt fr 0 0
python main.py siwis_train.txt fr 0 0
python main.py ljs_train.txt en 1 1
python main.py ljs_test.txt en 1 1
python main.py ljs_val.txt en 1 1
python main.py vctk_train.txt en 2 1

#python train.py config_speaker.json

