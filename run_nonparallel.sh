#!/bin/sh

export PATH=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/akulkarni/new_workspace/anaconda/bin:$PATH

source activate gradtts

python inference_nonparallel.py -f /home/ajkulkarni/workplace/subset2/neutral_baseline.txt
