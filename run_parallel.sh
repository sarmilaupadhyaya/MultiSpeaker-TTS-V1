#!/bin/sh

export PATH=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/akulkarni/new_workspace/anaconda/bin:$PATH

source activate gradtts

python inference_parallel.py -f /home/ajkulkarni/workplace/subset2/anger_baseline.txt -s siwis -e anger
python inference_parallel.py -f /home/ajkulkarni/workplace/subset2/joy_baseline.txt -s siwis -e joy
python inference_parallel.py -f /home/ajkulkarni/workplace/subset2/sad_baseline.txt -s siwis -e sad
