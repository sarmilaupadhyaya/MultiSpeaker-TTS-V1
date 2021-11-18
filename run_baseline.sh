#!/bin/sh

export PATH=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/akulkarni/new_workspace/anaconda/bin:$PATH

source activate gradtts

python inference.py -f /home/ajkulkarni/workplace/subset2/anger_baseline.txt -s christine -e anger
python inference.py -f /home/ajkulkarni/workplace/subset2/joy_baseline.txt -s christine -e joy
python inference.py -f /home/ajkulkarni/workplace/subset2/sad_baseline.txt -s christine -e sad
python inference.py -f /home/ajkulkarni/workplace/subset2/neutral_baseline.txt -s siwis -e neutral
