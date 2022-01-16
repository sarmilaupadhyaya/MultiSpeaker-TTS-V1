#!/bin/sh

#export PATH=/home/supadhyaya/home/supadhyaya/Speech-Backbones/Multispeaker-Grad-TTS_French_v1/Multispeaker-Grad-TTS_French_v1/newenv2/bin:$PATH
#source miniconda/bin/activate
source newenv2/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,3,4

python train_chkpt_multip.py --speaker id --lang id
#python eval.py
