#!/bin/sh

#export PATH=/home/supadhyaya/miniconda/bin:$PATH
#source miniconda/bin/activate
source newenv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python train_chkpt.py
#python eval.py
