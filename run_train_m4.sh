#!/bin/sh

#export PATH=/home/supadhyaya/miniconda/bin:$PATH
#source miniconda/bin/activate
source newenv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

python train_chkpt_multip.py --speaker id --lang embedding
#python eval.py
