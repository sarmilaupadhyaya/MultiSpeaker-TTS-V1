#!/bin/sh

#export PATH=/home/supadhyaya/miniconda/bin:$PATH
#source miniconda/bin/activate
source newenv2/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

python train_chkpt_multip.py --speaker embedding --lang embedding
#python eval.py
