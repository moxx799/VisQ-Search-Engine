#!/bin/bash
#SBATCH -J test_reID
#SBATCH -o test.o%j
#SBATCH -t 26:00:00
#SBATCH -N 1 -n 8 
#SBATCH --gpus=1
#SBATCH --mem=256G

source activate Ali


CUDA_VISIBLE_DEVICES=7
python examples/RWM_testUnet.py \
  -d brain -a unet --resume /home/lhuang37/repos/VisQ-Search-Engine/examples/logs/single_subtypew0/model_best.pth.tar \
  --height 50 --width 50 -nc=6 -b 4096 --data-dir /home/lhuang37/repos/VisQ-Search-Engine/examples/data \
  -dn single_subtype/ --output-tag singlesubtypeT
