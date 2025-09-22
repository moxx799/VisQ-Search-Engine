#!/bin/bash
#SBATCH -J train_reID
#SBATCH -o unetpos.o%j
#SBATCH -t 40:00:00
#SBATCH -N 1 -n 8
#SBATCH --gres gpu:1
#SBATCH --mem=256GB

source activate Ali

python /home/lhuang37/repos/VisQ-Search-Engine/examples/graph_train.py\
  -b 256 -a unet -d brain -dn myelo_panel_all  -nc=7 --lr 0.00035 --weight-decay 5e-4\
  --iters 200 --momentum 0.2 --eps 0.4 --num-instances 16 --features 0 --height 50 --width 50 --epochs 80 --eval-step 5\
  --k1 24 --loc-temp 0.0  \
  --logs-dir /home/lhuang37/repos/VisQ-Search-Engine/logs/myelo_unetw0


