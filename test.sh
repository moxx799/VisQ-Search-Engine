#!/bin/bash

#SBATCH -J test_reID
#SBATCH -o /project/roysam/rwmills/repos/cluster-contrast-reid/runs/test.o%j
#SBATCH -t 6:00:00
#SBATCH -N 1 -n 48 --mem=128GB 
#SBATCH --gres gpu:2 # 3


/project/roysam/rwmills/apps/miniconda3/envs/hhcl/bin/python /project/roysam/rwmills/repos/cluster-contrast-reid/examples/RWM_testUnet.py \
  -d brain -a unet --resume /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/UNet_ftTL_myelo22_175.2/model_best.pth.tar \
  --height 50 --width 50 -nc=5 -b 1\
  -dn connectivity0_noRECA_NeuN_TESTALL/ \
  
  
#width and height were at 100, not sure why 
