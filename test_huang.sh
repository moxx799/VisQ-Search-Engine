#!/bin/bash
#SBATCH -J test_reID
#SBATCH -o test.o%j
#SBATCH -t 6:00:00
#SBATCH -N 1 -n 36 
#SBATCH --gpus=2

conda activate Ali
python examples/RWM_testUnet.py \
  -d brain -a unet --resume /project/ece/roysam/lhuang37/cluster-contrast-reid/examples/logs/UNet_ftTL_myelo22_175.2/model_best.pth.tar \
  --height 50 --width 50 -nc=5 -b 1\
  -dn connectivity0_noRECA_NeuN_TESTALL/ \
  
  
#width and height were at 100, not sure why 
