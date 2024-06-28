#!/bin/bash
#SBATCH -J train_reID
#SBATCH -o /project/roysam/rwmills/repos/cluster-contrast-reid/runs/train.o%j
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 48 --mem=128GB
#SBATCH --gres gpu:2
#SBATCH --mail-user=rwmills@central.uh.edu
#SBATCH --mail-type=all

/project/roysam/rwmills/apps/miniconda3/envs/hhcl2/bin/python /project/roysam/rwmills/repos/cluster-contrast-reid/examples/triplet_loss_train.py\
  -b 100 -a unet -d brain -dn connectivity0_noRECA_NeuN -nc=5\
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --height 50 --width 50 --epochs 100 \
  --logs-dir /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/UNet_ftTL_myelo22_175.3/ 
  
  
# cluster_contrast_train_usl_RWM7.py \


#/project/roysam/rwmills/apps/miniconda3/envs/hhcl/bin/python /project/roysam/rwmills/repos/cluster-contrast-reid/examples/cluster_contrast_train_usl_infomap_RWM7.py \
#  -b 50 -a resnet50 -d market1501 -dn market1501 -nc=3\
#  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --height 50 --width 50 --epochs 70 \
#  --logs-dir /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/RGB_NeuN/
