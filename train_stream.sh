#!/bin/bash
#SBATCH -J train_reID
#SBATCH -o /project/roysam/rwmills/repos/cluster-contrast-reid/runs/trainMS.o%j
#SBATCH -t 24:00:00
#SBATCH -N 1 -n 24 --mem=128GB
#SBATCH --gres gpu:2
#SBATCH --mail-user=rwmills@central.uh.edu
#SBATCH --mail-type=all

/project/roysam/rwmills/apps/miniconda3/envs/hhcl/bin/python /project/roysam/rwmills/repos/cluster-contrast-reid/examples/cluster_contrast_train_usl_RWM_multiScale.py \
  -b 100 -a resnet50 -d brain -dn connectivity0_noRECA_Glial -nc=8\
  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --height 150 --width 150 --epochs 70 \
  --logs-dir /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/connectivity0_S10G_MS/ 
  
  
# cluster_contrast_train_usl_RWM7.py \


#/project/roysam/rwmills/apps/miniconda3/envs/hhcl/bin/python /project/roysam/rwmills/repos/cluster-contrast-reid/examples/cluster_contrast_train_usl_infomap_RWM7.py \
#  -b 50 -a resnet50 -d market1501 -dn market1501 -nc=3\
#  --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --height 50 --width 50 --epochs 70 \
#  --logs-dir /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/RGB_NeuN/