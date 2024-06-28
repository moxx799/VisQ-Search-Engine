#!/bin/bash

#SBATCH -J label_reID
#SBATCH -o /project/roysam/rwmills/repos/cluster-contrast-reid/runs/get_labels.o%j
#SBATCH -t 01:00:00
#SBATCH -N 1 -n 48 
#SBATCH --gres gpu:1



/project/roysam/rwmills/apps/miniconda3/envs/hhcl/bin/python /project/roysam/rwmills/repos/cluster-contrast-reid/examples/QC/get_labels.py \
  -d brain --resume /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/internal1.8_neuronal/model_best.pth.tar \
  --height 100 --width 100 -nc=10 -b 10 \
  -dn internal1.8_neuronal --data-dir=/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/ --logs-dir=/project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/internal1.8_neuronal/ \
  
  
#width and height were at 100, not sure why 

#/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/internal1.9_cortex2/
#/project/roysam/rwmills/repos/cluster-contrast-reid/examples/QC/data/internal1.9_cortex2