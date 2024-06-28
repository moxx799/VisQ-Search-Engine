#!/bin/bash

#SBATCH -J generate_samples
#SBATCH -o /project/roysam/rwmills/repos/cluster-contrast-reid/examples/QC/run.txt
#SBATCH -t 1-00:00:00
#SBATCH -N 2 -n 48


/project/roysam/rwmills/apps/miniconda3/envs/pytorch_gpu/bin/python generate_samples1.py 