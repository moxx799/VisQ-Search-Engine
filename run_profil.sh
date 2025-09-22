#!/bin/bash
#SBATCH -J make_profiling
#SBATCH -o profiling.o%j
#SBATCH -t 10:00:00
#SBATCH -N 1 -n 4
#SBATCH --gpus=1


source activate Ali

python examples/profileing.py \
 --dataset glial_panel_all\
 -b 2048 \
 --outputCSV /home/lhuang37/repos/VisQ-Search-Engine/examples/paper_results/glial.csv\
 --panel glial_panel



