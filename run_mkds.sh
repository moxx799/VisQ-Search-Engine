#!/bin/bash
#SBATCH -J make_ds
#SBATCH -o make_ds.o%j
#SBATCH -t 20:00:00
#SBATCH -N 1 -n 8
#SBATCH --mem=128G

source activate Ali

python examples/makeDS.py \
 --INPUT_DIR=/home/lhuang37/datasets/50_plex/S1/final \
 --OUTPUT_DIR=/home/lhuang37/repos/VisQ-Search-Engine/examples/data/myelo_panel_all \
 --BBXS_FILE=/home/lhuang37/datasets/50_plex/S1/classification_results/classification_table.csv \
 --CROP_SIZE 75 \
 --BIO_MARKERS_PANNEL='myelo_panel_all'



