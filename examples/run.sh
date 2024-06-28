#!/bin/bash

#SBATCH -J make_ds
#SBATCH -o /project/roysam/rwmills/repos/cluster-contrast-reid/runs/create_ds.o%j
#SBATCH -t 5:00:00
#SBATCH -N 2 -n 72 #3 -n 72

#MDA: 
#/project/roysam/rwmills/apps/miniconda3/envs/pytorch_gpu/bin/python make_blindDS_maui.py \
#--INPUT_DIR=/project/roysam/rwmills/data/brain/MDA_GBM/1168457/intra_corrected/ \
#--OUTPUT_DIR=/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/MDA_GBM_1168457_whole.2/ \
#--BBXS_FILE=/project/roysam/rwmills/data/brain/MDA_GBM/1168457/detection_results/bbxs_detection.txt \
#--DAPI=R1C1.tif \
#--HISTONES=R1C2.tif \
#--NEUN=R1C3.tif \
#--S100=R1C4.tif \
#--OLIG2=R1C5.tif \
#--IBA1=R1C6.tif \
#--RECA1=R2C2.tif \
#--other1=R2C3.tif \
#--other2=R2C4.tif \
#--other3=R2C5.tif \


/project/roysam/rwmills/apps/miniconda3/envs/pytorch_gpu/bin/python grid_image_DS.py


#50 plex: 
#/project/roysam/rwmills/apps/miniconda3/envs/pytorch_gpu/bin/python makeDS.py --INPUT_DIR=/project/roysam/rwmills/data/brain/50Plex/S1/final/ \
#--OUTPUT_DIR=/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/GlialMarkers_all \
#--BBXS_FILE=/project/roysam/rwmills/data/brain/50Plex/S1/classification_results/regions/regions_table.csv \
#--DAPI=S1_R3C5.tif \
#--HISTONES=S1_R4C5.tif \
#--NEUN=S1_R3C3.tif \
#--S100=S1_R3C9.tif \
#--OLIG2=S1_R1C9.tif \
#--IBA1=S1_R5C4.tif \
#--RECA1=S1_R5C6.tif \
#--other1=S1_R1C5.tif \
##--other2=S1_R5C4.tif \
##--other3=S1_R5C6.tif \

#NeuN, Map2, nfH, nhM, TyrosineH, Paravalbumin, Calretnin, Synptophysin 

#missing: R2C5, R5C3, , R5C5


#added: R5C10, R4C6, R2C9, 
 

#remove dapi histon and add R5C3- for sure , R2C5?, R2C9? R4C6? 

#DAPI='S1_R2C4.tif', HISTONES='S1_R5C9.tif', IBA1='S1_R2C6.tif', NEUN='S1_R5C5.tif', OLIG2='S1_R1C2.tif', RECA1='S1_R5C3.tif', S100='S1_R5C7.tif', other1='S1_R2C5.tif', other2='S1_R3C5.tif', other3='S1_R4C5.tif'

#set2: DAPI='S1_R3C3.tif', HISTONES='S1_R3C9.tif', IBA1='S1_R1C4.tif',NEUN='S1_R1C9.tif', OLIG2='S1_R5C6.tif', RECA1='S1_R1C6.tif', S100='S1_R5C4.tif', other1='S1_R1C3.tif', other2='S1_R1C7.tif', other3='S1_R1C5.tif'


#Create Blind: 

#/project/roysam/rwmills/apps/miniconda3/envs/pytorch_gpu/bin/python make_blindDS.py \
#--INPUT_DIR=/project/roysam/rwmills/data/brain/TBI/G3_mFPI_Vehicle/G3_BR14_HC_11L/final/ \
#--OUTPUT_DIR=/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/Vehicle_G3_BR14_HC_11L \
#--BBXS_FILE=/project/roysam/rwmills/data/brain/TBI/G3_mFPI_Vehicle/G3_BR14_HC_11L/classification_table.csv \
#--DAPI=R1C1.tif \
#--HISTONES=R1C4.tif \
#--NEUN=R1C5.tif \
#--S100=R1C10.tif \
#--OLIG2=R2C3.tif \
#--IBA1=R2C5.tif \
#--RECA1=R2C6.tif \
#--other1=R2C7.tif \
#--other2=R2C10.tif \
#--other3=R1C3.tif \