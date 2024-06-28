CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --height 224 --width 224

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --logs-dir /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/infomap/ --height 50 --width 50 


#test: 
CUDA_VISIBLE_DEVICES=0 \
python examples/RWM_test.py -d brain --resume /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/7Ch_2k/model_best.pth.tar --height 50 --width 50
  
  
#train 7 ch brain: 
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_RWM7.py -b 256 -a resnet50 -d brain --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --height 50 --width 50 --logs-dir /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/7Ch_2k/

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_RWM7.py -b 256 -a resnet50 -d brain --iters 200 --momentum 0.1 --eps 0.6 --num-instances 100 --height 50 --width 50 --logs-dir /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/7Ch_3k_100inst/

python examples/cluster_contrast_train_usl_infomap_RWM7.py -b 256 -a resnet50 -d brain --iters 200 --momentum 0.1 --eps 0.6 --num-instances 100 --height 50 --width 50 --logs-dir /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/7Ch_3k_IM_100inst/

python examples/cluster_contrast_train_usl_RWM7.py -b 256 -a resnet50 -d brain --iters 200 --momentum 0.1 --eps 0.6 --num-instances 20 --height 50 --width 50 --logs-dir /project/roysam/rwmills/repos/cluster-contrast-reid/examples/logs/7Ch_5k_IM_20inst_noCam/ 