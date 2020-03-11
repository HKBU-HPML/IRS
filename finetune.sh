# kitti 2015
python kitti_finetune.py --maxdisp 192 \
                   --model dispnetc \
                   --devices 0,1,2,3 \
                   --datatype 2012 \
                   --datapath /media/ssd1/kitti2012/training/ \
                   --epochs 0 \
                   --loss loss_configs/dispnetcres_kitti.json \
                   --savemodel ./trained/dispnetc \
                   --loadmodel ./models/dispnetc-ft3d-irs/dispnetc_2_13.pth \
                   #--loadmodel ./models/dispnetc-irs/dispnetc_2_19.pth \
                   #--loadmodel ./models/finalize/dispnetc-flying-d0.91.pth \
                   #--savemodel ./trained/dispCSRes-imn-KITTI2015-valavg-split/ \
                   #--loadmodel ./trained/dispCSRes-imn-KITTI2015-valavg-split/best.tar \
                   #--loadmodel ./models/dispCSRes-imn/model_best.pth \
                   #--loadmodel  ./models/ShDispCSRes/model_best.pth
                   #--loadmodel /datasets/sh_kittis/ShKittiTrained/dispnet-imagenet-argument/best.tar 

## kitti 2012
#python kitti_finetune.py --maxdisp 192 \
#                   --model dispnetcres \
#                   --devices 0,1,2,3 \
#                   --datatype 2012 \
#                   --datapath /datasets/kitti2012/training/ \
#                   --epochs 1200 \
#                   --savemodel trained/dispnetcres-imn-bottom-2012/ \
#                   --loss loss_configs/dispnetcres_kitti.json \
#                   --loadmodel  ./models/dispCSRes-imn/model_best.pth \

