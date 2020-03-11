#net=dispnetcres
#net=dispnetcss
#net=dispnormnet
#net=dtonnet
net=dispnetc
#net=dnfusionnet
#net=dtonfusionnet
#net=normnets

#label=irs
#label=driving
#label=middlebury
#label=sintel
#label=flying
#label=over_exposure
#label=dark
label=glass_mirror
#label=len_flare
#label=metal
loss=loss_configs/test.json
#outf_model=models/dispnetc_ue_norm2
#logf=logs/dispnetc_norm.log

outf_model=models/test/
#logf=logs/dispnetcss_SIRS.log
#logf=logs/dispnetcss_fly_on_SIRS.log
#logf=logs/dispnetcss_sceneflow_on_${label}.log
logf=logs/${net}_ft3d_irs_on_${label}.log

lr=1e-4
devices=0,1,2,3
#trainlist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TRAIN.list
#vallist=lists/SHAOHUAI_CLEAN_FlyingThings3D_release_TEST.list
#trainlist=lists/home_train.list
#trainlist=lists/ue_sperate_train.list
#trainlist=lists/home_extend_train.list
#trainlist=lists/FlyingThings3D_release_TRAIN.list
#vallist=lists/ue_sperate_test.list
#vallist=lists/office_test.list
#vallist=lists/home_extend_test.list
#vallist=lists/home_test.list
#vallist=lists/KITTI_TRAIN.list

dataset=irs
trainlist=lists/IRSDataset_TRAIN.list
vallist=lists/IRS_${label}_test.list
#vallist=lists/IRSDataset_TEST.list
#vallist=lists/FlyingThings3D_release_TEST.list
#dataset=sintel
#trainlist=lists/Sintel_normal_all.list
#vallist=lists/Sintel_normal_all.list
#vallist=lists/Sintel_normal_test.list
#dataset=sceneflow
#trainlist=lists/FlyingThings3D_release_TRAIN.list
#vallist=lists/FlyingThings3D_release_TEST.list
#vallist=lists/driving_release.list
#vallist=lists/monkaa_release.list
#dataset=middlebury
#trainlist=lists/MB2014_TRAIN.list
#vallist=lists/MB2014_TRAIN.list

#load_norm=True
startR=0
startE=0
endE=0
batchSize=8
#model=none
#model=models/dispnetcss_SIRS/dispnetcss_0_0.pth
#model=models/dispnetcss_SIRS/model_best.pth
#model=models/dispnetcss_sceneflow/model_best.pth
#model=models/normnets-sf-fl1050-n21.128.pth
#model=models/dispnormnet-sf-fl1050-d0.928-n16.727.pth
#model=models/dnfusionnet-sf-fl1050-d0.926-n16.192.pth
#model=models/dtonfusionnet-sf-fl1050-d0.973-n15.768.pth
#model=models/dtonnet-sf-fl1050-d0.921-n15.6.pth
#model=models/dtonnet-flying/dtonnet_2_0.pth
#model=models/dtonnet-sceneflow/dtonnet_0_9.pth
#model=data/cvpr2020/models/dispnetc-sf-fl1050-d1.09.pth
#model=models/dispnetc_ue4_sperate.pth
#model=models/dispnetc_fly0.pth
#model=models/dispnetc_ue4_total2.pth
#model=models/dispnetc_texture_alter.pth
#model=models/dispnetc_ue_mix.pth
#model=models/dispCSRes-1/dispnetcres_0_19.pth
#model=models/dispCSRes/model_best.pth
#model=models/model_best.pth
#model=models/dispCSRes-regression/dispnetcres_1_16.pth
#model=models/multicorrnet/multicorrnet_0_3.pth
#model=models/dispnetc_ue_norm2/model_best.pth
#model=models/${net}-ft-sintel/model_best.pth
#model=models/dtonnet-irs/dtonnet_2_19.pth
#model=models/dispnetc-irs/dispnetc_2_19.pth
model=models/dispnetc-ft3d-irs/dispnetc_2_13.pth
#model=models/dispnetc-ft-irs/dispnetc_3_18.pth
#model=models/finalize/dispnetc-sf-fl1050-d0.917.pth
#model=models/finalize/dispnetc-flying-d0.91.pth
#model=models/normnets-irs/normnets_0_29.pth
#model=models/dnfusionnet-irs/dnfusionnet_2_19.pth
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --dataset $dataset \
               --startRound $startR --startEpoch $startE --endEpoch $endE \
               --model $model #\
	       #--load_norm $load_norm 

