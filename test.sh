net=fadnet
datapath=data
dataset=irs
trainlist=lists/IRS_TRAIN.list
vallist=lists/IRS_TEST.list

loss=loss_configs/test.json
outf_model=models/test/

logf=logs/${net}_${dataset}.log

devices=0,1,2,3
batchSize=4

startR=0
startE=0
endE=0
lr=1e-4
model=models/fadnet-irs.pth
python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --trainlist $trainlist --vallist $vallist \
               --dataset $dataset \
               --startRound $startR --startEpoch $startE --endEpoch $endE \
               --model $model 

