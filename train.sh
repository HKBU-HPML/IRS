#dnn="${dnn:-dispnormnet}"
#dnn="${dnn:-dtonfusionnet_test}"
dnn="${dnn:-dtonnet}"
source exp_configs/$dnn.conf

python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE --endEpoch $endE \
               --model $model \
               --maxdisp $maxdisp \
	       --manualSeed 1024 \

