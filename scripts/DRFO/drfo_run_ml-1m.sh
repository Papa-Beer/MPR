gpu_id=7
task_type="ml-1m"
epochs=500
for learning_rate in 1e-3
do
for weight_decay in 1e-5
do
for fair_reg in 10
do
for partial_ratio_male in 0.5
do
for partial_ratio_female in 0.4
do
for gender_train_epoch in 1000
do
for seed in 1
do 
for DRFO_specific_lr in 1e-3
do 
main_folder=./DRFO_result/MF_results_classifier_${task_type}_${seed}/change_ratio_and_epoch/partial_ratio_male_${partial_ratio_male}/partial_ratio_female_${partial_ratio_female}/gender_train_epoch_${gender_train_epoch}/
mkdir -p ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}_DRFO_specific_lr_${DRFO_specific_lr}
nohup python3 -u ./DRFO_main_program.py --gpu_id ${gpu_id} --learning_rate $learning_rate --partial_ratio_male $partial_ratio_male --partial_ratio_female $partial_ratio_female \
--data_path ./datasets/ --gender_train_epoch $gender_train_epoch --weight_decay $weight_decay --fair_reg $fair_reg --saving_path ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}_DRFO_specific_lr_${DRFO_specific_lr}/ --orig_unfair_model ./pretrained_model/ \
--result_csv ${main_folder}result.csv --data_name ${task_type} --seed ${seed} --num_epochs ${epochs} --DRFO_specific_lr ${DRFO_specific_lr} > ${main_folder}learning_rate_${learning_rate}_weight_decay_${weight_decay}_fair_reg_${fair_reg}_DRFO_specific_lr_${DRFO_specific_lr}/train.log 2>&1 &
done
done
done
done
done
done
done
done
