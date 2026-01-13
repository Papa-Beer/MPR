seed=1
dataset="ml-1m"
cuda='7'
for male_ratio in 0.5
    do 
    for female_ratio in 0.1 0.2 0.3 0.4
       do
       for gender_train_epoch in 1000
         do 
         nohup python3 DRFO_main_predict_sensitive_with_sst_batch.py \
         --partial_ratio_male ${male_ratio}\
         --partial_ratio_female ${female_ratio}\
         --seed ${seed}\
         --gpu_id ${cuda}\
         --gender_train_epoch ${gender_train_epoch} \
         --dataset ${dataset} \
         --orig_unfair_model ./pretrained_model \
         --workspace ./DRO/workspace/ &
         done 
       done 
    done 