#!/bin/bash
TAG="exp_ID-CPL"     #exp_ID

LR="$1" 
IFS=',' read -ra LR <<< "$LR"
echo "LR: ${LR[@]}"


run_job() {
    LOSS_TYPE=${loss_type} \
    CONF_THRESHOLD="quantile" \
    CANDIDATE_METHOD=${method} \
    CONF_QUANTILE=${CONF_QUANTILE} \
    REGULAR_THRESHOLD=${REGULAR_THRESHOLD} \
    TEMPERATURE=${TEMPERATURE} \
    UPDATE_FREQ=${UPDATE_FREQ} \
    USE_SOFT_PARTIAL=False \
    python LaFTer.py \
    --root ${DATA} \
    --epochs ${epoch_num} \
    --trainer ${TRAINER} \
    --seed ${optim_seed} \
    --dataset-config-file configs/datasets/"${dataset_name}".yaml \
    --config-file configs/trainers/text_cls/${CFG}.yaml \
    --output-dir ${DIR} \
    --lr ${lr} \
    --txt_cls ${txt_cls} \
    --use_candidate \
    DATASET.IMBALANCE_RATIO ${IMBALANCE_RATIO}
}   

EPOCHS=(50)
DATA=data
TRAINER=LaFTer
CFG=vit_b32
dataset_names=('oxford_flowers' 'cifar100' 'eurosat' 'dtd' 'ucf101' 'caltech101') # 'dtd' 'ucf101' 'oxford_flowers' 'caltech101'  'eurosat'
txt_cls=lafter
optim_seeds=(1 2 3) # 1 2 3 are the seeds we used
methods=('CPL') 
loss_types=('cc')
TEMPERATUREs=(0.25)
UPDATE_FREQs=(10)

for dataset_name in "${dataset_names[@]}"; do
# if [ "$dataset_name" == "cifar100" ]; then
IMBALANCE_RATIOs=(0.0)
# else
#     IMBALANCE_RATIOs=(0.0)
# fi
for optim_seed in "${optim_seeds[@]}"; do
for epoch_num in "${EPOCHS[@]}"; do
for lr in "${LR[@]}"; do
for loss_type in "${loss_types[@]}"; do
for method in "${methods[@]}"; do
# NOTE: CONF_QUANTILE is used to represent the hyperparameter (alpha*100) in the paper
# NOTE: REGULAR_THRESHOLD is used to represent the hyperparameter beta in the paper

    if [ "$method" == "CPL" ]; then
        
        if [ "$dataset_name" == "eurosat" ]; then
            CONF_QUANTILEs=(60)   
            REGULAR_THRESHOLDs=("auto*0.75")   

        elif [ "$dataset_name" == "caltech101" ]; then
            CONF_QUANTILEs=(90)   
            REGULAR_THRESHOLDs=("auto*0.5")   

        elif [ "$dataset_name" == "cifar100" ]; then
            CONF_QUANTILEs=(90)   
            REGULAR_THRESHOLDs=("auto*0.75")   

        elif [ "$dataset_name" == "dtd" ]; then
            CONF_QUANTILEs=(45)   
            REGULAR_THRESHOLDs=("auto*1.0")   

        elif [ "$dataset_name" == "oxford_flowers" ]; then 
            CONF_QUANTILEs=(90)   
            REGULAR_THRESHOLDs=("auto*0.75")   

        elif [ "$dataset_name" == "ucf101" ]; then
            CONF_QUANTILEs=(45)   
            REGULAR_THRESHOLDs=("auto*0.5")   

        else 
            echo "Invalid dataset name"
            exit 1
        fi
    else
        echo "Invalid rate for method"
        exit 1
    fi
for REGULAR_THRESHOLD in "${REGULAR_THRESHOLDs[@]}"; do
for CONF_QUANTILE in "${CONF_QUANTILEs[@]}"; do
for UPDATE_FREQ in "${UPDATE_FREQs[@]}"; do
for TEMPERATURE in "${TEMPERATUREs[@]}"; do
for IMBALANCE_RATIO in "${IMBALANCE_RATIOs[@]}"; do

    LOG_FILE="script_results/log_${TAG}_${dataset_name}.txt"
    
    total_iterations=$((${#EPOCHS[@]} * ${#LR[@]} * ${#dataset_names[@]} * ${#optim_seeds[@]} * ${#loss_types[@]} * ${#methods[@]} * ${#REGULAR_THRESHOLDs[@]} * ${#CONF_QUANTILEs[@]} * ${#TEMPERATUREs[@]} * ${#UPDATE_FREQs[@]} * ${#IMBALANCE_RATIOs[@]}))

    echo "The loop will iterate $total_iterations times."

    common_id="dataset-${dataset_name}_seed-${optim_seed}_epoch-${epoch_num}_lr-${lr}_loss-${loss_type}_method-${method}_regularThr-${REGULAR_THRESHOLD}_confQ-${CONF_QUANTILE}_updateFreq-${UPDATE_FREQ}_T-${TEMPERATURE}_imbalanceR-${IMBALANCE_RATIO}"
    DIR=./output/${dataset_name}/${TRAINER}_${CFG}-${TAG}/SEED${optim_seed}/${common_id}
    
    if [ -d "$DIR" ]; then
        echo -e "------------\n Results are available in ${DIR}. Skip this job"
    else
        echo "======>>> Run this job and save the output to ${DIR}"
        run_job

        TEST_ACCURACY=$(grep 'Testset accuracy:' ${DIR}/log.txt | awk -F': ' '{print $2}')
        BEST_ACCURACY=$(grep 'Best Accuracy:' ${DIR}/log.txt | awk -F': ' '{print $2}')
        RECORD="id: ${common_id} ACC: ${TEST_ACCURACY}, best accuracy: ${BEST_ACCURACY}"
        echo "${RECORD}" | tee -a ${LOG_FILE}
        echo "${RECORD}" >> ${DIR}/log.txt
    fi

done
done
done
done

done
done
done
done
done
done
done
