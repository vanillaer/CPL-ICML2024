#!/bin/bash
# custom config
TAG="exp_ID-original"     #exp_ID

LR="$1" 
IFS=',' read -ra LR <<< "$LR"
echo "LR: ${LR[@]}"


DATA=data
TRAINER=LaFTer
CFG=vit_b32
dataset_names=('oxford_flowers' 'cifar100' 'eurosat' 'dtd' 'ucf101' 'caltech101') # 'dtd' 'ucf101' 'oxford_flowers' 'caltech101'  'eurosat'
optim_seeds=(1 2 3) # 1 2 3 are the seeds we used
txt_cls=lafter

for dataset_name in "${dataset_names[@]}"; do
# if [ "$dataset_name" == "cifar100" ]; then
IMBALANCE_RATIOs=(0.0)
# else
#     IMBALANCE_RATIOs=(0.0)
# fi
for optim_seed in "${optim_seeds[@]}"; do
for lr in "${LR[@]}"; do
for IMBALANCE_RATIO in "${IMBALANCE_RATIOs[@]}"; do

LOG_FILE="script_results/log_${TAG}_${dataset_name}.txt"
total_iterations=$((${#dataset_names[@]} * ${#optim_seeds[@]} * ${#IMBALANCE_RATIOs[@]}))
echo "The loop will iterate $total_iterations times."
common_id="dataset-${dataset_name}_seed-${optim_seed}_imbalanceR-${IMBALANCE_RATIO}"
DIR=./output/${dataset_name}/${TRAINER}_${CFG}-${TAG}/SEED${optim_seed}/${common_id}


if [ -d "$DIR" ]; then
    echo -e "------------\n Results are available in ${DIR}. Skip this job"
else
    echo "======>>> Run this job and save the output to ${DIR}"

    python LaFTer.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/"${dataset_name}".yaml \
    --config-file configs/trainers/text_cls/${CFG}.yaml \
    --output-dir ${DIR} \
    --lr ${lr} \
    --txt_cls ${txt_cls} \
    --seed ${optim_seed} \
    DATASET.IMBALANCE_RATIO ${IMBALANCE_RATIO} \

    TEST_ACCURACY=$(grep 'Testset accuracy:' ${DIR}/log.txt | awk -F': ' '{print $2}')
    BEST_ACCURACY=$(grep 'Best Accuracy:' ${DIR}/log.txt | awk -F': ' '{print $2}')
    RECORD="id: ${common_id} ----> test * accuracy: ${TEST_ACCURACY}, best accuracy: ${BEST_ACCURACY}"
    echo "${RECORD}" | tee -a ${LOG_FILE}
    echo "${RECORD}" >> ${DIR}/log.txt
fi

done
done
done
done

