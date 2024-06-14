#!/bin/bash
TAG="exp_ID-visualPT"     #exp_ID

LR="$1" 
IFS=',' read -ra LR <<< "$LR"
echo "LR: ${LR[@]}"

# Function to run the job
run_job() {
    OPTIM_SEED=${optim_seed} \
    VIS_ENCODER=${vis_encoder} \
    DATASET_NAME=${dataset_name} \
    SPLIT_SEED=${split_seed} \
    MODEL=${SETTING} \
    DATASET_DIR=${dataset_dir} \
    OUTPUT_DIR=${DIR} \
    LOSS_TYPE=${loss_type} \
    EPOCHS=${epoch_num} \
    LR=${lr} \
    DECAY=${decay} \
    BATCH_SIZE=${batch_size} \
    CANDIDATE_METHOD=${method} \
    TEMPERATURE=${TEMPERATURE} \
    CONF_QUANTILE=${CONF_QUANTILE} \
    CONF_THRESHOLD="quantile" \
    REGULAR_THRESHOLD=${REGULAR_THRESHOLD} \
    STEP_QUANTILE=${STEP_QUANTILE} \
    USE_SOFT_PARTIAL=${USE_SOFT_PARTIAL} \
    accelerate launch --config_file methods_config/accelerate_localtest_config.yml run_main_${learning_paradigm}.py \
                      --model_config ${SETTING}_config_PLL.yml --learning_paradigm ${learning_paradigm}
}


EPOCHS=(50)
DECAY=(0.05)
BATCH_SIZE=(64)
dataset_dirs=('data_')    #  add the path here containing datasets
vis_encoders=('ViT-B/32') #  'ViT-B/32' or 'ViT-L/14'
split_seeds=(500)         # This indicate the split for TRZSL, i.e., 500, 0, or 200. For other learning setting this is 500 as default.
dataset_names=('CUB' 'Flowers102'  'DTD' 'EuroSAT' 'FGVCAircraft' 'RESICS45') #  'RESICS45' 'Flowers102' 'EuroSAT' 'FGVCAircraft'
SETTINGS=('grip_visual' 'visual_fpl')   # Different prompt SETTINGS grip_visual visual_fpl
optim_seeds=(1 2 3)         # 1 2 3 are the seeds we used
loss_types=('cc')         # Choose among different loss func: 'cc' (default) 'rc_rc' (RC) 'lw_lw' (LW) 'rc_cav' (CAV)
methods=('CPL')      

TEMPERATUREs=(1.0)
USE_SOFT_PARTIALs=(False)
LAMBDAs=(1.0) 

learning_paradigms=('ul' 'ssl' 'trzsl') # Choose among: ul, ssl, trzsl


for dataset_dir in "${dataset_dirs[@]}"; do
for dataset_name in "${dataset_names[@]}"; do

if [ "$dataset_name" == "CUB" ]; then
STEP_QUANTILE=20
else
STEP_QUANTILE=10
fi

for learning_paradigm in "${learning_paradigms[@]}"; do

for vis_encoder in "${vis_encoders[@]}"; do
for optim_seed in "${optim_seeds[@]}"; do
for split_seed in "${split_seeds[@]}"; do

for epoch_num in "${EPOCHS[@]}"; do
for lr in "${LR[@]}"; do
for decay in "${DECAY[@]}"; do
for batch_size in "${BATCH_SIZE[@]}"; do
for SETTING in "${SETTINGS[@]}"; do
for loss_type in "${loss_types[@]}"; do

for USE_SOFT_PARTIAL in "${USE_SOFT_PARTIALs[@]}"; do
for method in "${methods[@]}"; do
# NOTE: CONF_QUANTILE is used to represent the hyperparameter (alpha*100) in the paper
# NOTE: REGULAR_THRESHOLD is used to represent the hyperparameter beta in the paper

    if [ "$method" == "CPL" ]; then
        
        if [ "$dataset_name" == "EuroSAT" ]; then
            CONF_QUANTILEs=(90)
            if [ "$learning_paradigm" == "trzsl" ]; then
                REGULAR_THRESHOLDs=("auto*2.0")  
            else
                REGULAR_THRESHOLDs=("0.80")
            fi

        elif [ "$dataset_name" == "Flowers102" ]; then 
            CONF_QUANTILEs=(60)
            if [ "$learning_paradigm" == "trzsl" ]; then
                REGULAR_THRESHOLDs=("auto*1.0")  
            else
                REGULAR_THRESHOLDs=("0.99")
            fi

        elif [ "$dataset_name" == "FGVCAircraft" ]; then 
            CONF_QUANTILEs=(90)
            if [ "$learning_paradigm" == "trzsl" ]; then
                REGULAR_THRESHOLDs=("auto*2.0")  
            else
                REGULAR_THRESHOLDs=("0.99")
            fi
            
        elif [ "$dataset_name" == "CUB" ]; then 
            CONF_QUANTILEs=(60)
            if [ "$learning_paradigm" == "trzsl" ]; then
                REGULAR_THRESHOLDs=("auto*2.50")  
            else
                REGULAR_THRESHOLDs=("0.99")
            fi

        elif [ "$dataset_name" == "DTD" ]; then 
            CONF_QUANTILEs=(60)
            if [ "$learning_paradigm" == "trzsl" ]; then
                REGULAR_THRESHOLDs=("auto*2.0")  
            else
                REGULAR_THRESHOLDs=("0.97")
            fi

        elif [ "$dataset_name" == "RESICS45" ]; then 
            CONF_QUANTILEs=(90)
            if [ "$learning_paradigm" == "trzsl" ]; then
                REGULAR_THRESHOLDs=("auto*2.0")
            else
                REGULAR_THRESHOLDs=("0.97")
            fi
        
        else 
            echo "Invalid Dataset name"
            exit 1
        fi
    else
        echo "Invalid method name"
        exit 1
    fi

for REGULAR_THRESHOLD in "${REGULAR_THRESHOLDs[@]}"; do
for TEMPERATURE in "${TEMPERATUREs[@]}"; do
for CONF_QUANTILE in "${CONF_QUANTILEs[@]}"; do

    LOG_FILE="script_results/log_${TAG}_${dataset_name}.txt"
    total_iterations=$((${#EPOCHS[@]} * \
                        ${#LR[@]} * \
                        ${#DECAY[@]} * \
                        ${#BATCH_SIZE[@]} * \
                        ${#dataset_dirs[@]} * \
                        ${#vis_encoders[@]} * \
                        ${#split_seeds[@]} * \
                        ${#dataset_names[@]} * \
                        ${#SETTINGS[@]} * \
                        ${#optim_seeds[@]} * \
                        ${#loss_types[@]} * \
                        ${#methods[@]} * \
                        ${#REGULAR_THRESHOLDs[@]} * \
                        ${#TEMPERATUREs[@]} * \
                        ${#USE_SOFT_PARTIALs[@]} * \
                        ${#learning_paradigms[@]}))
  
    echo "The loop will iterate $total_iterations times."

    common_id="dataset-${dataset_name}_setting-${SETTING}_lpardigm-${learning_paradigm}_encoder-${vis_encoder}_split-${split_seed}_seed-${optim_seed}_epoch-${epoch_num}_lr-${lr}_decay-${decay}_bs-${batch_size}_loss-${loss_type}_method-${method}_T-${TEMPERATURE}_regularThr-${REGULAR_THRESHOLD}_confQ-${CONF_QUANTILE}"
    DIR=./output/${dataset_name}/${SETTING}/${vis_encoder}_SplitSeed${split_seed}-${TAG}/SEED${optim_seed}/${common_id}
    
    if [ -d "$DIR" ]; then
        echo -e "------------\n Results are available in ${DIR}. Skip this job"
    else
        echo "======>>> Run this job and save the output to ${DIR}"
        run_job

        ACCURACY=$(grep 'Testset accuracy:' ${DIR}/log.txt | awk -F': ' '{print $2}')
        RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
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

done
done
done
done
done
done

