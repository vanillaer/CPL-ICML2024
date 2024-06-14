#!/bin/bash
# custom config
DATA=data
TRAINER=LaFTerUFT
CFG=vit_b32
dset="$1"
txt_cls=zero_shot
CUDA_VISIBLE_DEVICES=1 python LaFter.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/"${dset}".yaml \
--config-file configs/trainers/text_cls/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}" \
--lr 0.0005 \
--zero_shot \
--txt_cls ${txt_cls}
