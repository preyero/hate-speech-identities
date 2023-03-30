#!/bin/bash

PROJ_DIR=$(pwd)
##############################
## 1. Transformers          ##
##############################
# mkdir models/llm
#python target_identification.py \
#    --data_path ./models/measuring-hate-speech.csv \
#    --save_folder ./models \
#    --model_type llm \
#    --model_name roberta-base \
#    --pooling mean \
#    --lr 2.5e-6 \
#    --epsilon 1e-8 \
#    --early_stopping_min_delta 0. \
#    --early_stopping_patience 3 \
#    --n_dense 256 \
#    --dropout_rate 0.05 \
#    --batch_size 8 \
#    --uni_output &> ./models/llm/roberta-base_soft_H256_B8_D0.05.log

##############################
## 2. Ne-Sy                 ##
##############################
infer=('none' 'hierarchical')
weight=('docf' 'logits' 'multiNB')
mkdir models/hybrid
mkdir models/hybrid/gso_soft_H256_B8_D0.05

for i in "${infer[@]}"
do
    for j in "${weight[@]}"
    do
        # Hybrid: uni
        mkdir models/hybrid/gso_soft_H256_B8_D0.05/gsso_jigsaw_gendersexualorientation_0.5-stem-${i}-${j}
        python identity_group_identification.py \
            --data_path "$PROJ_DIR"/models/measuring-hate-speech.csv \
            --save_folder "$PROJ_DIR"/models \
            --model_type hybrid \
            --kg_path "$PROJ_DIR"/models/adaptation/gsso.owl \
            --weights_path "$PROJ_DIR"/models/adaptation/gsso_jigsaw_gendersexualorientation_0.5-stem-${i}-${j}. \
            --lr 2.5e-6 \
            --epsilon 1e-8 \
            --early_stopping_min_delta 0. \
            --early_stopping_patience 3 \
            --n_dense 256 \
            --dropout_rate 0.05 \
            --batch_size 8  \
            --uni_output >> models/hybrid/gso_soft_H256_B8_D0.05/gsso_jigsaw_gendersexualorientation_0.5-stem-${i}-${j}/train_model.log
    done
done