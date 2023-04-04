#!/bin/bash

# Configure system paths in case not automated
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
##############################
## 1. Transformers          ##
##############################
mkdir models/llm
mkdir models/llm/gso_soft_H256_B8_D0.05
mkdir models/llm/gso_soft_H256_B8_D0.05/roberta-base

python identity_group_identification.py \
   --data_path models/measuring-hate-speech.csv \
   --save_folder models \
   --model_type llm \
   --model_name roberta-base \
   --pooling mean \
   --lr 2.5e-6 \
   --epsilon 1e-8 \
   --early_stopping_min_delta 0. \
   --early_stopping_patience 3 \
   --n_dense 256 \
   --dropout_rate 0.05 \
   --batch_size 8 \
   --uni_output &> models/llm/gso_soft_H256_B8_D0.05/roberta-base/train_model.log

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
            --data_path models/measuring-hate-speech.csv \
            --save_folder models \
            --model_type hybrid \
            --kg_path models/adaptation/gsso.owl \
            --weights_path models/adaptation/gsso_jigsaw_gendersexualorientation_0.5-stem-${i}-${j}. \
            --lr 2.5e-6 \
            --epsilon 1e-8 \
            --early_stopping_min_delta 0. \
            --early_stopping_patience 3 \
            --n_dense 256 \
            --dropout_rate 0.05 \
            --batch_size 8  \
            --uni_output &> models/hybrid/gso_soft_H256_B8_D0.05/gsso_jigsaw_gendersexualorientation_0.5-stem-${i}-${j}/train_model.log
    done
done