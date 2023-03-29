#!/bin/bash

##############################
## Import datasets          ##
##############################
DNAMES=('jigsaw' 'xtremespeech' 'gabhatecorpus' 'hatexplain')
DPATHS=('/Users/prl222/bias_datasets/civil_comments/all_data.csv'
        '/Users/prl222/bias_datasets/Xtremespeech_request'
        '/Users/prl222/bias_datasets/GabHateCorpus/GabHateCorpus_annotations.tsv'
        '/Users/prl222/bias_datasets/HateXplain-master/Data/dataset.json'
        )
for (( j=0; j<${#DNAMES[@]}; j++ ));
do
  echo "Data collection: ${DNAMES[$j]}"
  python hate_datasets.py \
    --d_name "${DNAMES[$j]}" \
    --d_path "${DPATHS[$j]}"
done

##############################
## KG Adaptation            ##
##############################
infer=('hierarchical' 'none')
weight=('docf' 'logits' 'multiNB')

for i in "${infer[@]}"
do
    for j in "${weight[@]}"
    do
      python kg_adaptation.py \
        --d_name jigsaw \
        --knowledge_graph_path /Users/prl222/language_resources/GSSO-master/gsso.owl \
        --identities_pretraining gender,sexual_orientation \
        --thr 0.5 \
        --match_method stem \
        --infer_method "$i" \
        --weight_f "$j"
    done
done
