# Identity Group Identification in Hate Speech Corpora

This is the data and code in [A Semantically Enriched Approach for Identity Group
Identification in Hate Speech Corpora]()

## Project description

The project is organised in three main folders. `data` contains the hate speech datasets, `baselines` contains the lexical-based and transformer-based models 
considered in the paper, and `models` contains the model outputs.

There are three main files:
- *kg_adaptation.py*: script to learn weights for the KG entities (`./models/adaptation`).
- *identity_group_identification.py*: script to train models based on adapted KG features or Huggingface transformers [1].

## Requirements

#### Virtual Environment
Python 3.8.2 and install requirements.txt (generated using `pip freeze` command).
```commandline
    $ python3 -m venv ./<env_name>
    $ source <env_name>/bin/activate
    (<env_name>) $ python -m pip install -r requirements.txt
```

#### Docker
```commandline
# Build docker image
$ docker build . -t <docker-image>:tag

# Run interactive docker container
$ docker run -it --name <docker-container>:tag -v `pwd`:/app <docker-image>:tag
# ... with port to open jupyter notebooks
$ docker run -it -p 8888:8888 --name <docker-container>:tag -v `pwd`:/app <docker-image>:tag

# Inside the container to run notebooks
$ jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
# ... or from command line with nbconvert
$ jupyter nbconvert --to notebook --execute my_notebook.ipynb --output=result.ipynb
```
#### Optional (To train Transformer models on GPU)

Follow instructions to install TF with GPU-enabled using [Docker](https://www.tensorflow.org/install/docker) or a [Virtual Environment](https://www.tensorflow.org/install/pip) if applicable.

## Paper experiments

All results from the paper are in `notebooks` folder. 

We provide two bash scripts to reproduce all models (`hate-speech-identities <username>$ bash notebooks/<script-name>.sh`), including the export of Jigsaw Sample (jigsaw_0.5_gendersexualorientation.csv). Please contact us if you would prefer us to share it directly. 

All outputs are shown in the following Jupyter notebooks:
- *1_data_statistics.ipynb*: statistics for the datasets used in the paper (Table 1).
- *2_KG_adaptation.ipynb*: evaluation for different weighting schemes for the KG adaptation (Table 2).
- *3_identity_group_identification.ipynb*: evaluation of hybrid, transformers, and lexical-based models (Figure 3, 
Table 3, Table 4, Table 5).

The excel files in `models/interpretations` folder correspond to the error and entity analyses.

## Resources

- Jigsaw Toxicity Corpus (`all_data.csv`): download directly via [Kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data?select=all_data.csv) 
- Gab Hate Corpus (`GabHateCorpus_annotations.tsv`): download through this [link](https://osf.io/edua3/)
- HateXplain (`dataset.json`): obtained from this [repo](https://github.com/hate-alert/HateXplain/tree/master/Data)
- XtremeSpeech (`kenya_re.csv`): keep an eye on its [repo](https://github.com/antmarakis/xtremespeech). Please contact the [authors](mailto:antmarakis@cis.lmu.de) to request for access.
- Gender, Sex, and Sexual Orientation Ontology (`gsso.owl`): latest release at [repo](https://github.com/Superraptor/GSSO)

## Baseline repositories

[1] [Targeted Identity Group Prediction in Hate Speech Corpora](https://github.com/dlab-projects/hate_target) by 
Sachdeva et al (2022). In Proceedings of the Sixth Workshop on Online Abuse and Harms (WOAH). Association for 
Computational Linguistics, Seattle, Washington (Hybrid), 231–244.

[2] [Challenges in Automated Debiasing for Toxic Language Detection](https://github.com/XuhuiZhou/Toxic_Debias/tree/main/data) 
by Zhou et al (2021). Proceedings of the 16th Conference of the European Chapter of the Association for Computational 
Linguistics: Main Volume.



