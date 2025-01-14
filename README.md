[![DOI](https://zenodo.org/badge/616586386.svg)](https://zenodo.org/badge/latestdoi/616586386)

# kg-hateguard-eval

### Try out our hybrid models in this Demo! 👉  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://hate-speech-identities-demo.streamlit.app/)


This is the data and code in the paper [Knowledge-Grounded Target Group Language Recognition in Hate Speech](https://ebooks.iospress.nl/volumearticle/64009), together with conference [slides](./supplemental_material/submission650_sem23.pdf) and a [poster](./supplemental_material/poster_sem23.pdf)!


We use a Knowledge Graph (KG), and adapt it, to provide interpretability to deep learning predictions as to why hate speech texts may refer to particular target group identities ([Figure 1](./supplemental_material/Figure_1.pdf)). Specifically, by enriching probability scores with the entities at input that contribute to a prediction.

<p align="center">
 <img src="supplemental_material/Figure_1.png" alt="drawing" width="800" class="center"/>
</p>

Grounding the classification task in semantic knowledge gives context about the identity groups most impacted by these technologies. Our experiments show how *knowledge-grounded interpretations* help better understand model outcomes, the training data, and the ambiguious cases in human annotations.

## Project description

The project is organised in three folders. `data` contains the hate speech datasets, `baselines` the lexicon-based (System B) and transformer-based (System A) models used in the paper, and `models` the files from model training.

There are two main code files:
- *kg_adaptation.py*: script to learn weights for the KG entities based on a classification task (`./models/adaptation`).
- *identity_group_identification.py*: script to train text classifiers based on adapted KG features or Huggingface transformer embeddings [1].

If you are planning to use a KG as feature extractor in a text classification task, get in touch!

## Requirements

#### Conda Environment
Python 3.8.2 and install requirements.txt (generated using `pipreqs`).
```commandline
    $ conda create --name <env_name> python=3.8.2
    $ conda activate <env_name>
    (<env_name>) $ python -m pip install -r requirements.txt
    # Inside virtual environment to run notebooks with this conda environment
    (<env_name>) $ python -m ipykernel install --user --name=<env_name>
    (<env_name>) $ jupyter notebook
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

Follow instructions to install TF with GPU-enabled using a [Virtual Environment](https://www.tensorflow.org/install/pip) if applicable. Ensure to install requirements after setting up the virtual environment as indicated above (Conda Environmnt). To use a specific GPU, adjust `CUDA_VISIBLE_DEVICES` in identity_group_identification script.

## Paper experiments

All results from the paper are in `notebooks` folder. 

We provide two bash scripts to reproduce all models (`hate-speech-identities <username>$ bash notebooks/<script-name>.sh &> notebooks/<script-name>.log`), including the export of Jigsaw Sample (jigsaw_0.5_gendersexualorientation.csv). Please contact us if you would prefer us to share it directly. 

All paper results are shown in the following Jupyter notebooks:
- *1_data_statistics.ipynb*: statistics for the datasets used in the paper (Table 2).
- *2_KG_adaptation.ipynb*: evaluation for different weighting schemes in the KG adaptation phase.
- *3_identity_group_identification.ipynb*: evaluation of hybrid, transformers, and lexicon-based models (Figure 3, Table 3, Table 4, Table 5).

The excel files in `models/interpretations` contain the qualitative analyses of model errors and interpretability.

## Resources

- Jigsaw Toxicity Corpus (`all_data.csv`): download directly via [Kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data?select=all_data.csv) 
- Gab Hate Corpus (`GabHateCorpus_annotations.tsv`): download through this [link](https://osf.io/edua3/)
- HateXplain (`dataset.json`): obtained from this [repo](https://github.com/hate-alert/HateXplain/tree/master/Data)
- XtremeSpeech (`kenya_re.csv`): keep an eye on its [repo](https://github.com/antmarakis/xtremespeech), and contact the [authors](mailto:antmarakis@cis.lmu.de) to request for access.
- Gender, Sex, and Sexual Orientation Ontology (`gsso.owl`): v2.0.10, latest release at [repo](https://github.com/Superraptor/GSSO)

Meauring Hate Speech (`measuring-hate-speech.csv`) downloads directly from Huggingface when using `identity_group_identification.py` for model training.

## Baseline repositories

[1] [Targeted Identity Group Prediction in Hate Speech Corpora](https://github.com/dlab-projects/hate_target) by Sachdeva et al (2022). In Proceedings of the Sixth Workshop on Online Abuse and Harms (WOAH). Association for Computational Linguistics, Seattle, Washington (Hybrid), 231–244.

[2] [Challenges in Automated Debiasing for Toxic Language Detection](https://github.com/XuhuiZhou/Toxic_Debias/tree/main/data) by Zhou et al (2021). Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume.

### Citation


```bibtex
@incollection{reyero2023knowledge,
    title       = {Knowledge-Grounded Target Group Language Recognition in Hate Speech},
    author      = {Reyero Lobo, Paula and 
                   Daga, Enrico and 
                   Alani, Harith and 
                   Fernandez, Miriam},
    booktitle   = {Knowledge Graphs: Semantics, Machine Learning, and Languages},
    pages       = {1--18},
    year        = {2023},
    publisher   = {IOS Press}
}
```