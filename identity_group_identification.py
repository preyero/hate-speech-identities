"""
Target identification  for extending sota DL models based on PLMs (transformer)
to pre-trained KG feature representations.
"""
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA mysteries of the machine
import random
import numpy as np
import json
import argparse
import pandas as pd
from typing import Dict, List
import pickle
import tensorflow as tf
import transformers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense,
                                     Dropout)
import owlready2.namespace
from baselines.target_classification.hate_measure.nn.layers import TargetIdentityLayer, UniTargetIdentityLayer
from baselines.target_classification.hate_measure.nn import classifiers
from baselines.target_classification.hate_target import keys
from baselines.target_classification.hate_target.utils import cv_wrapper, analyze_experiment
import functions.kg.utils as kg_utils
import functions.kg.indexing as kg_index
import functions.kg.weighting as kg_weight
import kg_adaptation as kg_adapt
from functions.helper import load_dict, save_dict

# Define relevant quantities
SAVE_DIR = './models'
DATA_PATH = f'{SAVE_DIR}/measuring-hate-speech.csv'
MODEL_TYPES = ['llm', 'hybrid']
TRANSFORMER_NAMES = ['roberta-base', 'roberta-large']
KG_PATH = './models/adaptation/gsso.owl'
FEXT_KWARGS_KEYS = ['kg_path', 'kg_name', 'weights_folder', 'identity_pretraining', 'd_pretrain', 'thr', 'match_method',
                    'infer_method', 'weight_f']
WEIGHTING_SAMPLES = ['none', 'unit', 'sqrt', 'log']
# ... name of multi-output binary layer (TargetIdentityLayer)
MULTI_MODEL_OUTPUTS = sorted(keys.target_groups)
# ... identity (group and subgroup) columns for training uni-output models
IDENTITIES = keys.target_groups + keys.target_cols
PIPELINE_KEYS = ['feature_extractor', 'model', 'kwargs']
# Train deterministic model
# https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
seed = 1
# set seed
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

##################
# Lexical-based functions
##################
def toxic_debias_load(lexicon_path: str):
    data = pd.read_csv(lexicon_path)
    # keep demographic descriptors:
    # offensive-not-minority          324
    # offensive-minority-reference     53
    # harmless-minority                26
    data_identity = data.loc[data.categorization != 'offensive-not-minority']
    # Filter gender and sexual orientation minorities
    data_gso = data_identity.loc[data_identity['gender/sex/sex_orientation']!=0]
    # offensive-minority-reference    33
    # harmless-minority               14
    return data_gso['word'].to_list()

def toxic_debias_predict(lexicon: List, data: pd.DataFrame, text_col: str):
    import regex as re
    descRe = re.compile(r"\b"+r"\b|\b".join(lexicon)+"\b", re.IGNORECASE)
    matches = data[text_col].apply(descRe.findall)
    y_preds = matches.astype(bool)
    return y_preds.values, [';'.join(match) for match in matches]

##################
# Transformer functions
##################
def load_mhs_dataset(save: bool = True):
    """ Load dataset from huggingface and prepare for training target identification model.
    Export to data_path"""
    if not os.path.exists(DATA_PATH):
        print('  importing from huggingface server')
        import datasets
        dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')
        df = dataset['train'].to_pandas()

        print('  preprocessing text')
        from baselines.target_classification.hate_target.utils import preprocess
        df.insert(15, 'predict_text', df['text'].apply(lambda x: preprocess(x)))

        print('  adding gso column from max(gender, sexuality) annotations.')
        gso_cols = ['target_gender', 'target_sexuality']
        df.insert(61, 'target_gso', df.apply(lambda row: max([row[gso_col] for gso_col in gso_cols]), axis=1))

        print(f'  exported to: {DATA_PATH}')
        if save:
            df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
        print(f'  imported from: {DATA_PATH}')
    return df


# Plot cross-validation results and incidence rates from analysis dict (multi-output models)
def plot_cv_from_analysis(analysis, export_name, x_labels=None):
    if x_labels is None:
        x_labels = sorted(keys.target_labels)

    import matplotlib.pyplot as plt
    import numpy as np

    n_groups = analysis['roc_aucs'].shape[1]

    incidence_rates = analysis['incidence_rate']
    sorted_idx = np.flip(np.argsort(incidence_rates))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plt.subplots_adjust(wspace=0.1)
    width = 0.28
    axes[0].bar(
        x=np.arange(n_groups) - width,
        height=analysis['precision'].mean(axis=0)[sorted_idx],
        width=width,
        yerr=np.std(analysis['precision'], axis=0)[sorted_idx],
        color='C0',
        edgecolor='black',
        error_kw={'capsize': 2},
        label='Precision')
    axes[0].bar(
        x=np.arange(n_groups),
        height=analysis['recall'].mean(axis=0)[sorted_idx],
        width=width,
        yerr=np.std(analysis['recall'], axis=0)[sorted_idx],
        color='C1',
        edgecolor='black',
        error_kw={'capsize': 2},
        label='Recall')
    axes[0].bar(
        x=np.arange(n_groups) + width,
        height=analysis['f1_scores'].mean(axis=0)[sorted_idx],
        width=width,
        yerr=np.std(analysis['f1_scores'], axis=0)[sorted_idx],
        color='C2',
        edgecolor='black',
        error_kw={'capsize': 2},
        label='F1 Score')

    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y')
    axes[0].set_axisbelow(True)

    width = 0.40
    axes[1].bar(
        x=np.arange(n_groups) - width / 2,
        height=analysis['roc_aucs'].mean(axis=0)[sorted_idx],
        width=width,
        yerr=np.std(analysis['roc_aucs'], axis=0)[sorted_idx],
        color='C4',
        edgecolor='black',
        error_kw={'capsize': 3},
        label='ROC AUC')
    axes[1].bar(
        x=np.arange(n_groups) + width / 2,
        height=analysis['pr_aucs'].mean(axis=0)[sorted_idx],
        width=width,
        yerr=np.std(analysis['pr_aucs'], axis=0)[sorted_idx],
        color='lightgrey',
        edgecolor='black',
        error_kw={'capsize': 3},
        label='PR AUC')

    for idx, rate in enumerate(analysis['incidence_rate'][sorted_idx]):
        axes[1].plot([idx + width, idx], [rate, rate], color='black', lw=2.5)

    axes[1].grid(axis='y')
    axes[1].set_axisbelow(True)

    for ax in axes:
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels(np.array(x_labels)[sorted_idx], ha='right', rotation=20)

    axes[0].legend(bbox_to_anchor=(0.5, 1.06), loc='center', ncol=3, prop={'size': 13})
    axes[1].legend(bbox_to_anchor=(0.5, 1.06), loc='center', ncol=2, prop={'size': 13})
    axes[0].set_ylim([0, 1.03])

    for ax in axes:
        ax.tick_params(labelsize=14)

    axes[0].set_ylabel('Metric', fontsize=17)

    plt.savefig(f'{export_name}.pdf', bbox_inches='tight')


def plot_cv_from_exp_files(exp_files: List[str], identity_col: str, model_names: List[str], export_folder: str):
    # Retrieve analysis of target_label from experiment files
    analyses, idxs = [], []
    for exp_file in exp_files:
        kwargs = load_dict(exp_file.replace('exp_file.pkl', 'other_kwargs'))
        soft = True if kwargs['labelling'] == 'soft' else False
        analysis = analyze_experiment(exp_file, soft=soft, verbose=True)
        analyses.append(analysis)
        # append index of prediction from target_label
        model_outputs = kwargs['model_output'].split(',')
        if identity_col in model_outputs:
            idxs.append(model_outputs.index(identity_col))
        else:
            raise Exception(f'Invalid model identity output. Select from list {model_outputs}')

    # Create analysis dict with identity col prediction for all models: scores and incident rates
    results = ['accuracy', 'chance', 'accuracy_by_chance', 'log_odds_difference', 'roc_aucs',
               'pr_aucs', 'f1_scores', 'precision', 'recall', 'incidence_rate']
    analysis = {k: [] for k in results}
    # ... take n folds of column idx: ndarrays: (n_folds, n_outputs).
    for i, full_analysis in enumerate(analyses):
        for k, v in full_analysis.items():
            if k != 'overall_loss' and k != 'label_loss' and k != 'incidence_rate':
                analysis[k].append(list(full_analysis[k][:, idxs[i]]))
            elif k == 'incidence_rate':
                analysis[k].append(full_analysis[k][idxs[i]])
    analysis = {k: np.array(v).T if k != 'incidence_rate' else np.array(v) for k, v in analysis.items()}

    export_file = os.path.join(export_folder, '_'.join(model_names))
    plot_cv_from_analysis(analysis, export_file, x_labels=model_names)
    return


##################
# Hybrid model functions
##################
def load_weights_from_root(weights_root, weight_f):
    # Load weights if path given
    if weight_f in kg_adapt.WEIGHT_BY_SCORE:
        # Dict: {IRI: weight}
        with open(f'{weights_root}.json', 'r') as fp:
            weights_dict = json.load(fp)
    elif weight_f in kg_adapt.WEIGHT_BY_MODEL:
        # sklearn.pipeline.Pipeline: {'vectorizer': sklearn.feature_extraction.text.TfidfVectorizer,
        #                             'model': sklearn.linear_model._logistic.LogisticRegression}
        weights_dict = kg_weight.import_ML_coefficients(weights_root)
    else:
        raise Exception(f'{weight_f}: Invalid weight f')
    return weights_dict


def __apply_entity_weights(ent_assert: list, kg: owlready2.namespace.Ontology, kg_dict: dict,
                           weights_dict: dict = None, infer_method: str = None, weight_f: str = None):
    """ Group label float and entities weights from lists of asserted IRIs
    return: pd.Series with 2 rows (g_label and terminology columns)
    """
    if weight_f in kg_adapt.WEIGHT_BY_MODEL:
        # Create vector of entity weights from model coefficients
        pipeline = weights_dict
        feature_names, model_coef = kg_weight.get_feature_names_and_weights(pipeline, weight_f)
        weights_dict = {c_iri: weight for c_iri, weight in zip(feature_names, model_coef)}

    terminology = {}
    # Get terminology dict {(IRI, label): w}
    for c_iri in ent_assert:
        if weight_f:
            # ... compute weights from asserted+inferred path.
            weights = [weights_dict[c_iri]]
            # update weight with the entities inferred in that path: distribution average
            if infer_method == 'hierarchical':
                from functions.kg.utils import get_hierarchical_info
                ent_infer = get_hierarchical_info(c_iri, kg)
                for c_iri_infer in ent_infer:
                    if c_iri_infer in weights_dict.keys():
                        # ... add weight of entities in inferred path with label if in weights
                        # (e.g., the infer entities from an assert entity that was in the infer path of another one).
                        weights.append(weights_dict[c_iri_infer])
                weight = np.mean(np.array(weights))
            elif infer_method == 'none':
                weight = weights[0]
            else:
                raise Exception(f'{infer_method} Invalid method for using KG structure to infer information '
                                f'about terminology. Method selected in the list:  ' + ', '.join(kg_utils.INFER_METHODS))
            terminology[c_iri, kg_dict[c_iri][0]] = weight
        else:
            # ... add all entities asserted
            terminology[(c_iri, kg_dict[c_iri][0])] = 1

    # Get group label
    if weight_f:
        if weight_f in kg_adapt.WEIGHT_BY_SCORE:
            # ... [-1, 1] bounds: average path weight of entities asserted in text.
            n = len(terminology.keys())
            g_label = sum(terminology.values()) / n if n > 0 else 0
        else:
            #
            # ... create features from asserted and inferred paths of asserted terminology
            ent_infer = kg_adapt.__get_inferred(ent_assert, kg, infer_method)
            # ... do prediction of this one instance
            Xi = [ent_assert + ent_infer]
            # .... take prob to being from class 1
            g_label = pipeline.predict_proba(Xi)[0, 1]
    else:
        g_label = 1 if len(terminology.values()) > 0 else 0
    return pd.Series([g_label, terminology])


def get_entities(df: pd.DataFrame,
                 text_col: str,
                 id_col: str,
                 kg_path: str,
                 match_method: str,
                 weights_root: str = None,
                 weight_f: str = None,
                 checkpoint_root: str = None,
                 verbose: bool = True) -> pd.DataFrame:
    """ Get list of entities found in texts :
    :return np.ndarray with KG entity lists in texts
    """

    msg = "weighted entity matching" if weight_f else "entity matching with whole KG"
    if verbose:
        print(f"Computing {msg}")
    # Create inverted index
    inv_index = kg_index.indexing_df(df, text_col, id_col, match_method)
    # Load KG from path
    kg = kg_utils.load_owl(kg_path)

    terminology_df = pd.DataFrame()
    check_asserted = f'{checkpoint_root}.pkl'
    # Entities asserted are the same until the weighting f part (checkroot: thr,sample level, match, infer)
    if not os.path.exists(check_asserted):
        if verbose:
            print('  matching entities')
        # Create KG dicts for entity matching ({entity: [label, synonym, etc]})
        kg_dict = kg_utils.get_kg_dict(kg)

        if weight_f:
            if weights_root:
                weights_dict = load_weights_from_root(weights_root, weight_f)
            if weight_f in kg_adapt.WEIGHT_BY_SCORE:
                # Filter only the ones in the weights vector of entity scores
                kg_dict = {c_iri: syn for c_iri, syn in kg_dict.items() if c_iri in weights_dict.keys()}
            elif weight_f in kg_adapt.WEIGHT_BY_MODEL:
                # Filter only the ones in model features seen during training
                kg_dict = {c_iri: syn for c_iri, syn in kg_dict.items()
                           if c_iri in weights_dict['vectorizer'].vocabulary_}
        terminology_df['ent_assert'] = kg_utils.get_entity_matches(df, inv_index, text_col, id_col, kg_dict, match_method)
        if checkpoint_root is not None:
            terminology_df['ent_assert'].to_pickle(check_asserted)
            print('  checkpoint to: {}'.format(check_asserted))
    else:
        print('  found checkpoint of matched entities. Importing from: {}'.format(check_asserted))
        terminology_df['ent_assert'] = pd.read_pickle(check_asserted)
    return terminology_df['ent_assert'].values


def get_weights(entities: np.ndarray,
                kg_path: str,
                weights_root: str = None,
                infer_method: str = None,
                weight_f: str = None,
                checkpoint_root: str = None,
                verbose: bool = True) -> pd.DataFrame:
    """ Terminology dictionary from text entities with:
    -- entities asserted in the text (from whole KG or in weights vect)
    -- weight: 1 or weight from the entity weights of their inferred path
    :return pd.DataFrame g_label and terminology columns
    """
    # Load KG from path
    kg = kg_utils.load_owl(kg_path)

    terminology_df = pd.DataFrame.from_dict({'ent_assert': entities})
    kg_dict = kg_utils.get_kg_dict(kg)
    if verbose:
        print('  getting terminology dict and group labels')
    # Weighted terminology columns have checkpoint if using a weights vector
    if weight_f:
        # Load weights if path given
        if weights_root:
            weights_dict = load_weights_from_root(weights_root, weight_f)
        check_wterminology = f'{checkpoint_root}-{weight_f}.pkl'
        if not os.path.exists(check_wterminology):
            if infer_method:
                terminology_df[['g_label', 'terminology']] = \
                    terminology_df['ent_assert'].apply(
                        lambda ent_assert: __apply_entity_weights(ent_assert=ent_assert,
                                                                  kg=kg,
                                                                  kg_dict=kg_dict,
                                                                  weights_dict=weights_dict,
                                                                  infer_method=infer_method,
                                                                  weight_f=weight_f)
                    )
                if checkpoint_root is not None:
                    terminology_df[['g_label', 'terminology']].to_pickle(check_wterminology)
                    print('  checkpoint to: {}'.format(check_wterminology))
            else:
                raise Exception("Required infer method to weight entities")
        else:
            print('  found checkpoint of weighted entities. Importing from: {}'.format(check_wterminology))
            terminology_df[['g_label', 'terminology']] = pd.read_pickle(check_wterminology)
    else:
        terminology_df[['g_label', 'terminology']] = \
            terminology_df['ent_assert'].apply(
                lambda ent_assert: __apply_entity_weights(ent_assert=ent_assert, kg=kg, kg_dict=kg_dict)
            )

    return terminology_df[['g_label', 'terminology']]


# Class functions to build Hybrid Pipeline
class WeightedKGEmbeddings:
    """ Extracts feature vectors from adapted KG of a text input, if any.

        Parameters
    ----------
    fext_kwargs : Dict
        Configuration of the embedding model for weighted KG that serve as input features.
        Required keys:
        -- kg_path: str.
        -- d_pretrain: str.
        -- thr: int.
        -- match_method: str.
        -- infer_method: str.
        -- weight_f: str.

        Arguments
    ----------
    vocab : List
        KG entities with weights from a metric-based or model-based weighting method.

    """

    def __init__(self, fext_kwargs=None):
        if not fext_kwargs:
            raise Exception(f'Missing KG adaptation arguments: {FEXT_KWARGS_KEYS}')
        self.fext_kwargs = fext_kwargs
        self.fext_kwargs['weights_root'] = self.get_weight_root_from_kwargs(self.fext_kwargs['weights_folder'])
        # Weighted entities
        weights = load_weights_from_root(fext_kwargs['weights_root'], fext_kwargs['weight_f'])
        if fext_kwargs['weight_f'] in kg_adapt.WEIGHT_BY_SCORE:
            vocab = weights.keys()
        else:
            vocab = weights['vectorizer'].vocabulary_.keys()
        self.vocab = list(vocab)

    # Utils
    def get_weight_root_from_kwargs(self, weights_folder):
        """ Get path to weights file from configuration dict """
        method_name, method_name_keys = self.fext_kwargs['thr'], ['match_method', 'infer_method', 'weight_f']
        for k in method_name_keys:
            method_name = '-'.join([method_name, self.fext_kwargs[k]])
        weights_root = f'{weights_folder}/' \
                       f'{self.fext_kwargs["kg_name"]}_' \
                       f'{self.fext_kwargs["d_pretrain"]}_' \
                       f'{self.fext_kwargs["identity_pretraining"]}_' \
                       f'{method_name}'
        return weights_root

    # Prepare input: entities in text
    def get_entities_from_texts(self, df, text_col, id_col, train_set=False):
        """ Return entities in dataframe using text and id columns """
        entities = get_entities(df=df, text_col=text_col, id_col=id_col,
                                kg_path=self.fext_kwargs['kg_path'],
                                match_method=self.fext_kwargs['match_method'],
                                weights_root=self.fext_kwargs['weights_root'],
                                weight_f=self.fext_kwargs['weight_f'], verbose=False)

        # Add number of matches if training set
        if train_set:
            self.fext_kwargs['n_matches'] = len(set([entity for subl in entities for entity in subl]))
        return entities

    # Feature extraction: vectors of entity feature importance
    def get_KG_feature_vectors(self, entities):
        """
        Return np.ndarray of weights from a list of lists of entities: [n_texts, n_entities]
        """
        # Compute weights of entities in texts
        terminology_df = get_weights(entities, self.fext_kwargs['kg_path'], self.fext_kwargs['weights_root'],
                                     self.fext_kwargs['infer_method'], self.fext_kwargs['weight_f'], verbose=False)
        # Create embeddings with weights of vocab entities (i.e., seen in training).
        embeddings = pd.DataFrame(
            list(map(lambda t_dict: {k[0]: v for k, v in t_dict.items()}, terminology_df['terminology'].to_list())),
            columns=self.vocab
        ).fillna(0.0)
        return embeddings.values


class TargetIdentityClassifierHybrid(Model):
    """ Classifies the target identity of a text input, if any.

    This models stacks a dense layer on top of an input weighted kg embedding
    model(e.g., weighted GSSO), followed by a multi-output classification layer.
    See the `TargetIdentityLayer` layer for details on the endpoint of this model.
    Importantly, multiple identities (or none) can be targeted in a given text input.


        Parameters
    ----------
    n_entities : int
        The number of entities in the weighted kg embeddings.
    n_dense : int
        The number of feedforward units after the feature exraction.
    dropout_rate : float
        The dropout rate applied to the dense layer.
    """

    # Init with all required arguments
    # - create each layer to apply in the call function
    def __init__(self, n_dense=64, dropout_rate=0.1, n_entities=None, uni_output=False):
        super(TargetIdentityClassifierHybrid, self).__init__()
        self.n_entities = n_entities
        self.n_dense = n_dense
        self.dense = Dense(n_dense, activation='relu')
        self.dropout = Dropout(dropout_rate)
        if uni_output:
            self.target_identity = UniTargetIdentityLayer()
        else:
            self.target_identity = TargetIdentityLayer()

    # Build model method:
    #  - create input of weight vectors using tf.keras.Input((), dtype=tf.float, name = 'input_terminology')
    #  - create network with all args ()
    #  - call function to get outputs
    #  - return class model
    @classmethod
    def build_model(cls, n_dense=64, dropout_rate=0.1, n_entities=None, uni_output=False):
        """Builds a model using the Functional API."""
        embeddings = tf.keras.Input(shape=(n_entities,), dtype=tf.float32, name='input_embeddings')
        network = cls(n_dense=n_dense,
                      dropout_rate=dropout_rate,
                      n_entities=n_entities,
                      uni_output=uni_output)
        outputs = network.call(inputs=embeddings)
        model = Model(inputs=embeddings, outputs=outputs)
        return model

    # Call function
    # with each layer fast foward x
    def call(self, inputs):
        """Forward pass. Inputs must be a list of length 3, with the first two
        entries being the transformer input, and the third entry as the
        severity.
        """
        # Apply dense layer with dropout on input embeddings
        x = self.dense(inputs)
        x = self.dropout(x)
        # Target identity prediction
        x = self.target_identity(x)
        return x

    # Get config: with model_kwargs
    def get_config(self):
        return {'n_dense': self.dense.units,
                'dropout_rate': self.dropout.rate}


# Exec function to cross-validate a Target Identification model and save to folder (refitting to all data).
# -- model_types: LLM (transformer based) and Hybrid (KG and weights files).
def run_target_prediction_model(data_path: str, save_folder: str, model_type: str, id_col: str = 'comment_id',
                                text_col: str = 'predict_text', threshold: int = 0.5, soft: bool = True,
                                weights: str = 'sqrt', learning_rate: float = 2.5e-6, epsilon: float = 1e-8,
                                early_stopping_min_delta: float = 0., early_stopping_patience: float = 3,
                                model_name: str = None, pooling: str = 'mean', mask_pool: bool = False,
                                fext_kwargs: Dict = None, n_dense: int = 256, dropout_rate: float = 0.05,
                                batch_size: int = 8, max_epochs: int = 10, n_folds: int = 5, val_frac: float = 0.15,
                                uni_output: bool = False, identity_training: str = 'target_gso'):
    # Check required input arguments
    if model_type == 'llm':
        if not model_name:
            raise Exception(f'Missing model_name input. Select from list {TRANSFORMER_NAMES}')
        print(f'Starting target prediction model training using transformers: '
              f'\n pooling {pooling} \n mask pool {mask_pool}')
    elif model_type == 'hybrid':
        if not fext_kwargs or not all([k in fext_kwargs.keys() for k in FEXT_KWARGS_KEYS]):
            raise Exception(f'Missing dict with feature extractor arguments '
                            f'specifying values for keys: {FEXT_KWARGS_KEYS}')
        print(f'Starting target prediction model training using weighted KG embeddings.')
    else:
        raise Exception(f'Invalid model type. Select from list: {MODEL_TYPES}')

    # Train multi-output or uni-output models
    if uni_output:
        # Uni_output could be the column to use as y
        if identity_training in IDENTITIES:
            model_outputs = [identity_training]
        else:
            raise Exception(f'Invalid identity for training {identity_training}. Select from list {IDENTITIES}')
    else:
        # Take names of multi-output TargetIdentityLayer
        model_outputs = MULTI_MODEL_OUTPUTS
    print(f'  model outputs: {model_outputs}')
    outputs_tag = ''.join([group.split('_')[1] for group in model_outputs])

    # Read in data
    if data_path == DATA_PATH:
        data = load_mhs_dataset()
    else:
        data = pd.read_csv(data_path)

    comments = data[[id_col, text_col]].drop_duplicates().sort_values(id_col)
    # Determine target identities
    agreement = data[[id_col] + model_outputs].groupby(id_col).agg('mean')
    agreement = agreement[model_outputs]
    is_target = (agreement >= threshold).astype('int').reset_index(level=0).merge(right=comments, how='left')

    # Extract data for training models
    x = is_target[text_col].values
    identities = is_target[model_outputs]
    # Assign labels (hard or soft labels)
    y_soft = [agreement[col].values[..., np.newaxis] for col in identities]
    y_hard = [identities[col].values.astype('int')[..., np.newaxis] for col in identities]
    if soft:
        y_true = y_soft
    else:
        y_true = y_hard
    # Assign weights to samples
    if weights == 'unit':
        sample_weights = data[id_col].value_counts().sort_index().values
    elif weights == 'sqrt':
        sample_weights = np.sqrt(data[id_col].value_counts().sort_index().values)
    elif weights == 'log':
        sample_weights = 1 + np.log(data[id_col].value_counts().sort_index().values)
    else:
        sample_weights = None
    # Create callback function
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=early_stopping_min_delta,
        restore_best_weights=True,
        patience=early_stopping_patience)
    # Create model parameters
    labelling = 'soft' if soft else 'hard'
    save_name = f'{labelling}_H{n_dense}_B{batch_size}_D{dropout_rate}'
    if model_type == 'llm':
        if model_name == 'roberta-base':
            tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
        elif model_name == 'roberta-large':
            tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
        else:
            raise Exception(f'Invalid model_name input. Select from list {TRANSFORMER_NAMES}')
        tokens = tokenizer(x.tolist(), return_tensors='np', padding=True)
        inputs = [tokens['input_ids'], tokens['attention_mask']]
        model_builder = classifiers.TargetIdentityClassifier.build_model
        model_kwargs = {
            'transformer': model_name,
            'max_length': tokens['input_ids'].shape[1],
        }
        if model_name == 'roberta-base' or model_name == 'roberta-large':
            model_kwargs['pooling'] = pooling
            model_kwargs['mask_pool'] = mask_pool
        base_model_save_name = model_name
    else:
        # Prepare input features through entity matching
        # model inputs will be the embeddings [n_texts, n_entities] generated from the feature_extractor
        feature_extractor = WeightedKGEmbeddings(fext_kwargs=fext_kwargs)
        entities = feature_extractor.get_entities_from_texts(df=comments, text_col=text_col, id_col=id_col, train_set=True)
        kg_features = feature_extractor.get_KG_feature_vectors(entities=entities)
        inputs = [kg_features]
        model_builder = TargetIdentityClassifierHybrid.build_model
        model_kwargs = {
            'n_entities': len(feature_extractor.vocab),
        }
        base_model_save_name = fext_kwargs['weights_root'].split('/')[-1]
    kwargs = {'n_dense': n_dense,
              'dropout_rate': dropout_rate,
              'uni_output': uni_output}
    model_kwargs = {**model_kwargs, **kwargs}

    # Export files to save_folder/model_type/<model-configuration>/<feature-extraction>
    export_folder = os.path.join(save_folder, model_type, f'{outputs_tag}_{save_name}', base_model_save_name)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    if not os.path.isdir(os.path.join(save_folder, model_type)):
        os.mkdir(os.path.join(save_folder, model_type))
    if not os.path.isdir(os.path.join(save_folder, model_type, f'{outputs_tag}_{save_name}')):
        os.mkdir(os.path.join(save_folder, model_type, f'{outputs_tag}_{save_name}'))
    if not os.path.isdir(export_folder):
        os.mkdir(export_folder)

    # Run cross-validation
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    try:
        from tensorflow.keras.optimizers.legacy import Adam
        print('imported legacy optimizer')
    except ModuleNotFoundError:
        # No legacy on cpu: ModuleNotFoundError: No module named 'tensorflow.keras.optimizers.legacy'
        from tensorflow.keras.optimizers import Adam
    # Limit GPU memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Number of GPU available: {}'.format(len(gpus)))
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    # Request specific GPU: Data Card 1:
    gpus = [x for x in gpus if x.name == '/physical_device:GPU:1']
    if len(gpus) > 0:
        gpu = gpus[0]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        print(f'Set visible {gpu}')
    compile_kwargs = {'optimizer': Adam(learning_rate=learning_rate, epsilon=epsilon),
                      'loss': 'binary_crossentropy'}
    cv_results = cv_wrapper(
        x=inputs,
        y=y_true,
        model_builder=model_builder,
        model_kwargs=model_kwargs,
        compile_kwargs=compile_kwargs,
        batch_size=batch_size,
        max_epochs=max_epochs,
        n_folds=n_folds,
        val_frac=val_frac,
        refit=True,
        refit_fold=True,
        verbose=True,
        callbacks=[callback],
        cv_verbose=True,
        unwrap_predictions=True,
        store_models=False,
        sample_weights=sample_weights)

    # save feature extraction and other kwargs
    other_kwargs = {'batch_size': batch_size,
                    'labelling': labelling,
                    'model_output': ','.join(model_outputs)}
    save_dict(other_kwargs, os.path.join(export_folder, 'other_kwargs'))
    save_dict(fext_kwargs, os.path.join(export_folder, 'fext_kwargs'))

    exp_file = os.path.join(export_folder, 'exp_file.pkl')
    results = {
        'x': x,
        'y_true': y_true,
        'y_soft': y_soft,
        'y_hard': y_hard,
        'y_pred': cv_results['test_predictions'],
        'train_idxs': cv_results['train_idxs'],
        'test_idxs': cv_results['test_idxs'],
        'test_scores': cv_results['test_scores'],
        'n_epochs': cv_results['n_epochs']
    }
    with open(exp_file, 'wb') as results_file:
        pickle.dump(results, results_file)

    # save model kwargs
    save_dict(model_kwargs, os.path.join(export_folder, 'model_kwargs'))
    model_file = os.path.join(export_folder, 'model.h5')
    if 'model_refit' in cv_results:
        # save weights
        cv_results['model_refit'].save_weights(model_file)
    return exp_file


# Loads pipeline from experiment file.
def model_load(model_folder: str):
    model_type = model_folder.split('/')[-3]
    model_kwargs = load_dict(f'{model_folder}/model_kwargs')
    if model_type == 'llm':
        # Input function: tokenizer
        if model_kwargs['transformer'] == 'roberta-base' or model_kwargs['transformer'] == 'roberta-large':
            tokenizer = transformers.RobertaTokenizer.from_pretrained(model_kwargs['transformer'])
        else:
            raise Exception(f'Transformer not in list {TRANSFORMER_NAMES}. Found: {model_kwargs["transformer"]}')
        # Model
        model = classifiers.TargetIdentityClassifier.build_model(**model_kwargs)
        model.load_weights(model_folder + '/model.h5')

        # To predict: tokenizer(texts, return_tensors='np', padding=True) to get [input_ids, attention mask]
        pipeline_kwargs = model_kwargs
        pipeline = {'feature_extractor': tokenizer, 'model':model, 'kwargs': pipeline_kwargs}
    elif model_type == 'hybrid':
        # Input function: weighted kg embeddings
        fext_kwargs = load_dict(f'{model_folder}/fext_kwargs')
        feature_extractor = WeightedKGEmbeddings(fext_kwargs)
        # Model
        model = TargetIdentityClassifierHybrid.build_model(**model_kwargs)
        model.load_weights(model_folder + '/model.h5')

        # To predict: get_entities_from_texts to get [entities], then get_weighted_kg_embeddings to get [embeddings]
        other_kwargs = load_dict(f'{model_folder}/other_kwargs')
        pipeline_kwargs = {**model_kwargs, **fext_kwargs, **other_kwargs}
        pipeline = {'feature_extractor': feature_extractor, 'model': model, 'kwargs': pipeline_kwargs}
    else:
        raise Exception(f'Invalid type of target identification model: {model_type}. Valid types: {MODEL_TYPES}')
    # use with input (tokenizer or the feature extractor.get_entities_from_texts)
    return pipeline


def order_join(entities, weights):
    """ Return sorted entities by descending weight separated by commas """
    sorted_entities = [x for _, x in sorted(zip(weights, entities), reverse=True)]
    return ';'.join(sorted_entities)


def model_predict(pipeline: Dict, data: pd.DataFrame, identity_col: str, text_col: str):
    # 0. Check required values
    if not all([k in pipeline.keys() for k in PIPELINE_KEYS]):
        raise Exception(f'Invalid pipeline for transformer or hybrid based '
                        f'identity group identification models: {PIPELINE_KEYS}')
    # 1. Get predictions
    model_outputs = pipeline['kwargs']['model_output'].split(',')
    x = data[text_col].values
    # Transformer-based models
    if 'transformer' in pipeline['kwargs'].keys():
        tokenizer, model = pipeline['feature_extractor'], pipeline['model']
        tokens = tokenizer(x.tolist(), return_tensors='np', max_length=pipeline['kwargs']['max_length'],
                           truncation=True, padding='max_length')
        inputs = [tokens['input_ids'], tokens['attention_mask']]
        outputs = model.predict(inputs, batch_size=pipeline['kwargs']['batch_size'], verbose=1)
    # Hybrid-based models
    else:
        feature_extractor, model = pipeline['feature_extractor'], pipeline['model']
        # ... custom id colum
        id_col = 'custom_id'
        data[id_col] = range(0, data.shape[0])
        data[id_col] = data[id_col].apply(lambda id: str(id))
        # ... feature extraction
        entities = feature_extractor.get_entities_from_texts(df=data, text_col=text_col, id_col=id_col)
        kg_features = feature_extractor.get_KG_feature_vectors(entities=entities)
        inputs = [kg_features]
        # ... classification
        outputs = model.predict(inputs, batch_size=pipeline['kwargs']['batch_size'], verbose=1)
    if type(outputs) != list:
        print(' uni output model predictions')
        outputs = [outputs]

    predict_idx = model_outputs.index(identity_col)
    y_trues, y_preds = data[identity_col].ravel(), outputs[predict_idx].ravel()
    # 2. Get interpretations
    if 'feature_extractor' in locals():
        # ... list of features in descending order by their weight values.
        kg = kg_utils.load_owl(feature_extractor.fext_kwargs['kg_path'])
        kg_dict = kg_utils.get_kg_dict(kg)
        vocab_iris = feature_extractor.vocab
        vocab = np.array([kg_dict[iri][0] for iri in vocab_iris])
        if pipeline['kwargs']['weight_f'] == 'multiNB':
            # MultiNB: Coefficients are log-probabilities of class 0: the highest probability for the negative class has the lowest absolute value
            cutoff, kg_features = np.absolute(np.log(0.5)), np.absolute(kg_features)
        else:
            cutoff = 0.0
        idx_pos = [np.argwhere((kg_features[i] > cutoff) & (kg_features[i] != 0.0)).flatten().tolist() for i in
                   range(0, len(kg_features))]
        idx_neg = [np.argwhere((kg_features[i] < cutoff) & (kg_features[i] != 0.0)).flatten().tolist() for i in
                   range(0, len(kg_features))]
        pos_matches = [order_join(vocab[idx_i], kg_features[i, idx_i]) for i, idx_i in enumerate(idx_pos)]
        neg_matches = [order_join(vocab[idx_i], kg_features[i, idx_i]) for i, idx_i in enumerate(idx_neg)]
        # ... save synonyms of the match with highest weight
        idxs = [np.argwhere(kg_features[i] == np.amax(kg_features[i])).flatten().tolist()
                if np.amax(kg_features[i]) != 0.0 else [] for i in range(0, len(kg_features))]
        high_syns = [','.join(kg_dict[vocab_iris[idxs[i][0]]]) if len(idxs[i]) > 0 else '' for i in
                     range(0, len(kg_features))]
        interpretations = [pos_matches, neg_matches, high_syns]
    else:
        interpretations = None
    return y_trues, y_preds, interpretations


def main():
    desc = " Train target identification models and export to save folder. " \
           " Use load function to import model pipeline from pkl experiment file and use if for new predictions."
    parser = argparse.ArgumentParser(description=desc)

    # Required arguments
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to dataset selected for training the models.",
                        )

    parser.add_argument("--save_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="Name of folder root to export models.",
                        )

    parser.add_argument("--model_type",
                        default=None,
                        type=str,
                        required=True,
                        choices=MODEL_TYPES,
                        help=f"Option of target identification models: {MODEL_TYPES}.",
                        )

    # Optional arguments
    parser.add_argument("--id_col",
                        default='comment_id',
                        type=str,
                        required=False,
                        help="Dataset argument: \n"
                             "-- name of comment ID column (default: comment_id).",
                        )

    parser.add_argument("--text_col",
                        default='predict_text',
                        type=str,
                        required=False,
                        help="Dataset argument: \n"
                             "-- name of text column (default: predict_text).",
                        )

    parser.add_argument("--threshold",
                        default=0.5,
                        type=float,
                        required=False,
                        help="Dataset argument: \n"
                             "-- threshold to binarize identity group labels "
                             "(default: 0.5 percentage of annotator agreement).",
                        )

    parser.add_argument("--soft",
                        action='store_false',
                        help="Training procedure: \n"
                             "-- If use proportion of annotators as \"label\" (default: True).",
                        )

    parser.add_argument("--weights",
                        default='sqrt',
                        type=str,
                        required=False,
                        choices=WEIGHTING_SAMPLES,
                        help="Training procedure: \n"
                             f"-- weighting samples by number of annotators (default: sqrt). "
                             f"Select from list: {WEIGHTING_SAMPLES}",
                        )
    parser.add_argument('--lr',
                        type=float,
                        default=2.5e-6,
                        help="Optimizer hyperparameter learning rate (default: 2.5e-6).")

    parser.add_argument('--epsilon',
                        type=float,
                        default=1e-8,
                        help="Optimizer hyperparameter epsilon (default: 1e-8).")

    parser.add_argument("--early_stopping_min_delta",
                        default=0.,
                        type=float,
                        required=False,
                        help="Training procedure: \n"
                             "-- min delta param to set early stopping on validation loss "
                             "(default: 0.001).",
                        )

    parser.add_argument("--early_stopping_patience",
                        default=3,
                        type=int,
                        required=False,
                        help="Training procedure: \n"
                             "-- patience param to set early stopping on validation loss "
                             "(default: 2).",
                        )

    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=False,
                        choices=[None] + TRANSFORMER_NAMES,
                        help="Model arguments: \n"
                             f"-- transformer name when using large language models (LLM) (default: None). "
                             f"Select from list {[None] + TRANSFORMER_NAMES}",
                        )

    parser.add_argument("--pooling",
                        default='mean',
                        type=str,
                        required=False,
                        choices=['mean', 'max'],
                        help="Model arguments: \n"
                             f"-- pooling layer when using LLM (default: mean). "
                             f"Select from list ['mean', 'max']",
                        )

    parser.add_argument("--mask_pool",
                        action='store_true',
                        help="Model arguments: \n"
                             f"-- mask layer when using LLM (default: False)",
                        )

    parser.add_argument("--kg_path",
                        default=None,
                        type=str,
                        required=False,
                        help="Model argument: \n"
                             "-- path to kg when training hybrid models "
                             "(default: None).",
                        )

    parser.add_argument("--weights_path",
                        default=None,
                        type=str,
                        required=False,
                        help="Model argument: \n"
                             "-- path of folder with entity weights for training hybrid models"
                             "(default: None).",
                        )

    parser.add_argument("--n_dense",
                        default=256,
                        type=int,
                        required=False,
                        help="Model arguments: \n"
                             f"-- number of hidden layers (default: 256)",
                        )

    parser.add_argument("--dropout_rate",
                        default=0.05,
                        type=float,
                        required=False,
                        help="Model arguments: \n"
                             f"-- dropout in feedforward layer (default: 0.05)",
                        )

    parser.add_argument("--batch_size",
                        default=8,
                        type=int,
                        required=False,
                        help="Cross-validation: \n"
                             f"-- size of training batches (default: 8)",
                        )

    parser.add_argument("--max_epochs",
                        default=10,
                        type=int,
                        required=False,
                        help="Cross-validation: \n"
                             f"-- max epochs (default: 10)",
                        )

    parser.add_argument("--n_folds",
                        default=5,
                        type=int,
                        required=False,
                        help="Cross-validation: \n"
                             f"-- number of folds (default: 5)",
                        )

    parser.add_argument("--val_frac",
                        default=0.15,
                        type=float,
                        required=False,
                        help="Cross-validation: \n"
                             f"-- fraction for validation (default: 0.15)",
                        )

    parser.add_argument("--uni_output",
                        action='store_true',
                        help="Model arguments: \n"
                             f"-- whether to use uni-dimensional output layer when training a model,"
                             f" -- e.g., for target_gso  (default: False)",
                        )

    parser.add_argument("--identity_training",
                        default='target_gso',
                        type=str,
                        required=False,
                        help="Model arguments: \n"
                             f"-- name of identity column in training dataset to use in uni-output model training. "
                             f"Select from list: {IDENTITIES} "
                             "(default: target_gso).",
                        )

    args = parser.parse_args()
    # parse kwargs from optional arguments (specified or default values)
    # optional arguments
    if args.weights_path and args.kg_path:
        # ... using a subset of the KG from weights vector
        method_name = args.weights_path.split('_')[-1].rsplit('.', 1)[-2]
        fext_kwargs = {'kg_path': args.kg_path,
                       'kg_name' :  args.kg_path.rsplit('.', 1)[-2].split('/')[-1],
                       'weights_folder': args.weights_path.rsplit('/', 1)[0],
                       'd_pretrain': args.weights_path.split('_')[-3],
                       'identity_pretraining': args.weights_path.split('_')[-2],
                       'thr': method_name.split('-')[-4],
                       'match_method': method_name.split('-')[-3],
                       'infer_method': method_name.split('-')[-2],
                       'weight_f': method_name.split('-')[-1]}

    else:
        fext_kwargs = None

    print('Training a target prediction model: {}\n Optional args: {}'.format(args, fext_kwargs))
    run_target_prediction_model(args.data_path, args.save_folder, args.model_type, args.id_col, args.text_col,
                                args.threshold, args.soft, args.weights, args.lr, args.epsilon,
                                args.early_stopping_min_delta, args.early_stopping_patience, args.model_name,
                                args.pooling, args.mask_pool, fext_kwargs, args.n_dense, args.dropout_rate,
                                args.batch_size, args.max_epochs, args.n_folds, args.val_frac, args.uni_output,
                                args.identity_training)
    return


if __name__ == "__main__":
    main()
