""" Create and save in results folder weighted KG and corresponding subset of toxicity corpus for evaluation """

import os, argparse
import operator, math
import time

import pandas as pd
import owlready2.namespace
import joblib
from typing import List

import hate_datasets as dc
import functions.kg.utils as kg_utils
import functions.kg.indexing as kg_index
import functions.kg.weighting as kg_weight
from functions.helper import save_dict

# Global variables: cant use - or _ as separators
THRS = [0.5]  # int (standard thresholding), str (different aggregation functions?)
MATCH_METHODS = kg_index.MATCH_METHODS
INFER_METHODS = kg_utils.INFER_METHODS
WEIGHT_BY_SCORE = kg_weight.WEIGHT_BY_SCORE
WEIGHT_BY_MODEL = kg_weight.WEIGHT_BY_MODEL
WEIGHT_FS = kg_weight.WEIGHT_FS

DNAMES = dc.DNAMES
OUTPUT_FOLDER = './models/adaptation'
CHECKPOINTS_FOLDER = './models/adaptation/checkpoints'
DATA_FOLDER = './data'


def collect_owl_from_path(kg_path: str,
                          output_folder: str = OUTPUT_FOLDER) -> owlready2.namespace.Ontology:
    """ Load saved version of ontology from local repository or import from owl path and save a copy """
    fname = kg_path.split('/')[-1]
    o_path = f'./{output_folder}/{fname}'
    if os.path.exists(o_path):
        kg = kg_utils.load_owl(o_path)
        print('Found OWL file in output folder. Importing.')
    else:
        kg = kg_utils.load_owl(kg_path)
        kg.save(file=o_path)
        print(f'OWL collected from {kg_path} and saved in output folder.')
    return kg


def disproportionate_stratified_sampling(n: int, df: pd.DataFrame, col_to_sample: List) -> pd.DataFrame:
    """
    Get sample of size n from df using stratified sample.
    Returning balanced samples of populations indicated by binary columns in col_to_sample list
    More info on disproportionate sampling: https://www.geeksforgeeks.org/stratified-sampling-in-pandas/
    """
    # 1. Create sampling dict: needed for balanced sampling with no repetitions
    # dict of index lists which are unique in each group in ascending order

    # a. get group sample sizes in ascending order
    sample_sizes = {gi: df.loc[df[gi] == 1].shape[0] for gi in col_to_sample}
    sample_sizes = dict(sorted(sample_sizes.items(), key=operator.itemgetter(1)))

    # b. get sample id lists for each group that are unique according to sample size ascending order
    sampling_dict, ids_inc = {}, []
    for g in sample_sizes.keys():
        sampling_dict[g] = df.loc[(df[g] == 1) & (~df.index.isin(ids_inc))].index.to_list()
        ids_inc += sampling_dict[g]

    # 2. Create balanced sampled df
    # df subset of n size

    # a. get subgroup sample sizes as per how many samples can be extracted from each subgroup given final sample size N
    sampling_sizes = {}
    n_remaining, g_remaining = n, list(sample_sizes.keys())
    for g in sample_sizes.keys():
        sample_size = sample_sizes[g]
        subg_n = math.ceil(n_remaining/len(g_remaining))
        sampling_sizes[g] = sample_size if subg_n > sample_size else subg_n
        # update remaining sample sizes and groups to sample from
        n_remaining = n_remaining - sampling_sizes[g]
        g_remaining.remove(g)

    # b. return the balanced sample of df by drawing randomly samples from each subgroup of its corresponding
    # sample size
    sampled_ids = []
    for g in sample_sizes.keys():
        g_sample = pd.Series(sampling_dict[g]).sample(n=sampling_sizes[g], random_state=1)
        sampled_ids += g_sample.to_list()

    sampled_df = df.loc[df.index.isin(sampled_ids)].copy()
    return sampled_df


def adaptation_subset(d: pd.DataFrame,
                      g_labels: dict,
                      dname: str,
                      thr: int,
                      identities: List = None,
                      data_folder: str = DATA_FOLDER):
    # Rule to determine n (max possible sample for Npos=Nneg)
    # Add 'identities' to df_train: binary column with sample of related and not related texts for KG adaptation.
    # Can be at group (e.g., gender) or subgroup (e.g., male) level.
    if identities is None:
        identities = ['gender', 'sexual_orientation']
    y_col = ''.join([i for identity in identities for i in identity.split('_')])
    print(f"Sampling distribution with thr={thr} for {y_col}")
    # Create binary columns with groups over thr (g_labels keys) and
    # nones (i.e., texts with no identity group labels over thr ("None"))
    g_labels = list(g_labels.keys()) if identities[0] in g_labels.keys() else \
        [subgi for subg in g_labels.items() for subgi in subg]
    g_labels_bin = []
    for g in list(g_labels):
        d.loc[:, f'{g}_{thr}'] = d[f'{g}'].apply(lambda perc: 1 if perc >= thr else 0)
        g_labels_bin.append(f'{g}_{thr}')
    d.loc[:, f'none_{thr}'] = d.apply(lambda row: 1 if sum(row[g_labels_bin]) == 0 else 0, axis=1)
    g_labels_bin.append(f'none_{thr}')
    print(f'  {d.loc[d[f"none_{thr}"] == 1,].shape[0]}/{d.shape[0]} samples with no identity annotations under {thr}')

    if len(identities) == 2:
        id_1, id_2 = identities
        # Take n (minimum samples of either positive group)
        d_1, d_2 = d.loc[d[f'{id_1}_{thr}'] == 1], d.loc[d[f'{id_2}_{thr}'] == 1]
        n_pos_min = min(d_1.shape[0], d_2.shape[0])
        print(f'  min {id_1} or {id_2} sample: {n_pos_min}')

        # Get balanced positive sample: with all samples from min pos group and n sample from the other pos group
        col_pos = [f'{g}_{thr}' for g in identities]
        if (d_1.loc[~d_1.index.isin(d_2.index)].shape[0] >= n_pos_min) or (d_2.loc[~d_2.index.isin(d_1.index)].shape[0] >= n_pos_min):
            # ... there are enough disjoint examples from the majority class to draw a balanced sample
            pos_df = disproportionate_stratified_sampling(2*n_pos_min, d, col_pos)
        else:
            # ... need to take n sample from both and remove duplicates
            pos_samples = [d_1.sample(n=n_pos_min, random_state=1), d_2.sample(n=n_pos_min, random_state=1)]
            pos_df = pd.concat(pos_samples, join='inner')
            pos_df = pos_df[~pos_df.duplicated()]
        neg_df = d.loc[(d[f'{id_1}_{thr}'] == 0) & (d[f'{id_2}_{thr}'] == 0)].copy()
    elif len(identities) == 1:
        col_pos = [f'{y_col}_{thr}']
        pos_df = d.loc[d[f'{y_col}_{thr}'] == 1].copy()
        neg_df = d.loc[d[f'{y_col}_{thr}'] == 0].copy()
    else:
        raise Exception(f'Adaptation not supported for more than 2 identities, provided {identities}')
    n_pos = pos_df.shape[0]
    pos_df[y_col] = n_pos * [1]
    print(f'  {n_pos} unique positive samples ')
    if len(identities) == 2:
        print(f'2*n ({n_pos_min}) = {2 * n_pos_min} - {2 * n_pos_min - n_pos} duplicates')
    for g in col_pos:
        print(f'  -- {g}: {pos_df.loc[pos_df[g] == 1,].shape[0]}')

    # Get balanced negative sample of n_pos sample size: stratification with disproportionate sampling
    col_to_stratify = [x for x in g_labels_bin if x not in col_pos]
    neg_df = disproportionate_stratified_sampling(n_pos, neg_df, col_to_stratify)
    neg_df[y_col] = n_pos * [0]
    print(f'  {neg_df.shape[0]} unique negative samples:')
    for g in col_to_stratify:
        print(f'  -- {g}: {neg_df.loc[neg_df[g] == 1,].shape[0]}')

    # Take df_eval as d.notin(df_train)
    df_train = pd.concat([pos_df, neg_df])
    # ... ensure that values in either pos_label are unique
    df_train = df_train[~df_train.duplicated(subset=df_train.columns.to_list()[:-1])]
    print(f' {df_train.shape[0]} unique train samples: '
          f'2*n ({n_pos}) = {2 * n_pos} - {2 * n_pos - df_train.shape[0]} duplicates:')
    print(f'  -- {y_col}: \n{df_train[y_col].value_counts()}')

    # Save pre-training corpus as CSV in data folder
    if data_folder:
        export_name = '{}_{}_{}'.format(dname, thr, y_col)
        o_path = f'./{data_folder}/{export_name}.csv'
        df_train.to_csv(o_path, index=False)
        print(f'  Pre-training corpus exported to {data_folder}: {export_name}')

    return df_train


def __get_inferred(ent_assert: list,
                   kg: owlready2.namespace.Ontology,
                   infer_method: str) -> List:
    """ Return list of all entities inferred in the list of asserted entities """
    # Use KG structure to infer new entity information
    if infer_method == 'hierarchical':
        # ... [c1.iri, c2.iri, c11.iri]
        from functions.kg.utils import get_hierarchical_info
        ent_infer = [c_infer for c_assert in ent_assert
                     for c_infer in get_hierarchical_info(c_assert, kg)]
    elif infer_method == 'none':
        ent_infer = []
    else:
        raise Exception(f'{infer_method} Invalid method for using KG structure to infer information about terminology.'
                        f'Method selected in the list:  ' + ', '.join(INFER_METHODS))
    return ent_infer


def __entity_matching(df: pd.DataFrame,
                      inv_index: kg_index.EntityMatching,
                      text_col: str,
                      id_col: str,
                      kg: owlready2.namespace.Ontology,
                      checkpoint_root: str,
                      match_method: str,
                      infer_method: str):
    print(f'Identifying entities asserted and inferred in train subset: infer_method = {infer_method}')
    matching_df = pd.DataFrame()
    check_asserted = f'{checkpoint_root}.pkl'
    if not os.path.exists(check_asserted):
        print('  matching entities')
        # Create KG dicts for entity matching ({entity: [label, synonym, etc]})
        kg_dict = kg_utils.get_kg_dict(kg)

        # Return list of entities asserted in the text
        matching_df['ent_assert'] = kg_utils.get_entity_matches(df, inv_index, text_col, id_col, kg_dict, match_method)
        matching_df['ent_assert'].to_pickle(check_asserted)
        print('  checkpoint to: {}'.format(check_asserted))
    else:
        print('  found checkpoint of matched entities. Importing from: {}'.format(check_asserted))
        matching_df['ent_assert'] = pd.read_pickle(check_asserted)

    check_inferred = f'{checkpoint_root}-{infer_method}.pkl'
    if not os.path.exists(check_inferred):
        print('  inferring information from asserted entities')
        matching_df['ent_infer'] = matching_df['ent_assert'].apply(
            lambda ent_assert: __get_inferred(ent_assert, kg, infer_method)
        )
        matching_df['ent_infer'].to_pickle(check_inferred)
        print('  checkpoint to: {}'.format(check_inferred))
    else:
        print('  found checkpoint of inferred entities. Importing from: {}'.format(check_inferred))
        matching_df['ent_infer'] = pd.read_pickle(check_inferred)

    # Return list of inferred entities
    return matching_df.apply(lambda row: row['ent_assert'] + row['ent_infer'], axis=1)


def __compute_weights(d_train: pd.DataFrame,
                      X_col: str,
                      y_col: str,
                      weighting_f: str,
                      o_path: str):
    print("Computing and saving entity weights to get weighted KG ({IRI: weight}): "
          f"weighting f={weighting_f}")
    if weighting_f in WEIGHT_BY_SCORE:
        # Exporting weights from ratios in positive and negative class
        ent_match_pos = d_train.loc[d_train[y_col] == 1, X_col]
        ent_match_neg = d_train.loc[d_train[y_col] == 0, X_col]

        # Computing the weights
        if weighting_f == 'docf':
            # Get document frequencies in positive and negative space (unique occurrences by number of docs)
            freq_pos = kg_weight.get_DocF(ent_match_pos)
            freq_neg = kg_weight.get_DocF(ent_match_neg)

            # Compute weights as the avg of the difference in both classes
            weights = kg_weight.get_ratio(freq_pos, freq_neg)
        else:
            raise Exception(f'{weighting_f} weight function not in list of scoring methods: {WEIGHT_BY_SCORE}')
        # Saving results:
        save_dict(weights, o_path)

    elif weighting_f in WEIGHT_BY_MODEL:
        # Exporting weights from the feature coefficients of LR model trained on entities
        pipeline = kg_weight.get_ML_coefficients(d_train, X_col, y_col, weighting_f)
        # Dist vect is the vectorizer and model. Save them with joblib.
        joblib.dump(pipeline, f'{o_path}.joblib')

    else:
        raise Exception(f'{weighting_f} Invalid method for weighting entities based on their distribution.'
                        f'Method selected in the list:  ' + ', '.join(WEIGHT_FS))


def kg_adaptation(dname: str,
                  kg_path: str,
                  identities: List[str],
                  **opt_config):
    # Parse with default config parameters
    config = {'thr': THRS[0],
              'match_method': MATCH_METHODS[0],
              'infer_method': INFER_METHODS,
              'weight_f': WEIGHT_FS}
    for k, v in opt_config.items():
        if k in config.keys():
            config[k] = v

    # Import processed df from data folder
    d, text_col, id_col, g_labels = dc.import_dataset(dname)
    y_col = ''.join([i for identity in identities for i in identity.split('_')])

    # Import and save kg to result folder
    kg = collect_owl_from_path(kg_path)
    kg_name = kg_path.rsplit('.', 1)[-2].split('/')[-1]

    # Draw pre-training corpus fom identities
    d_train = adaptation_subset(d, g_labels, dname, config["thr"], identities)

    # Entity matching: identify entities asserted and inferred in text using KG
    check_root = f'{CHECKPOINTS_FOLDER}/{kg_name}_{dname}_{y_col}_{config["thr"]}-{config["match_method"]}'
    # Method 3: how to create index to do the entity matching
    t0 = time.time()
    # ... create custom index col (not required for creating entity weights)
    d_train[id_col] = range(0, d_train.shape[0])
    d_train[id_col] = d_train[id_col].apply(lambda id: str(id))
    # ... create inverted index
    inv_index = kg_index.indexing_df(d_train, text_col, id_col, config["match_method"])
    print("Executed in %s seconds." % str(time.time() - t0))

    # Method 4: how to use KG structure to identify terminology (infer methods)
    for infer_method in config["infer_method"]:
        t0 = time.time()
        # ... do entity matching and infer information from kg
        d_train['entity_matches'] = __entity_matching(d_train,
                                           inv_index,
                                           text_col,
                                           id_col,
                                           kg,
                                           check_root,
                                           config["match_method"],
                                           infer_method)
        print("Executed in %s seconds." % str(time.time() - t0))

        for weight_f in config["weight_f"]:
            # Method 5: how to weight entities based on the training corpus context
            # Create or expand weights of a KG contextual utterance
            t0 = time.time()
            # ... get the path to output weights
            method_name = '-'.join([str(config["thr"]), config["match_method"], infer_method, weight_f])
            o_path = f'./{OUTPUT_FOLDER}/{kg_name}_{dname}_{y_col}_{method_name}'

            __compute_weights(d_train,
                              'entity_matches',
                              y_col,
                              weight_f,
                              o_path)
            print("Executed in %s seconds." % str(time.time() - t0))
            # ...to explore: weights = pd.DataFrame.from_dict(weights, orient='index')
            print(' Success exporting entity weights to: {}'.format(o_path))

    return


def main():
    desc = "Create and save weighted KG and its evaluation subset in result folder"
    parser = argparse.ArgumentParser(description=desc)

    # Required arguments
    parser.add_argument("--d_name",
                        default=None,
                        type=str,
                        required=True,
                        help=f"Pre-training corpus for the KG adaptation: {DNAMES}",
                        )

    parser.add_argument("--knowledge_graph_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to knowledge graph OWL file.",
                        )

    parser.add_argument("--identities",
                        default=None,
                        type=str,
                        required=True,
                        help="Column names in pre-training corpus for the identities (groups or subgroups) "
                             "based on which to assign weights to the KG"
                             " (up to 2 identity groups or subgroups, separated by ,)",
                        )

    # Optional arguments
    parser.add_argument("--thr",
                        default=0.5,
                        type=str,
                        required=False,
                        help="Configuration argument: \n"
                             "-- threshold to binarize labels (default: 0.5 percentage of annotator agreement).",
                        )

    parser.add_argument("--match_method",
                        default='stem',
                        type=str,
                        required=False,
                        help="Configuration argument: \n"
                             "-- method for matching algorithm to texts (default: stemming).",
                        )

    parser.add_argument("--infer_method",
                        default='hierarchical',
                        type=str,
                        required=False,
                        help="Configuration argument: \n"
                             "-- method for inferring information of entity from KG (default: use hierarchy).",
                        )

    parser.add_argument("--weight_f",
                        default='docf',
                        type=str,
                        required=False,
                        help="Configuration argument: \n"
                             "-- method for weighting entities based on distribution "
                             "(default: use document frequencies).",
                        )

    args = parser.parse_args()

    # parse kwargs from optional arguments (specified or default values)
    other_args = {'thr': float(args.thr),
                  'match_method': args.match_method,
                  'infer_method': [args.infer_method],
                  'weight_f': [args.weight_f]}

    print('Computing entity weights: {}\n Optional args: {}'.format(args, other_args))
    kg_adaptation(args.d_name,
                  args.knowledge_graph_path,
                  args.identities.split(','),
                  **other_args)
    return


if __name__ == "__main__":
    main()
