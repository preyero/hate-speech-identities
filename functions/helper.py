import json
from typing import Dict, List
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def save_dict(mydict: Dict, path: str):
    with open(f'{path}.json', 'w') as fp:
        json.dump(mydict, fp)


def load_dict(path: str):
    with open(f'{path}.json', 'r') as fp:
        mydict = json.load(fp)
    return mydict


def get_metrics(y_trues: List[int],
                y_pred_hard: List[int],
                y_preds: List[float] = None,
                by_n: int = 2) -> Dict[str, List[float]]:
    metrics = {
        'Accuracy': [round((y_true == y_pred).mean() * 100, by_n) for (y_true, y_pred) in zip(y_trues, y_pred_hard)],
        'Chance': [round((1 - y_true.mean()) * 100, by_n) for (y_true, y_pred) in zip(y_trues, y_pred_hard)],
        'F1': [round(f1_score(y_true, y_pred) * 100, by_n) for (y_true, y_pred) in zip(y_trues, y_pred_hard)]
        }
    if y_preds:
        thr_agnostic = {
            'ROC AUC': [round(roc_auc_score(y_true, y_pred) * 100, by_n) for (y_true, y_pred) in zip(y_trues, y_preds)],
            'PR AUC': [round(average_precision_score(y_true, y_pred) * 100, by_n) for (y_true, y_pred) in
                       zip(y_trues, y_preds)]}
        metrics = {**metrics, **thr_agnostic}
    return metrics


# Functions for error and entity analyses
def draw_sample(d_subset, n, random_seed):
    """ Draw n sample with at least all available samples """
    n_subset = d_subset.shape[0]
    n_available = n if n < n_subset else n_subset
    print(f' adding {n_available} samples')
    return d_subset.sample(n_available, random_state=random_seed).to_list()


def sample_error_partition(d, y_true_col, y_pred_col, id_col, sample_col, sample_size: int, partition_range: float, random_seed=1):
    """ Create binary column with random error samples from a 4-quartile partition with predicted probability """
    n = round(sample_size/4)
    sample_ids = []
    fn, fp = d.loc[(d[y_true_col] >= 0.5) & (d[y_pred_col] < 0.5)], d.loc[(d[y_true_col] < 0.5) & (d[y_pred_col] >= 0.5)]
    pos_partition_proba, neg_partition_proba = 0.5+partition_range, 0.5-partition_range
    # append FNs
    sample_ids = sample_ids + draw_sample(fn.loc[fn[y_pred_col] < neg_partition_proba, id_col], n, random_seed)
    sample_ids = sample_ids + draw_sample(fn.loc[fn[y_pred_col] >= neg_partition_proba, id_col], n, random_seed)
    # append FPs
    sample_ids = sample_ids + draw_sample(fp.loc[fp[y_pred_col] < pos_partition_proba, id_col], n, random_seed)
    sample_ids = sample_ids + draw_sample(fp.loc[fp[y_pred_col] >= pos_partition_proba, id_col], n, random_seed)
    #
    d[sample_col] = [0]*d.shape[0]
    d.loc[d[id_col].isin(sample_ids), sample_col] = 1
    return d


def sample_true_partition(d, y_true_col, y_pred_col, id_col, sample_col, sample_size: int, partition_range: float, random_seed=1):
    """ Export true predictions sample:
    e.g., 0.125 partition range: [0.5, 0.625), [0.625, 0.75), [0.75, 0.825), [0.825, 1.0] """
    n = round(sample_size/4)
    sample_ids = []
    high_tp, low_tp = d.loc[(d[y_true_col] >= 0.5) & (d[y_pred_col] >= 0.75)], d.loc[(d[y_true_col] >= 0.5) & (d[y_pred_col] >= 0.5) & (d[y_pred_col] < 0.75)]
    high_partition_proba, low_partition_proba = 0.75 + partition_range, 0.75 - partition_range
    # append low probable TPs
    sample_ids += draw_sample(low_tp.loc[low_tp[y_pred_col] < low_partition_proba, id_col], n, random_seed)
    sample_ids += draw_sample(low_tp.loc[low_tp[y_pred_col] >= low_partition_proba, id_col], n, random_seed)
    # append high probable TPs
    sample_ids += draw_sample(high_tp.loc[high_tp[y_pred_col] < high_partition_proba, id_col], n, random_seed)
    sample_ids += draw_sample(high_tp.loc[high_tp[y_pred_col] >= high_partition_proba, id_col], n, random_seed)
    #
    d[sample_col] = [0]*d.shape[0]
    d.loc[d[id_col].isin(sample_ids), sample_col] = 1
    return d


def find_elbow(y, s=3):
    """  identify the elbow point of a line plot of y-values in descending order implementing the elbow method """
    from kneed import KneeLocator

    # Find the knee/elbow point using the KneeLocator
    kl = KneeLocator(range(len(y)), y, curve='convex', direction='decreasing', S=s)
    kl.plot_knee()
    elbow_point = kl.knee

    return elbow_point


def entities_in_categories(sample, categories, categories_col, id_col, model_name):
    """ include all entities matched in error categories """
    detect_ids = []
    for category in categories:
        detect_ids += sample.loc[sample[categories_col].str.contains(category), id_col].to_list()
    category_sample = sample.loc[sample[id_col].isin(detect_ids)]
    print(category_sample['dataset'].value_counts())
    entities = [entity for pos_entities in category_sample[f'{model_name}_pos_matches'].dropna().to_list() for entity
                in pos_entities.split(';')]
    return entities