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