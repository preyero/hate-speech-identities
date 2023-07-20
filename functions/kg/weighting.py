""" sklearn functions to weight entities based on scores or hybrid models coefficients """

import collections
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

WEIGHT_BY_SCORE = ['docf', 'tfidf']
WEIGHT_BY_MODEL = ['logits', 'multiNB']
WEIGHT_FS = WEIGHT_BY_SCORE + WEIGHT_BY_MODEL


# A. Weighting entities from distribution scores.
def get_DocF(occurrence_col):
    # Time entity appears in documents by number of documents: [0, 1]
    # ... number of documents
    n_doc = occurrence_col.shape[0]
    # ... unique occurrences
    unique_occ = occurrence_col.apply(lambda occs: list(set(occs)))
    # ... get times in the corpus
    unique_occ = [subl for l in unique_occ.to_list() for subl in l]
    unique_occ_counts = collections.Counter(unique_occ)
    return {c: counts / n_doc for c, counts in unique_occ_counts.items()}


def get_ratio(pos_dict, neg_dict):
    """ Compute average between two dicts {IRI: weight}"""
    weight_vect = {}
    all_c = set(list(pos_dict.keys()) + list(neg_dict.keys()))
    for c in all_c:
        # ... weight 0 to entities found in only one space
        pos_val = pos_dict[c] if c in pos_dict.keys() else 0
        neg_val = neg_dict[c] if c in neg_dict.keys() else 0
        weight_vect[c] = (pos_val - neg_val) / 2
    return weight_vect


# B. Weighting entities from model weights.
def return_doc(doc):
    """ Dummy function to use sklearn text feature extraction of tokanized texts (i.e., IRIs) """
    return doc


def get_ML_coefficients(d_train, X_col, y_col, weight_f):
    """ Create pipeline to train ML on entity features """
    # Exporting weights from the feature coefficients of LR model trained on entities
    X, y = d_train[X_col], d_train[y_col]

    # Extract features from list of entities as if they were the tokenized text:
    tfidf = TfidfVectorizer(analyzer='word',
                            tokenizer=return_doc,
                            preprocessor=return_doc,
                            token_pattern=None)
    # ... X with string feature names (tfidf.get_features_names_out())

    # Select ML model with feature weights and predicted probabilities
    if weight_f == 'logits':
        model = LogisticRegression(random_state=1)
    elif weight_f == 'sgd':
        # Should be same as Logits but potential partial_fit
        model = SGDClassifier(random_state=1, loss='log_loss')
    elif weight_f == 'multiNB':
        model = MultinomialNB()
    else:
        raise Exception('Model not in list of valid models: {}'.format(WEIGHT_BY_MODEL))

    # Create pipeline and fit
    # ... consider the option of saving the gridsearch and .best_params
    pipeline = Pipeline(steps=[('vectorizer', tfidf), ('model', model)])
    pipeline.fit(X, y)  # used for predict, and coef_ with vectorizer.feature_names.

    return pipeline


def import_ML_coefficients(o_path):
    return joblib.load(f'{o_path}.joblib')


def get_feature_names_and_weights(pipeline, weight_f):
    feature_names = pipeline['vectorizer'].get_feature_names_out().tolist()
    if weight_f == 'logits' or weight_f == 'sgd':
        # coef_ is of shape (1, n_features) when the given problem is binary
        model_coef = pipeline['model'].coef_[0].tolist()
    elif weight_f == 'multiNB':
        # Empirical log probability of features given a class: log-probabilities of class 0 (always negative)
        # Higher probabilities correspond to "less negative" numbers: less important would have the lowest value
        model_coef = pipeline['model'].feature_log_prob_[0].tolist()
    else:
        raise Exception(f'{weight_f} invalid or missing from not in list of model methods: {WEIGHT_BY_MODEL}')
    return feature_names, model_coef

