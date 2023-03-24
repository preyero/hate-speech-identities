""" Collect and save in data folder datasets with group labels of human annotations """

import os
import argparse
import pandas as pd
from typing import Dict

# Global vars
DNAMES = ['jigsaw', 'xtremespeech', 'gabhatecorpus', 'hatexplain']
# Names to match output in Multi-output TargetIdentityLayer.
OUTPUT_GROUP_NAMES = {'age': 'target_age', 'disability': 'target_disability', 'gender': 'target_gender',
                      'origin': 'target_origin', 'race': 'target_race', 'religion': 'target_religion',
                      'sexual_orientation': 'target_sexuality'}


def import_dataset(dname: str, d_path: str = None):
    """ Import processed data from 'data' folder or collect and save it from d_path"""
    o_path = f'./data/{dname}.csv'
    if dname == "jigsaw":
        from functions.dataCollect import jigsaw as data_processor
    elif dname == 'xtremespeech':
        from functions.dataCollect import xtremespeech as data_processor
    elif dname == 'gabhatecorpus':
        from functions.dataCollect import gabhatecorpus as data_processor
    elif dname == 'hatexplain':
        from functions.dataCollect import hatexplain as data_processor
    else:
        raise ValueError(f"Need to include data collection functions and path to original data file: {dname}. "
                         f"List of available datasets: " + " , ".join(DNAMES))

    if os.path.exists(o_path):
        # import processed file from data folder
        df = pd.read_csv(o_path)
        text_col, id_col = data_processor.TEXT_COL, data_processor.ID_COL
        identities_dict = data_processor.IDENTITIES_DICT
        print(f'{dname} imported successfully from data folder: {df.shape[0]} annotations samples.')
    else:
        # import file from source folder
        try:
            df, text_col, id_col, identities_dict = data_processor.process_data(d_path, o_path=o_path)
            print(f'{dname} fetched, prepared, and saved to data folder')
        except FileNotFoundError:
            raise FileNotFoundError("{} dataset not found in data folder. "
                                    "Provide valid path to original data file: {}".format(dname, d_path))
    return df, text_col, id_col, identities_dict


# Evaluating identity group identification models
def prepare_for_model_evaluation(df: pd.DataFrame, text_col0: str, id_col0: str, identities_dict: Dict, output_names=None):
    """ prepare for evaluating identity group identification model as an external dataset"""
    # 1. Rename to match model output convention
    if output_names is None:
        output_names = OUTPUT_GROUP_NAMES
    text_col, id_col = 'predict_text', 'comment_id'
    to_rename = {id_col0: id_col, text_col0: text_col}
    for group in identities_dict.keys():
        if group in output_names.keys():
            to_rename[group] = output_names[group]
    df = df.rename(to_rename, axis=1)

    # 2. Apply 0.5 hard thresholding to target annotations (y true)
    target_cols = sorted([col for col in df.columns if (col in output_names.values())])
    for target_col in target_cols:
        df[target_col] = df[target_col].apply(lambda target_col: int(target_col >= 0.5))

    # 3. Add the OR(Gender, Sexuality) column
    df['target_gso'] = ((df['target_gender'] == 1) | (df['target_sexuality'] == 1)).astype('int')
    target_cols.append('target_gso')
    return df, sorted(target_cols), text_col, id_col


def main():
    desc = "Collect and save dataset with identity labels in data folder"
    parser = argparse.ArgumentParser(description=desc)

    # Required parameters
    parser.add_argument("--d_name",
                        default=None,
                        type=str,
                        required=True,
                        help="Dataset selected in the list: " + ", ".join(DNAMES),
                        )

    # Other parameters
    parser.add_argument("--d_path",
                        default=None,
                        type=str,
                        required=False,
                        help="Path to original data file.",
                        )

    args = parser.parse_args()

    _ = import_dataset(args.d_name, args.d_path)
    return


if __name__ == "__main__":
    main()
