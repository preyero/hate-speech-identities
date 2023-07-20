""" Gab Hate Corpus data collection functions from root CSV file """

import pandas as pd

# Global variables
IDENTITIES_DICT = {'gender': ['GEN'], 'sexual_orientation': ['SXO'], 'religion': ['REL'],
                   'race': ['RAE'], 'disability': ['MPH'], 'origin': ['NAT'], 'politics': ['IDL_POL']}
TEXT_COL, ID_COL = 'Text', 'ID'


def process_data(d_path: str, o_path: str = None, text_col: str = TEXT_COL, id_col: str = ID_COL):
    """ Load dataset from downloaded data repo (GabHateCorpus_annotations.tsv). Export to o_path"""
    # 1. Import original tsv file
    gab = pd.read_csv(d_path, delimiter='\t')
    print(f'  imported from path: {d_path}. {gab.shape[0]} samples.')
    # ...  86529 samples

    target_cols = [col for values in IDENTITIES_DICT.values() for col in values]
    gab['IDL_POL'] = ((gab['IDL'] == 1) | (gab['POL'] == 1)).astype('int')
    gab = gab[~gab[target_cols].isna().any(axis=1)]
    print(f'{gab.shape[0]} annotations samples with target labels.')
    # ... 11249 annotations samples with target labels.

    # 2. Aggregate individual annotations
    print('  aggregating annotations:')
    targets = gab[[id_col] + target_cols]
    annotator_props = targets.fillna(0).groupby(id_col).mean()
    gab = annotator_props.merge(gab[[id_col, text_col]].drop_duplicates(id_col), how='left', on=id_col)
    print(f'{gab.shape[0]} aggregated samples with categorical target labels.')
    # ... 7813 aggregated samples with categorical target labels.

    # 3. Create group label integer (if contains any subgroup label) and subgroup list
    # -- group label maximum probability (max subgroup of each group)
    # -- list of subgroups with max probability
    for g, subg_l in IDENTITIES_DICT.items():
        gab.loc[:, f"{g}"] = gab.apply(lambda row: max([row[subg] for subg in subg_l]), axis=1)
        gab.loc[:, f"{g}_list"] = gab.apply(
            lambda row: [subg for subg in subg_l if (row[subg] == row[f'{g}']) & (row[subg] != 0)], axis=1)
        print(f'  created {g} column with max annotator percentage and list of subgroups with that value')

    # 3. Save processed file to data folder
    if o_path:
        gab.to_csv(o_path, index=False)
    print(f'  prepared data file saved to: {o_path}')
    return gab, text_col, id_col, IDENTITIES_DICT





