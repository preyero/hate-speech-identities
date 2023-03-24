""" Jigsaw Toxicity 448k data collection functions from root CSV file """

import pandas as pd

# Global variables
IDENTITIES_DICT = {'gender': ['male', 'female', 'transgender', 'other_gender'],
                   'sexual_orientation': ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
                                          'other_sexual_orientation'],
                   'religion': ['christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion'],
                   'race': ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity'],
                   'disability': ['physical_disability', 'intellectual_or_learning_disability',
                                  'psychiatric_or_mental_illness', 'other_disability']}
TEXT_COL, ID_COL = 'comment_text', 'id'


def process_data(d_path: str, o_path: str = None, text_col: str = TEXT_COL, id_col: str = ID_COL):
    """ Data with group label maximum probability and corresponding subgroup list provided by annotators """
    # import original csv file
    d = pd.read_csv(d_path)
    print(f'  imported from path: {d_path}. {d.shape[0]} samples.')

    # only with group labels: 448k
    d = d.loc[d.identity_annotator_count != 0, ]
    print(f'  excluded data without entity annotations. {d.shape[0]} samples.')

    # group label maximum probability (max subgroup of each group),
    # and list of subgroups with max probability, and
    for g, subg_l in IDENTITIES_DICT.items():
        d.loc[:, f"{g}"] = d.apply(lambda row: max([row[subg] for subg in subg_l]), axis=1)
        d.loc[:, f"{g}_list"] = d.apply(
            lambda row: [subg for subg in subg_l if (row[subg] == row[f'{g}']) & (row[subg] != 0)], axis=1)
        print(f'  created {g} column with max annotator percentage and list of subgroups with that value')

    # save processed file to data folder
    if o_path:
        d.to_csv(o_path, index=False)
    print(f'  prepared data file saved to: {o_path}')
    return d, text_col, id_col, IDENTITIES_DICT

