""" XTREMESPEECH 20k data collection functions from root folder: India and Kenya datasets """

import pandas as pd
import pandas.errors

COUNTRY = ['india', 'kenya_re']
SPLIT = ['train', 'dev', 'test']
IDENTITIES_DICT = {'gender': ['women'],
                   'sexual_orientation': ['sexual minorities'],
                   'religion': ['religious minorities'],
                   'race': ['racialized groups', 'large ethnic groups, ethnic minorities, indigenous groups'],
                   'economic_status': ['historically oppressed caste groups'],
                   'other': ['any other']}
TEXT_COL, ID_COL = 'Text', 'ID'


def process_data(d_path: str, o_path: str = None, text_col: str = TEXT_COL, id_col: str = ID_COL):
    """ Data with group labels integer and subgroup list provided by annotators"""
    # 1. Import original csv files: indian and kenya datasets
    d_list = []
    for c in COUNTRY:
        d_c_list = []
        for s in SPLIT:
            try:
                d = pd.read_csv(f"{d_path}/{c}_{s}.csv")
                # add split column
                d['split'] = [s] * d.shape[0]
                d_c_list.append(d)
            except pandas.errors.ParserError:
                print(f'  {d_path}/{c}_{s}.csv not included due to possible malformed input file.')
        # concatenate splits of country c
        if d_c_list:
            d_c = pd.concat(d_c_list, ignore_index=True)
            # add country (subdataset) identifier column
            d_c['country'] = [c] * d_c.shape[0]
            print('  imported {} samples from {} dataset.\n'
                  '... split counts: \n{}'.format(d_c.shape[0], c, d_c.split.value_counts()))
            #  ... imported 5180 samples from kenya_re dataset.
            # remove empty text fields
            d_c = d_c[d_c[text_col].str.strip().astype(bool)]
            print('  {} after removing empty text fields.\n'
                  '... split counts: \n{}'.format(d_c.shape[0], d_c.split.value_counts()))
            d_list.append(d_c)

    # concatenate countries of df
    d = pd.concat(d_list, ignore_index=True)
    print('  loaded data: {} samples.\n ... country counts: \n{}'.format(d.shape[0], d.country.value_counts()))
    # ...  loaded data: 5180 samples. ... country counts: kenya_re    5180
    # in paper 405+2695+2081=5181

    # 2. Create group label integer (if contains any subgroup label) and subgroup list
    for g, subg_l in IDENTITIES_DICT.items():
        d.loc[:, f"{g}_list"] = d['Target (protected)'].apply(
            lambda text: [subg for subg in subg_l if type(text) == str if subg in text])
        d.loc[:, f"{g}"] = d[f"{g}_list"].apply(lambda x: 0 if len(x) == 0 else 1)
        print(f'  created {g} categorical column (1 if text annotated with any of its subgroup) and list of subgroups. '
              f'\n... {g} counts: \n{d[g].value_counts()}'
              f'\n ... of languages: \n{d.loc[d[g]==1, "Language"].value_counts()}')

    # 3. Save processed file to data folder
    if o_path:
        d.to_csv(o_path, index=False)
    print(f'  prepared data file saved to: {o_path}')

    return d, text_col, id_col, IDENTITIES_DICT
