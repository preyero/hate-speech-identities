""" Gab Hate Corpus data collection functions from root CSV file """
import json
import pandas as pd
from collections import Counter, defaultdict

# Global variables
IDENTITIES_DICT = {'gender': ['Women', 'Men'], 'sexual_orientation': ['Heterosexual', 'Homosexual'],
                   'religion': ['Buddhism', 'Christian', 'Hindu', 'Islam', 'Jewish'],
                   'race': ['African', 'Arab', 'Asian', 'Caucasian', 'Hispanic', 'Indian'],
                   'origin': ['Indigenous', 'Refugee'],
                   'disability': ['Disability'],
                   'economic_status': ['Economic'],
                   'miscellaneous': ['Other']}
TEXT_COL, ID_COL = 'Text', 'post_id'


def generate_target_info(target1, target2, target3):
    """ This function counts target list of the three annotation and created a dictionary with only the ones with at least 2 votes"""
    final_target = []
    # Extract all communities for this post
    community_dict = Counter(target1 + target2 + target3)
    # Select only communities present more than once
    for key in community_dict:
        if community_dict[key] > 1:
            final_target.append(key)
    # If no community is selected based on majority voting (at least 2), we assign none
    if len(final_target)==0:
        final_target.append('None')
    return final_target


def process_data(d_path: str, o_path: str = None, text_col: str = TEXT_COL, id_col: str = ID_COL,
                 class_names: str = 'Data/classes_two.npy'):
    """ Load dataset from downloaded data repo (dataset.json). Export to o_path"""
    # 1. Import original json file
    with open(d_path, 'r') as fp:
        data = json.load(fp)
    dict_data = []
    for key in data:
        temp = {}
        temp[id_col] = key
        temp['text'] = data[key]['post_tokens']
        final_label = []
        for i in range(1, 4):
            temp['annotatorid' + str(i)] = data[key]['annotators'][i - 1]['annotator_id']
            #             temp['explain'+str(i)]=data[key]['annotators'][i-1]['rationales']
            temp['target' + str(i)] = data[key]['annotators'][i - 1]['target']
            temp['label' + str(i)] = data[key]['annotators'][i - 1]['label']
            final_label.append(temp['label' + str(i)])
        final_label_id = max(final_label, key=final_label.count)
        temp['rationales'] = data[key]['rationales']
        if ( class_names == 'Data/classes_two.npy'):
            to_discard = 'non-toxic'
            if (final_label.count(final_label_id) == 1):
                temp['final_label'] = 'undecided'
            else:
                if (final_label_id in ['hatespeech', 'offensive']):
                    final_label_id = 'toxic'
                else:
                    final_label_id = 'non-toxic'
                temp['final_label'] = final_label_id
        else:
            to_discard = 'normal'
            if (final_label.count(final_label_id) == 1):
                temp['final_label'] = 'undecided'
            else:
                temp['final_label'] = final_label_id

        dict_data.append(temp)
    temp_read = pd.DataFrame(dict_data)
    print(f' imported {temp_read.shape[0]} original samples')

    # 2. Data selection: only community targeted in hateful samples > at least two annotators agree
    d = temp_read.loc[temp_read.final_label != to_discard]
    print(f' {d.shape[0]} hateful samples with identity annotations')

    # 3. Data aggregation (Mathew et al 2020): group label if at least two agreed (otherwise, none).
    d['final_target'] = d.apply(lambda row: generate_target_info(row['target1'], row['target2'], row['target3']), axis=1)
    for g, subg_l in IDENTITIES_DICT.items():
        d.loc[:, f"{g}_list"] = d['final_target'].apply(lambda final_target: [subg for subg in subg_l
                                                                              if subg in final_target])
        d.loc[:, f"{g}"] = d[f"{g}_list"].apply(lambda x: 0 if len(x) == 0 else 1)
        print(f'  created {g} categorical column (1 if text annotated with any of its subgroup) and list of subgroups. '
              f'\n... {g} counts: \n{d[g].value_counts()}')

    # 3. Data preparation: concatenate tokens for the text
    d.insert(loc=2, column=text_col, value=d['text'].apply(lambda text: " ".join(text)))

    # 4. Saved processed file
    if o_path:
        d.to_csv(o_path, index=False)

    return d, text_col, id_col, IDENTITIES_DICT


