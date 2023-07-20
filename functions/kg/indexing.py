""" indexing documents to an in-memory index and performing a search using a list of queries"""

from typing import Dict, List, Sequence

import pandas as pd
pd.options.display.max_colwidth = 10000
from whoosh.fields import *
from whoosh.query import *
from whoosh import qparser, query
from whoosh.qparser import QueryParser
from whoosh.filedb.filestore import RamStorage
from whoosh.analysis import StemmingAnalyzer

from functions.kg.utils import load_owl

D_IMPORT_PATH = './data/jigsaw.csv'
TEXT_COL = 'comment_text'
ID_COL = None
KG_PATH = './models/adaptation/gsso.owl'

MATCH_METHODS = ['exact', 'stem', 'variations']


class EntityMatching:
    def __init__(self, schema):
        self.schema = schema
        # The FileStorage was not saving the docs
        self.ix = RamStorage().create_index(self.schema)

    def index_documents(self, docs: Sequence):
        writer = self.ix.writer()
        for doc in docs:
            d = {k: v for k,v in doc.items() if k in self.schema.stored_names()}
            writer.add_document(**d)
        writer.commit(optimize=True)

    def get_index_size(self) -> int:
        return self.ix.doc_count_all()

    def query(self, q_list: List, field: str, id_col: str, analyzer: str) -> List:
        # Create query of list of synonyms
        # e.g.,  "(syn1) OR (syn2)"
        c_query_str = " OR ".join([f'("{subq}")' for subq in q_list])

        # Return list of doc ids
        with self.ix.searcher() as searcher:
            plugins = [qparser.PhrasePlugin, qparser.GroupPlugin, qparser.OperatorsPlugin]
            # DEFAULT: [<whoosh.qparser.plugins.WhitespacePlugin at 0x104ce58b0>,
            #  <whoosh.qparser.plugins.WhitespacePlugin at 0x104ce53d0>,
            #  <whoosh.qparser.plugins.SingleQuotePlugin at 0x104ce5370>,
            #  <whoosh.qparser.plugins.FieldsPlugin at 0x104ce57c0>,
            #  <whoosh.qparser.plugins.WildcardPlugin at 0x104ce5ee0>,
            #  <whoosh.qparser.plugins.PhrasePlugin at 0x104ce56a0>,
            #  <whoosh.qparser.plugins.RangePlugin at 0x104ce5af0>,
            #  <whoosh.qparser.plugins.GroupPlugin at 0x104ce52b0>,
            #  <whoosh.qparser.plugins.OperatorsPlugin at 0x104ce5fa0>,
            #  <whoosh.qparser.plugins.BoostPlugin at 0x104ce58e0>,
            #  <whoosh.qparser.plugins.EveryPlugin at 0x104ce5e20>]
            parser = QueryParser(field, self.schema, plugins=plugins)
            if analyzer == 'variations':
                parser.termclass = query.Variations
            results = searcher.search(parser.parse(c_query_str), limit=None)
            # return doc ids
            search_results = [r[id_col] for r in results]
        return search_results


def indexing_df(df: pd.DataFrame, text_col: str, id_col: str, analyzer: str):
    """ Create an inverted index from df """
    df[id_col] = df[id_col].apply(lambda id: str(id))
    docs = df[[id_col, text_col]].to_dict('records')

    schema = Schema()
    # Analyzer: wraps a tokenizer and zero or more filters
    # e.g. analyzer = Tokenizer () | OptFilter1() | OptFilter2()
    if analyzer == 'exact' or analyzer == 'variations':
        # Exact -> Standard analyzer
        # Variations -> Standard analyzer and expand at query time
        schema.add(id_col, ID(stored=True))
        # STOPWORDS = frozenset({'to', 'the', 'for', 'from', 'are', 'your', 'you', 'an', 'as', 'on', 'and', 'is',
        # 'will', 'tbd', 'if', 'it', 'or', 'yet', 'may', 'when', 'can', 'with', 'have', 'by', 'a', 'at', 'we', 'this',
        # 'that', 'not', 'us', 'in', 'be', 'of'})
        # StandardAnalyzer() = CompositeAnalyzer(
        #                          RegexTokenizer(expression=re.compile('\\w+(\\.?\\w+)*'), gaps=False),
        #                          LowercaseFilter(),
        #                          StopFilter(stops=STOPWORDS, min=2, max=None, renumber=True)
        #                       )
        schema.add(text_col, TEXT(stored=True))
    elif analyzer == 'stem':
        # Stemming: different forms and accents - enhancing recall
        schema.add(id_col, ID(stored=True))
        # stemming filter applies the stemming function to the terms it indexes, and to words in user queries
        # Combines: tokenizer, lower-case filter, optional stop filter, and stem filter (porter)
        schema.add(text_col, TEXT(stored=True, analyzer=StemmingAnalyzer()))
        # Example of stemming: kg_dict['http://purl.obolibrary.org/obo/GSSO_001968']
        # from whoosh.lang.porter import stem
        # stream = ['marital partner', 'marital partners', 'married partner', 'spouse', 'husband', 'wife']
        # [print(stem(syn)) for syn in stream]
    else:
        raise Exception(f'{analyzer} Invalid method for processing text to create index for entity matching.'
                        f'Method selected in the list:  ' + ', '.join(MATCH_METHODS))

    inv_index = EntityMatching(schema)
    inv_index.index_documents(docs)
    print(f"indexed {inv_index.get_index_size()} documents")
    return inv_index


def main():
    # 0. Input arguments
    text_col, id_col = TEXT_COL, ID_COL

    d0 = pd.read_csv(D_IMPORT_PATH)
    df = d0.sample(10, random_state=1).reset_index(drop=True)
    example_texts = {0: "I told him, this is messed up.",
                     1: "Flag it.",
                     2: "He is the best person I ever met, I love his company.",
                     3: "My best friends,the best gay couple!",
                     4: "My main friends are the happiest gay couple!",
                     5: "My best friends are the happiest gey couple!",
                     6: "This is the best couple!"}
    for idx, text in example_texts.items():
        df.loc[idx, text_col] = example_texts[idx]

    kg_path = KG_PATH
    kg = load_owl(kg_path)
    kg_cls_dict = {k.iri: k.label + k.alternateName + k.short_name + k.hasSynonym + k.hasExactSynonym +
                      k.hasBroadSynonym + k.hasNarrowSynonym + k.hasRelatedSynonym + k.replaces + k.isReplacedBy
                   for k in kg.classes()}

    entity_sample = list(kg_cls_dict.keys())[:4]
    kg_dict = {c_iri: kg_cls_dict[c_iri] for c_iri in entity_sample}
    # example: D_IMPORT_PATH='./data/jigsaw.csv'
    # try single occurrence: v1 (only 3 has gay - 5 is misspelled)
    kg_dict['example1'] = ['gay']
    # try multiple synonyms: v1 (0 has him, 7 has he, 2 has he and his)
    kg_dict['example2'] = ['him', 'he', 'his']
    # try spelling: v1 (1 has Flag)
    kg_dict['example3'] = ['flag']
    # try compount nouns: v1 (3 has gay couple, 5 has gey couple singular)
    kg_dict['example4'] = ['gay couple', 'gey couples']
    # try variations (e.g. plural)
    kg_dict['example5'] = ['gays']

    if not id_col:
        id_col = 'id'
        df['id'] = range(0, df.shape[0])

    # look examples using kg
    kg_ind_dict = {k: k.label + k.alternateName + k.short_name + k.hasSynonym + k.hasExactSynonym +
                      k.hasBroadSynonym + k.hasNarrowSynonym + k.hasRelatedSynonym + k.replaces + k.isReplacedBy
                   for k in kg.individuals()}
    entity_example = kg.search_one(iri="/add/iri/from/cmd+U/protege")
    kg_dict = {**kg_cls_dict, **kg_ind_dict}
    kg_dict_onto = {}
    c_examples = ["http://purl.obolibrary.org/obo/GSSO_000369", # woman cls(were missing bc of query limit)
                  "http://purl.obolibrary.org/obo/GSSO_000370", # men cls
                  "http://purl.obolibrary.org/obo/GSSO_001180", # womyn ind
                  "http://purl.obolibrary.org/obo/GSSO_001591", # gay cls
                  "http://purl.obolibrary.org/obo/GSSO_001968", # marital partner cls
                  "http://purl.obolibrary.org/obo/GSSO_000374", # b√∏sse gay man cls(over-matched in v2 errors/weird label assigned)- not in v1
                  "http://purl.obolibrary.org/obo/GSSO_000124", # assigned male at birth cls(over-matched in v2 errors/weird label assigned)- not in v1
                  "http://purl.obolibrary.org/obo/GSSO_004152", # fuck ind(over-matched in v2 errors)- not in v1
                  "http://purl.obolibrary.org/obo/GSSO_010081", # A-Gay ind(over-matched in v2 errors)- not in v1 *
                  "http://purl.obolibrary.org/obo/GSSO_001306"] # submission cls(missing from v2 errors)- not in v2
    # Adjusting q params
    # label[0] order may change in any version (e.g., consent vs gay man or assigned male at birth)
    # ?* wildcards: fuck/assigned at birth bc there were asterisk in the synonyms, plugin disabled in queries
    # phrases: gay man bc smaller compound syns ("entity synonym") OR ("...")
    # single quotes stripped in RegexTokenizer (A is in gay): A-Gay bc of single '' is matching to all "gay" entries (' is not a special character)
    # submission was being match with s (to any text with ...'s)
    for c_iri in c_examples:
        c = kg.search_one(iri=c_iri)
        kg_dict_onto[c] = kg_dict[c]

    #  Create or load an inv index from the df (using text col and id col (default: None))
    inv_index = indexing_df(df, text_col, id_col, 'stem')

    # 1. term/ontology matching: full-text query to search engine
    # a. ... use str examples
    ids_dict = {}
    for c, synonyms in kg_dict.items():
        # ... query with synonyms of each concept:
        q_list = [si for si in synonyms if isinstance(si, str)]
        ids_dict[c] = inv_index.query(q_list, text_col, id_col, 'stem')
    # b. ... or use same implementation in onto functions
    ids_dict = {}
    import owlready2.entity
    q_lists = {}
    for c, synonyms in kg_dict_onto.items():
        q_list = []
        for si in synonyms:
            if isinstance(si, owlready2.entity.ThingClass) or isinstance(si, owlready2.entity.Thing):
                si = si.label[0]
            if isinstance(si, owlready2.util.locstr):
                q_list.append(si)
        ids_dict[c.iri] = inv_index.query(q_list, text_col, id_col)

        q_lists[c.iri] = q_list

    # 2. matrix transform: get entities on each text
    iris_dict = {}
    for k, v in ids_dict.items():
        for x in v:
            iris_dict.setdefault(x, []).append(k)

    # ... inspect results:
    # df.loc[df[id_col] == 'enter/ID', text_col]

    # return ent_assert column
    df['ent_assert'] = df[id_col].apply(
        lambda id: iris_dict[id] if id in iris_dict.keys() else [])
    # ... print df['ent_assert'] and check with examples the matching is right
    return


if __name__ == '__main__':

    main()
