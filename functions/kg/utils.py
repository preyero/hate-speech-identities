""" owlready2 functions to use the semantic resource """
import pandas as pd
from owlready2 import *
import owlready2.entity

INFER_METHODS = ['none', 'hierarchical']


def load_owl(path_to_owl_file: str):
    """ load a local copy using owlready2 """
    return get_ontology(path_to_owl_file).load()


def get_kg_dict(kg, lang='en'):
    """ Get dict of {k.iri: [label, synonyms]}"""
    # Create dict from KG classess and inviduals
    kg_cls_dict = {k.iri: k.label + k.alternateName + k.short_name + k.hasSynonym + k.hasExactSynonym +
                   k.hasBroadSynonym + k.hasNarrowSynonym + k.hasRelatedSynonym + k.replaces + k.isReplacedBy
                   for k in kg.classes()}
    kg_ind_dict = {k.iri: k.label + k.alternateName + k.short_name + k.hasSynonym + k.hasExactSynonym +
                   k.hasBroadSynonym + k.hasNarrowSynonym + k.hasRelatedSynonym + k.replaces + k.isReplacedBy
                   for k in kg.individuals()}
    kg_dict = {**kg_cls_dict, **kg_ind_dict}

    # Filter only synonyms in English
    def filter_by_lang(syns, lang):
        return [syn for syn in syns if type(syn) == owlready2.util.locstr and syn.lang == lang]

    kg_dict_en = {}
    for c_iri, syns in kg_dict.items():
        syns_en = filter_by_lang(syns, lang)
        kg_dict_en[c_iri] = syns_en if len(syns_en) != 0 else syns

    return kg_dict_en


def get_entity_matches(df: pd.DataFrame, inv_index, text_col: str, id_col: str, kg_dict: dict, match_method: str):
    """ Match entities to text field using label, synonym: [c1.iri, c2.iri] """
    # 1. term/ontology matching: full-text query to search engine (for each synonym):
    # {ent1: [ID1, ID6], ent2: [ID90], ...}
    ids_dict = {}
    for c_iri, synonyms in kg_dict.items():
        # ... query with synonyms of each concept:
        q_list = [si for si in synonyms if isinstance(si, owlready2.util.locstr)]

        ids_dict[c_iri] = inv_index.query(q_list, text_col, id_col, match_method)

    # 2. matrix transform: get entities on each text
    # {ID1: [ent1, ent99], ID2: [ent4], ...}
    iris_dict = {}
    for k, v in ids_dict.items():
        for x in v:
            iris_dict.setdefault(x, []).append(k)

    # return ent_assert column
    return df[id_col].apply(
        lambda id: iris_dict[id] if id in iris_dict.keys() else [])


def get_hierarchical_info(c_iri: owlready2.entity, kg: owlready2.namespace.Ontology):
    """
    Infer new information from KG structure for an entity:
    from a class, all superclasses
    from an individual, its types
    return: [c11.iri, c12.iri,...]
    """
    ent_infer = []
    # If c is class, get all superclasses
    c = kg.search_one(iri=c_iri)
    if isinstance(c, owlready2.entity.ThingClass):
        r = list(default_world.sparql("""
                SELECT ?y
                {   ?? rdfs:subClassOf* ?y
                }
                """, [c]))
    # If c is individual, get all types
    elif isinstance(c, owlready2.entity.Thing):
        r = list(default_world.sparql("""
                SELECT ?y
                { ?? rdf:type ?y
                }
                   """, [c]))
        # ... types may be classes (i.e., ThingClass)
        r = [vi for vi in r if isinstance(vi[0], owlready2.entity.ThingClass)]
    else:
        raise Exception(f"Invalid concept type retrieved from IRI {c_iri}. "
                        "Permitted types: ThingClass, Thing")
    ent_infer += [k[0].iri for k in r if k[0].label]
    return ent_infer


