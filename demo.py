import os, glob
import streamlit as st
import pandas as pd

import identity_group_identification as model_f

PROJ_DIR = os.getcwd()
M_DIR = os.path.join(PROJ_DIR, 'models')
PREDICT = 'target_gso'


def main():
    """Knowledge-grounded target group detection models"""
    st.title("A Hybrid (KG + DL) Model")
    st.subheader("Knowledge-grounded target group detection in hate speech")
    st.write(""" This demo runs inference on a text with the models presented at [Semantics 2023 Conf](https://ebooks.iospress.nl/volumearticle/64009)!""")


    # List available models
    m_h = glob.glob(os.path.join(M_DIR, 'hybrid/gso*/*'))
    m_lm = glob.glob(os.path.join(M_DIR, 'llm/gso*/*'))

    m_paths = m_h + m_lm 
    m_names = [m.split('/')[-1] for m in m_paths]

    # Select the model
    m_tags = {'gsso_jigsaw_gendersexualorientation_0.5-stem-hierarchical-docf':r'HybridDocF$_{h}$', 
              'gsso_jigsaw_gendersexualorientation_0.5-stem-hierarchical-logits':r'HybridLR$_{h}$', 
              'gsso_jigsaw_gendersexualorientation_0.5-stem-hierarchical-multiNB': r'HybridMultiNB$_{h}$', 
              'gsso_jigsaw_gendersexualorientation_0.5-stem-none-docf':r'HybridDocF',
              'gsso_jigsaw_gendersexualorientation_0.5-stem-none-logits': r'HybridLR', 
              'gsso_jigsaw_gendersexualorientation_0.5-stem-none-multiNB': r'HybridMultiNB', 
              'roberta-base':r'RoBERTa$_{base}$'}
    
    #m_tag = st.sidebar.selectbox("Select model", [m_tags[m] for m in m_names if m in m_tags.keys()])
    m_tag = st.selectbox("Select model", [m_tags[m] for m in m_names if m in m_tags.keys()])
    st.info(f"Prediction with {m_tag}")

    # Load model
    m_name = list(m_tags.keys())[list(m_tags.values()).index(m_tag)]
    m_path = glob.glob(os.path.join(M_DIR, f'*/gso*/{m_name}'))[0]
    # st.write(f"""{m_path}""")
    m = model_f.model_load(m_path)

    # Enter text 
    text = st.text_area("Enter text", "Type here")
    text_c, id_c, pred_c = 'Text', 'ID', PREDICT
    data = pd.DataFrame(data={id_c: [0], text_c: [text], pred_c: ['no-label']})
    # st.write(data)

    # Get predictions
    if text != "Type here":
        _ , y_pred, interp = model_f.model_predict(m, data, PREDICT, text_c, id_c)

        # Print results
        st.subheader('Predicted probability of referring to gender/sexuality')
        data['Prediction'] = y_pred
        st.write(data[[text_c, 'Prediction']])

        st.subheader('model interpretations')
        if interp != None:
            labels, iris = interp[0][0].split(';')+interp[1][0].split(';'), interp[4][0].split(';')+interp[5][0].split(';')
            links = ', '.join([f"[{l}]({i})" for l,i in zip(labels, iris)])
            interp = {'Entities': interp[0], 'Other': interp[1], 'Most relevant': interp[2], 'Definition': interp[3]}
        st.write(interp)
        if interp != None:
            st.write(f""" more information about the entities in your text: \n {links}""")


if __name__ == '__main__':
    main()