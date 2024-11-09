import streamlit as st
from annotated_text import annotated_text
import pickle
import pandas as pd
from pickle import dump
import joblib
import os
#------
from streamlit_navigation_bar import st_navbar
from pathlib import Path

st.set_page_config(initial_sidebar_state="collapsed", page_title="Singapore HDB Re-Sale Price Predictor App", layout = "wide")

pages = ["Home", "Predict Resale Price", "About the Developer"]

styles = {
    "nav": {
        "background-color": "rgb(139, 0, 0)",
    },
    "div": {
        "max-width": "30rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(255, 255, 255)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

page = st_navbar(pages, options={"use_padding": False}, styles=styles)



#----------------------------------------------HOME PAGE-----------------------------------------------------------------

if page == "Home":
    #st.header("Singapore HDB Re-Sale Price Predictor App")
    st.subheader("Singapore HDB Re-Sale Price Predictor App", divider="gray")
    
    left_co,last_co = st.columns(2)
    with left_co:
        st.image("https://www.hdb.gov.sg/-/media/HDBContent/Images/CCG/current-corporate-signature-.ashx",width = 300)

    st.write("The Housing and Development Board (HDB) of Singapore plays a crucial role in the nation's public housing landscape. Established on February 1, 1960, as a statutory board under the Ministry of National Development, HDB is responsible for planning and developing public housing, managing estates, and ensuring that housing remains affordable for Singaporeans")
    st.write("On overwhemling pricing of HDB flat, this app is developed to predict resale price from the HDB data (https://data.gov.sg/collections/189/view) itself. This is a simple app which use regression algorithm to train the model, and have been deployed in cloud based for end-user consumption")

    
#----------------------------------------------Top10s PAGE-----------------------------------------------------------------


if page == "Predict Resale Price":

    st.subheader("Singapore HDB Re-Sale Price Predictor App", divider="gray")
    st.image("https://www.hdb.gov.sg/-/media/HDBContent/Images/CCG/current-corporate-signature-.ashx",width = 300)


    st.markdown("##### :blue[Please enter the below queries to predict re-sale price]")
    #C:\Users\ashfaq.ahamed\Documents\projects1\ICM\le_col_val_rg.pkl

    le_col_val_rg_path = Path(__file__).parent / 'data/le_col_val_rg.pkl'
    with open(le_col_val_rg_path, 'rb') as f:
        loaded_le_col_val_rg = pickle.load(f)

    oe_col_val_rg_path = Path(__file__).parent / 'data/oe_col_val_rg.pkl'
    with open(oe_col_val_rg_path, 'rb') as f:
        loaded_oe_col_val_rg = pickle.load(f)

    dt_rg_path = Path(__file__).parent / 'data/dt_rg.pkl'
    with open(dt_rg_path, 'rb') as f:
        loaded_dt_rg = pickle.load(f)

    scaler_rg_path = Path(__file__).parent / 'data/scaler_rg.pkl'
    with open(scaler_rg_path, 'rb') as f:
        loaded_scaler_rg = pickle.load(f)

    le_rg_path = Path(__file__).parent / 'data/le_rg.pkl'
    with open(le_rg_path, 'rb') as f:
        loaded_le_rg = pickle.load(f)

    oe_rg_path = Path(__file__).parent / 'data/oe_rg.pkl'
    with open(oe_rg_path, 'rb') as f:
        loaded_oe_rg = pickle.load(f)
    #loaded_le_col_val = joblib.load('le_col_val.pkl')
    #loaded_oe_col_val = joblib.load('oe_col_val.pkl')
    #path1 = os.access("le_col_val.pkl", os.F_OK)
    #file = open('le_col_val.pkl', 'rb')
    #loaded_le_col_val = pickle.load(file)
    #file.close()

    col_df_rg = ['town','flat_type','block',	'street_name', 'storey_range','floor_area_sqm','flat_model','lease_commence_date','year']
    le_col_rg = ['town' , 'block', 'street_name']
    oe_col_rg = ['flat_type', 'storey_range', 'flat_model','lease_commence_date','year']

    left_co, cent_co,last_co = st.columns(3)
    with left_co:

        inp_Town = st.selectbox(
            'Select Town',
            (loaded_le_col_val_rg['town'].keys()))

        inp_street_name = st.selectbox(
            'Select Street Name',
            (loaded_le_col_val_rg['street_name'].keys()))
        
        inp_block = st.selectbox(
            'Select Block',
            (loaded_le_col_val_rg['block'].keys()))

    with cent_co:
        

        inp_flat_type = st.selectbox(
            'Select Flat Type',
            (loaded_oe_col_val_rg['flat_type'].keys()))

        inp_storey_range = st.selectbox(
            'Select Stoery Range',
            (loaded_oe_col_val_rg['storey_range'].keys()))
        
        inp_flat_model = st.selectbox(
            'Flat Model',
            (loaded_oe_col_val_rg['flat_model'].keys()))

        

    with last_co:
        
        
        inp_lease_commence_date = st.selectbox(
            'Select Lease Commence Year',
            (loaded_oe_col_val_rg['lease_commence_date'].keys()))
        
        inp_year = st.selectbox(
            'Select Search year (2024) or any',
            (loaded_oe_col_val_rg['year'].keys()))


        inp_floor_area_sqm = st.number_input("Enter Floor area (Sq.m) (min 30)")

    ######### Form input dataframe

    if st.button("Predict"):
        if inp_floor_area_sqm >= 30 :

            input = [inp_Town, inp_flat_type, inp_block, inp_street_name,\
                        inp_storey_range, inp_floor_area_sqm, inp_flat_model,\
                            inp_lease_commence_date, inp_year]
            input_data = pd.DataFrame(columns = col_df_rg)
            input_data.loc[0] = input



            for col in le_col_rg:
                input_data[col] = loaded_le_rg[col].transform([input_data[col]])

            for col in oe_col_rg:
                input_data[col] = loaded_oe_rg[col].transform([input_data[col]])

            input_data[input_data.columns] = loaded_scaler_rg.transform(input_data[input_data.columns])


            predicted_val = loaded_dt_rg.predict(input_data)

            str_val = str(int(predicted_val[0]))

            sentence = (f"""
                            <span style="font-family:Arial;font-size: 36px;">The predicted resale price is 
                            <span style="background-color: yellow; font-size: 36px; font-weight: bold;">Singapore Dollars S$ {str_val}</span> 
                        """
            )

                        # Display the sentence in Streamlit
            st.markdown(sentence, unsafe_allow_html=True)
        else:

            st.markdown(f' ### :red[The floor square area must be of minimum 30 sq.m to generate the values] ')
    #print(f' For the given input data, the predicted PO status would be {predicted_val[0]}')

#streamlit run c:/Users/ashfaq.ahamed/Documents/projects1/HDB/main.py
