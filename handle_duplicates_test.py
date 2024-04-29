import os
import pandas as pd
import pickle
import numpy as np
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
import base64
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import io


st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")



working_dir = os.path.dirname(os.path.abspath(__file__))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu('Menu',
                           ['Data Analysis'],
                           menu_icon='hospital-fill',
                           icons=['cloud-upload', 'heart','data'],
                           default_index=0)


#----------------------------------------------------------------------------------------------------------    

if selected == 'Upload CSV':
    
    st.title('Upload CSV')


if selected == 'Data Analysis':
    
    st.title('Clean Data')

    
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    

    if uploaded_files:
        
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            st.write(uploaded_file.name)        
            
            def info_dataset(df):
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

            # Hàm xử lý duplicate
            def handle_duplicates(my_df):
                st.write('DataFrame before handling duplicates:')
                st.write(my_df)
                info_dataset(my_df)
                
                # Drop duplicates
                my_df.drop_duplicates(inplace=True)
                
                return my_df   
                                   
            # Kiểm tra nếu session state chưa tồn tại, khởi tạo mới
            if 'my_df' not in st.session_state: 
                st.session_state.my_df = df
            if 'deleted_columns' not in st.session_state:
                st.session_state.deleted_columns = []    
            
            tab1, tab2 = st.tabs(["Handle duplicates", "asdh"])
        
            with tab1:
                st.header("Handle Duplicates")
                            
                if st.button("Check for Duplicates"):
                    st.session_state.my_df_copy = handle_duplicates(my_df = df)
                    st.write(st.session_state.my_df_copy)
                    info_dataset(st.session_state.my_df_copy)
                    
                          
                         
               
            
            
            
            
            
            
            

        
    


    



    
    

    