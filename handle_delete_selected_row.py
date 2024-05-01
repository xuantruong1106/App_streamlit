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
           
            column_names = df.columns.tolist()
            st.write("Column names:", column_names) 
                      
# Kiểm tra nếu session state chưa tồn tại, khởi tạo mới
if 'my_df' not in st.session_state: 
    st.session_state.my_df = df
if 'deleted_columns' not in st.session_state:
    st.session_state.deleted_columns = []    

tab1, tab2 = st.tabs(["Remove Rows with Null", "asdh"])

with tab1:

    col1 , col2 = st.columns(2)
    
    with col1:
        st.write("Kiểm tra missing values")
        st.write(st.session_state.my_df.isnull().sum())

    with col2:
        
        selected_columns = st.multiselect("Select columns to remove rows with null values:", st.session_state.my_df.columns, key="RemoveRowsNull")
        
        # chứa các hàng đã chọn để xoá
        indices_to_remove = []
        
        # lấy các hàng null
        mask = st.session_state.my_df[selected_columns].isnull().any(axis = 1)
        
        # lấy thông tin các hàng null
        rows_with_null = st.session_state.my_df[mask]  
 
        for index, row in rows_with_null.iterrows():
            checkbox_value = st.checkbox(f'select row {index}')
            st.write(row.to_frame().T)
            if checkbox_value:
                indices_to_remove.append(index)
            
        if st.button("Remove Rows with Null"):
            if indices_to_remove:
                st.session_state.my_df = st.session_state.my_df.drop(indices_to_remove)
                st.write("remove successfully.")
            else:
                st.write('select rows you want to delete')
            
            