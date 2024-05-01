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

tab1, tab2 = st.tabs(["Check Outliers", "asdh"])



def check_outliers_plot(my_df, selected_column):
                # Tính giá trị Q1, Q3 và IQR
                Q1 = my_df[selected_column].quantile(0.25)
                Q3 = my_df[selected_column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Tìm giá trị ngoại lệ dưới và trên
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Tạo DataFrame chứa thông tin về outlier
                outliers = my_df[(my_df[selected_column] < lower_bound) | (my_df[selected_column] > upper_bound)]
                
                if outliers.empty:
                    st.write("No outliers found.")
                else:
                    # Vẽ biểu đồ box plot
                    fig = px.box(my_df, y=selected_column, title=f'Box plot of {selected_column}')
                    st.plotly_chart(fig)
                    
def remove_outliers(my_df, selected_column):
                # Tính giá trị Q1, Q3 và IQR
                Q1 = my_df[selected_column].quantile(0.25)
                Q3 = my_df[selected_column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Tìm giá trị ngoại lệ dưới và trên
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Loại bỏ các ngoại lệ khỏi dữ liệu
                my_df = my_df[(my_df[selected_column] >= lower_bound) & (my_df[selected_column] <= upper_bound)]
                
                return my_df
            
with tab1:
    st.header("Check Outliers")
    st.write("Choose column to check outliers")
    
    selected_column = st.selectbox("Column", st.session_state.my_df.columns, key="outlier_select")
    
    check_outliers_plot(st.session_state.my_df, selected_column)
        
    if st.button('Handle Outliers'):
        st.session_state.my_df = remove_outliers(df, selected_column)
        st.write(selected_column)
        st.write('remove succesfuly')
        
            