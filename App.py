import streamlit as st
import streamlit_authenticator as stauth
import yaml
import joblib
import pandas as pd
import numpy as np
import os
from yaml.loader import SafeLoader
from streamlit_option_menu import option_menu
from PIL import Image
import json

 #visualisation             
import matplotlib.pyplot as plt 
import seaborn as sns  

st.set_page_config(
    page_title="Aplikasi Prediksi Status Pengeboran",
    page_icon="🎯",
)

with open('auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
)

# Inisialisasi state
if 'rerun_called' not in st.session_state:
    st.session_state.rerun_called = False

if 'data_frame' not in st.session_state:
    st.session_state['data_frame'] = None

# Fungsi untuk memanggil st.experimental_rerun() sekali saja
def run_once():
    if not st.session_state.rerun_called:
        st.session_state.rerun_called = True
        st.experimental_rerun()

# Fungsi untuk membaca file Excel dengan caching
@st.cache_data
def load_excel_file(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df['date'] = pd.to_datetime(df['Date-Time'])
        df.set_index(['Date-Time'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        return None


def beranda():
    st.title('Selamat Datang!!')
    st.write('di Aplikasi Sistem Prediksi Status Pengeboran PT. Parama Data Unit')
    st.image("drilling.png", width=200)


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def beranda_logged_in():
    with st.sidebar:
        selected=option_menu(
            menu_title='Menu',
            options=['Prediksi','Visualisasi','Sumber data'],
            default_index=0,
        )
    if selected == "Prediksi":
        st.write('Silahkan untuk melakukan prediksi status pengeboran')
        model_path = os.path.abspath('hgb_model.pkl')
        loaded_model_hgb = joblib.load(model_path)

        uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx", "xls"])
        
        # Jika ada file yang di-upload
        if uploaded_file is not None:
                df = load_excel_file(uploaded_file)
                st.session_state['data_frame'] = df

        if st.session_state['data_frame'] is not None:
                # Meenyeleksi kolom
                selected_data = st.session_state['data_frame'][['BitDepth','Scfm','BVDepth','MudTempIn','MudTempOut','Stuck']]

                # Data Preprocessing
                selected_data_np = np.array(selected_data)
                selected_data_float = selected_data_np.astype(float)
                
                # PH = 5 menit
                n_steps_in = 60
                X_Test = selected_data_float.reshape((1, selected_data_float.shape[0]*selected_data_float.shape[1]))
                print(X_Test.shape)

                # Menggunakan model yang dimuat untuk membuat prediksi
                predictions_hgb = loaded_model_hgb.predict(X_Test)
            
                predictions_hg = pd.DataFrame(predictions_hgb, columns=['Stuck'])

                st.write('Data yang diunggah')
                # Load JSON data from file
                json_file_path = 'test_data.json'
                
                data = load_json(json_file_path)

                # Extract the data array
                data_array = data["data"]

                # Convert JSON to DataFrame
                df = pd.DataFrame(data_array)
                st.dataframe(df)
                # col1, col2 = st.columns([7,5])
                # with col1:
                #     st.write('Data yang diunggah')
                #     st.dataframe(selected_data)
                # with col2:
                #     st.write('Hasil Prediksi')
                #     st.dataframe(predictions_hg)

                if(predictions_hgb == 0):
                    st.success('Pengeboran normal, tidak stuck')
                else:
                    st.danger('Pengeboran stuck')

        # Jika tidak ada file yang di-upload
        else:
            st.info("Silakan upload file Excel untuk melihat data.")

    if selected == "Sumber data":
        st.write('Berikut adalah sumber data pengeboran')

        with st.sidebar:
            visual_option = st.selectbox(
                "Pilih sumber data",
                ["Sumur A", "Sumur B", "Sumur C"]
            )

        selected_data = st.session_state.get('data_frame')
        if visual_option == "Sumur A":
            st.header('Sumur A')
            well_A_path = os.path.abspath('WELL_C.csv')
            df = pd.read_csv(well_A_path)
            st.dataframe(df)
        elif visual_option == "Sumur B":
            st.header('Sumur B')
            st.dataframe(selected_data)
        elif visual_option == "Sumur C":
            st.header('Sumur C')
            st.dataframe(selected_data)
        else:
            st.warning("Belum ada data")
           
    if selected == 'Visualisasi':
          st.write('Berikut adalah visualisasi berdasarkan data pengeboran')

# Check authentication status
name, authentication_status, username = authenticator.login("sidebar")

# Registrasi
def register():
    if st.sidebar.checkbox('Register'):
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False,location='sidebar')
            if email_of_registered_user:
                st.sidebar.success('Registrasi user berhasil')
                with open('auth.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)

if st.session_state["authentication_status"]:
    st.title(f'Hallo *{st.session_state["name"]}*')
    beranda_logged_in()
    authenticator.logout("Logout", "sidebar")
    run_once()

    ## jika code ini diaktifkan maka jika halaman direfresh saat sudah login, maka otomatis terlogout
    # st.session_state.rerun_called = False

elif st.session_state["authentication_status"] is False:
    st.sidebar.error('Username/password anda salah')
    beranda()
    register()
    
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Silakan masukkan username dan password anda')
    beranda()
    register()