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
from datetime import datetime, time

 #visualisation             
import matplotlib.pyplot as plt 
import seaborn as sns  
sns.set(color_codes=True)

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

@st.cache_resource
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def stuck_percentage(df):
    # Asumsikan df adalah DataFrame yang telah diinisialisasi sebelumnya
    data = df['Stuck'].value_counts(normalize=True) * 100  # Menghitung persentase

    # Warna untuk setiap bar: hijau untuk 'Tidak Stuck' dan merah untuk 'Stuck'
    colors = ['green', 'red']

    # Membuat bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=data.index, y=data.values, palette=colors, ax=ax)

    # Menambahkan label dan judul
    ax.set_xlabel('')
    ax.set_ylabel('Persentase (%)')
    ax.set_title('Persentase Stuck')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Tidak Stuck', 'Stuck'])

    # Menambahkan label persentase di atas bar
    for i, v in enumerate(data.values):
        ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')

    # Menampilkan plot di Streamlit
    st.pyplot(fig)

def data_visual_of_drilling(df):
    # Ambil data dari baris pertama untuk visualisasi
    filtered_data = df.iloc[:,1:]
    data_first_row = filtered_data.iloc[0]

    # Plotting
    fig, ax = plt.subplots(figsize=(40, 10))

    # Menentukan warna yang berbeda untuk setiap bar
    colors = plt.cm.get_cmap('tab10', len(data_first_row))

    # Plotting bars
    bars = ax.bar(data_first_row.index, data_first_row.values, color=colors(np.arange(len(data_first_row))))

    # Menambahkan label dan judul
    ax.set_xlabel('Kolom')
    ax.set_ylabel('Nilai')
    ax.set_title('Bar Plot Data Pengeboran')

    # Menampilkan nilai di atas setiap bar
    for bar, value in zip(bars, data_first_row.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, round(value, 2), ha='center', va='bottom')

    # Menampilkan grafik di Streamlit
    st.pyplot(fig)

def time_filter():
    st.title('Filter Waktu')

    # Filter Tanggal (hanya satu tanggal)
    selected_date = st.date_input('Pilih Tanggal')

    # Filter Waktu (setiap jam)
    selected_hour = st.selectbox(
        'Pilih Jam:',
        [time(hour) for hour in range(24)]
    )

    # Konversi tanggal dan waktu yang dipilih menjadi objek datetime lengkap
    if selected_date:
        selected_datetime = datetime.combine(selected_date, selected_hour)
        return selected_datetime, selected_date

def data_filter_by_time(df, date, hour):
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])

    # Filter data untuk waktu tertentu
    filtered_data = df[(df['Date-Time'].dt.date == pd.to_datetime(date).date()) &
                    (df['Date-Time'].dt.hour == hour) &
                    (df['Date-Time'].dt.minute == 0) &
                    (df['Date-Time'].dt.second == 0)]

    return filtered_data


def beranda_logged_in():
    with st.sidebar:
        selected=option_menu(
            menu_title='Menu',
            options=['Prediksi','Sumber data'],
            default_index=0,
        )
    if selected == "Prediksi":
        st.write('Silahkan untuk melakukan prediksi status pengeboran')
        model_path = os.path.abspath('hgb_model.pkl')
        loaded_model = joblib.load(model_path)

        # uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx", "xls"])
        
        # Jika ada file yang di-upload
        # if uploaded_file is not None:
        #         df = load_excel_file(uploaded_file)
        #         st.session_state['data_frame'] = df

        drilling_data_json = st.text_area("Masukan Data Pengeboran (JSON)")
        pred_btn = st.button("Prediksi", type="primary")

        if 'data_frame' not in st.session_state:
            st.session_state['data_frame']

        if pred_btn or st.session_state['data_frame'] is not None:
            if drilling_data_json:
                try:
                    # Load JSON data
                    data = json.loads(drilling_data_json)

                    # Convert to DataFrame
                    data_array = data["data"]
                    df = pd.DataFrame(data_array)
                    st.session_state['data_frame'] =  df

                except json.JSONDecodeError:
                    st.error("Format JSON tidak benar")

                except ValueError as e:
                    # st.error(f"Error converting JSON to DataFrame: {e}")
                    st.error("Terjadi kesalahan")

            else:
                st.info("Silakan masukan data pengebroan lima menit terakhir (30 kumpulan data)")

            if st.session_state['data_frame'] is not None:
                try:
                    # Meenyeleksi kolom
                    selected_data = st.session_state['data_frame'][['BitDepth','BVDepth','Scfm','LogDepth','Hkld','Stuck']]

                    # Data Preprocessing
                    selected_data_np = np.array(selected_data)
                    selected_data_float = selected_data_np.astype(float)
                    
                    # PH = 5 menit
                    n_steps_in = 30
                    X_Test = selected_data_float.reshape((1, selected_data_float.shape[0]*selected_data_float.shape[1]))
                    print(X_Test.shape)

                    # Menggunakan model yang dimuat untuk membuat prediksi
                    predictions = loaded_model.predict(X_Test)

                    st.write('Data yang diunggah')

                    # Convert JSON to DataFrame
                    # df = pd.DataFrame(st.session_state['data_frame'])
                    st.dataframe(st.session_state['data_frame'])

                    if(predictions == 0):
                        st.success('Pengeboran normal, tidak stuck')
                    else:
                        st.danger('Pengeboran stuck')

                except json.JSONDecodeError:
                    st.error("Format JSON tidak benar")

                except ValueError as e:
                    # st.error(f"Error converting JSON to DataFrame: {e}")
                    st.error("Terjadi kesalahan")
        # Jika tidak ada file yang di-upload
        else:
            st.info("Silakan data pengebroan lima menit terakhir (30 kumpulan data)")

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
            well_path = os.path.abspath('WELL_A.csv')
            df = pd.read_csv(well_path)
            st.dataframe(df)
            stuck_percentage(df)

            #set datetime
            selected_datetime, selected_date  = time_filter()
            st.write(selected_datetime)
            
            #filter data by datetime
            filtered_data_by_datetime = data_filter_by_time(df, selected_date, selected_datetime.hour)
            st.dataframe(filtered_data_by_datetime)

            #data visualization of drilling
            if not filtered_data_by_datetime['Date-Time'].isna().all():
                data_visual_of_drilling(filtered_data_by_datetime)

        elif visual_option == "Sumur B":
            st.header('Sumur B')
            well_path = os.path.abspath('WELL_B.csv')
            df = pd.read_csv(well_path)
            st.dataframe(df)
            stuck_percentage(df)

            #set datetime
            selected_datetime, selected_date  = time_filter()
            st.write(selected_datetime)
            
            #filter data by datetime
            filtered_data_by_datetime = data_filter_by_time(df, selected_date, selected_datetime.hour)
            st.dataframe(filtered_data_by_datetime)

            #data visualization of drilling
            if not filtered_data_by_datetime['Date-Time'].isna().all():
                data_visual_of_drilling(filtered_data_by_datetime)

        elif visual_option == "Sumur C":
            st.header('Sumur C')
            well_path = os.path.abspath('WELL_C.csv')
            df = pd.read_csv(well_path)
            st.dataframe(df)
            stuck_percentage(df)

            #set datetime
            selected_datetime, selected_date  = time_filter()
            st.write(selected_datetime)
            
            #filter data by datetime
            filtered_data_by_datetime = data_filter_by_time(df, selected_date, selected_datetime.hour)
            st.dataframe(filtered_data_by_datetime)

            #data visualization of drilling
            if not filtered_data_by_datetime['Date-Time'].isna().all():
                data_visual_of_drilling(filtered_data_by_datetime)
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