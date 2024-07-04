import streamlit as st
import streamlit_authenticator as stauth
import yaml
import joblib
import pandas as pd
import numpy as np
import altair as alt
from altair import datum
import os
from yaml.loader import SafeLoader
from streamlit_option_menu import option_menu
from PIL import Image
import json
from datetime import datetime, time, timedelta

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
        st.error(f"An error occurred while reading the file: {e}")
        return None


def beranda():
    st.title('Welcome!!')
    st.write('In the Drilling Status Prediction System Application of PT. Parama Data Unit')
    st.image("drilling.png", width=200)

@st.cache_resource
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def stuck_percentage(df):
    # Hitung jumlah kemunculan tiap nilai
    count_values = df['Stuck'].value_counts(normalize=True) * 100

    # Konversi ke DataFrame untuk visualisasi
    count_df = pd.DataFrame({
        'Status': ['Normal', 'Stuck'],
        'Percentage': [count_values[0], count_values[1]],
    })

    # Buat chart menggunakan Altair
    bar_chart = alt.Chart(count_df).mark_bar().encode(
        x=alt.X('Status', sort=['Normal', 'Stuck']),
        y='Percentage',
        color=alt.condition(
            alt.datum.Status == 'Stuck',
            alt.value('#A91D3A'),     # Warna merah untuk 'Stuck'
            alt.value('#BBE9FF')     # Warna biru untuk 'Normal'
        )
    ).properties(
        title='Persentase Stuck vs Normal'
    )

    # Tampilkan chart di Streamlit
    st.altair_chart(bar_chart, use_container_width=True)

def data_visual_of_drilling(df):
    # Ambil data dari baris pertama untuk visualisasi
    filtered_data = df.iloc[:,1:]
    data_first_row = filtered_data.iloc[0]

    # Membuat line chart
    st.write("Line Chart:")
    st.line_chart(data_first_row)

def data_visual_of_drilling_by_hour(filtered_data_by_datetime):
    # Pastikan kolom 'Date-Time' dalam format datetime
    filtered_data_by_datetime.loc[:, 'Date-Time'] = pd.to_datetime(filtered_data_by_datetime.loc[:, 'Date-Time'])

    # Judul aplikasi
    st.title('Data Visualization')

    # Multiselect untuk memilih kolom
    # columns = st.multiselect('Pilih kolom untuk divisualisasikan', filtered_data_by_datetime.iloc[:,1:].columns)
    
    #Visualisasi
    # Mengubah data dari wide format ke long format
    datas_1 = filtered_data_by_datetime.melt('Date-Time', var_name='Variable', value_name='Value', value_vars=filtered_data_by_datetime.iloc[:,1:7].columns)
    datas_2 = filtered_data_by_datetime.melt('Date-Time', var_name='Variable', value_name='Value', value_vars=filtered_data_by_datetime.iloc[:,7:13].columns)
    datas_3 = filtered_data_by_datetime.melt('Date-Time', var_name='Variable', value_name='Value', value_vars=filtered_data_by_datetime.iloc[:,13:18].columns)
    datas_4 = filtered_data_by_datetime.melt('Date-Time', var_name='Variable', value_name='Value', value_vars=filtered_data_by_datetime.iloc[:,18:22].columns)
    # Membuat layout grid dengan 2 kolom dan 2 baris
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    
    # Inisialisasi chart dengan Altair
    with col1:
        st.write('Graph 1')
        chart = alt.Chart(datas_1).mark_line().encode(
            x='Value:Q',  # Menggunakan kolom 'Date-Time' sebagai sumbu x
            y= 'Date-Time:T',  # Menggunakan nilai dari kolom yang dipilih sebagai sumbu y
            color='Variable:N',  # Warna berdasarkan kolom yang dipilih
            tooltip=['Date-Time', 'Variable', 'Value']  # Menampilkan tooltip dengan kolom yang relevan
        ).properties(
                width=800,
            height=400
        ).interactive()

        # Tampilkan plot menggunakan st.altair_chart
        st.altair_chart(chart, use_container_width=True)
        
    with col2:
        st.write('Graph 2')
        chart = alt.Chart(datas_2).mark_line().encode(
            x='Value:Q',  # Menggunakan kolom 'Date-Time' sebagai sumbu x
            y= 'Date-Time:T',  # Menggunakan nilai dari kolom yang dipilih sebagai sumbu y
            color='Variable:N',  # Warna berdasarkan kolom yang dipilih
            tooltip=['Date-Time', 'Variable', 'Value']  # Menampilkan tooltip dengan kolom yang relevan
        ).properties(
                width=800,
            height=400
        ).interactive()

        # Tampilkan plot menggunakan st.altair_chart
        st.altair_chart(chart, use_container_width=True)

    with col3:
        st.write('Graph 3')
        chart = alt.Chart(datas_3).mark_line().encode(
            x='Value:Q',  # Menggunakan kolom 'Date-Time' sebagai sumbu x
            y= 'Date-Time:T',  # Menggunakan nilai dari kolom yang dipilih sebagai sumbu y
            color='Variable:N',  # Warna berdasarkan kolom yang dipilih
            tooltip=['Date-Time', 'Variable', 'Value']  # Menampilkan tooltip dengan kolom yang relevan
        ).properties(
                width=800,
            height=400
        ).interactive()

        # Tampilkan plot menggunakan st.altair_chart
        st.altair_chart(chart, use_container_width=True)
        
    with col4:
        st.write('Graph 4')
        chart = alt.Chart(datas_4).mark_line().encode(
            x='Value:Q',  # Menggunakan kolom 'Date-Time' sebagai sumbu x
            y= 'Date-Time:T',  # Menggunakan nilai dari kolom yang dipilih sebagai sumbu y
            color='Variable:N',  # Warna berdasarkan kolom yang dipilih
            tooltip=['Date-Time', 'Variable', 'Value']  # Menampilkan tooltip dengan kolom yang relevan
        ).properties(
                width=800,
            height=400
        ).interactive()

        # Tampilkan plot menggunakan st.altair_chart
        st.altair_chart(chart, use_container_width=True)

    # Visualisasi dengan Altair jika ada kolom yang dipilih
    # if columns:       
    #     # Mengubah data dari wide format ke long format
    #     data_long = filtered_data_by_datetime.melt('Date-Time', var_name='Variable', value_name='Value', value_vars=columns)
                        
    #     # Inisialisasi chart dengan Altair
    #     chart = alt.Chart(data_long).mark_line().encode(
    #         x='Date-Time:T',  # Menggunakan kolom 'Date-Time' sebagai sumbu x
    #         y='Value:Q',  # Menggunakan nilai dari kolom yang dipilih sebagai sumbu y
    #         color='Variable:N',  # Warna berdasarkan kolom yang dipilih
    #         tooltip=['Date-Time', 'Variable', 'Value']  # Menampilkan tooltip dengan kolom yang relevan
    #     ).properties(
    #          width=800,
    #         height=400
    #     ).interactive()

    #     # Tampilkan plot menggunakan st.altair_chart
    #     st.altair_chart(chart, use_container_width=True)
    # else:
    #     st.write('Pilih setidaknya satu kolom untuk memulai visualisasi.')

def data_filter_by_time(df, date, start_hour, end_hour):
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])

    # Filter data untuk waktu tertentu antara start_hour dan end_hour
    filtered_data = df[(df['Date-Time'].dt.date == pd.to_datetime(date).date()) &
                       (df['Date-Time'].dt.hour >= start_hour) &
                       (df['Date-Time'].dt.hour < end_hour)]

    return filtered_data

def time_range_filter():
    st.title('Time Filter')

    # Filter Tanggal (hanya satu tanggal)
    selected_date = st.date_input('Select Date')

    # Filter Jam Awal
    selected_start_hour = st.selectbox(
        'Select Start Time:',
        [time(hour) for hour in range(24)]
    )

    # Filter Jam Akhir
    selected_end_hour = st.selectbox(
        'Select End Time:',
        [time(hour) for hour in range(24)]
    )

    # Validasi jam akhir harus setelah jam awal
    if selected_end_hour < selected_start_hour:
        st.error('The ending time must be after the starting time.')
        return None, None, None

    # Konversi tanggal dan waktu yang dipilih menjadi objek datetime lengkap
    if selected_date:
        start_datetime = datetime.combine(selected_date, selected_start_hour)
        end_datetime = datetime.combine(selected_date, selected_end_hour)
        return start_datetime, end_datetime, selected_date

def beranda_logged_in():
    with st.sidebar:
        selected=option_menu(
            menu_title='Menu',
            options=['Prediction','Data Source'],
            default_index=0,
        )
    if selected == "Prediction":
        st.write('Please make predictions about drilling status')
        model_path = os.path.abspath('knn_model_ph10.pkl')
        loaded_model = joblib.load(model_path)

        drilling_data_json = st.text_area("Input drilling data (JSON)")
        pred_btn = st.button("Prediction", type="primary")

        if 'data_frame' not in st.session_state:
            st.session_state['data_frame']

        if 'filter_time' not in st.session_state:
            st.session_state['filter_time'] = None
        
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
                    st.error("JSON format is incorrect")

                except ValueError as e:
                    st.error(f"Error converting JSON to DataFrame: {e}")
                    # st.error("Terjadi kesalahan")

            else:
                st.info("Please enter the last five minutes of drilling data (30 data sets)")

            if st.session_state['data_frame'] is not None:
                try:
                    # Meenyeleksi kolom
                    selected_data = st.session_state['data_frame'][['LogDepth','BitDepth','BVDepth','Hkld','Scfm']]

                    # Data Preprocessing
                    selected_data_np = np.array(selected_data)
                    selected_data_float = selected_data_np.astype(float)
                    
                    # PH = 5 menit
                    X_Test = selected_data_float.reshape((1, selected_data_float.shape[0]*selected_data_float.shape[1]))

                    # Menggunakan model yang dimuat untuk membuat prediksi
                    predictions = loaded_model.predict(X_Test)

                    st.write('Previously uploaded data')

                    st.dataframe(st.session_state['data_frame'])

                    if(predictions == 0):
                        st.success('Normal drilling, not stuck')
                    else:
                        st.error('Drilling stuck')

                except json.JSONDecodeError:
                    st.session_state['data_frame'] = None
                    st.error("JSON format is incorrect")

                except ValueError as e:
                    st.session_state['data_frame'] = None
                    st.error(f"Error converting JSON to DataFrame: {e}")

        # Jika tidak ada file yang di-upload
        else:
            st.info("Please enter the last five minutes of drilling data (30 data sets)")

    if selected == "Data Source":
        st.write('The following is the drilling data source')

        with st.sidebar:
            visual_option = st.selectbox(
                "Select a data source",
                ["Well A", "Well B", "Well C"]
            )

        selected_data = st.session_state.get('data_frame')
        if visual_option == "Well A":
            st.header('Well A')
            well_path = os.path.abspath('WELL_A.csv')
            df = pd.read_csv(well_path)
            st.dataframe(df)
            stuck_percentage(df)

            #set datetime
            start_datetime, end_datetime, selected_datetime = time_range_filter()

            if start_datetime is not None and end_datetime is not None and selected_datetime is not None:
                st.write('Start Time : ',start_datetime,)
                st.write('Start Time : ',end_datetime)

                #filter data by datetime
                filtered_data_by_datetime = data_filter_by_time(df, selected_datetime, start_datetime.hour, end_datetime.hour)
                st.dataframe(filtered_data_by_datetime)

                # data visualization of drilling
                if not filtered_data_by_datetime['Date-Time'].isna().all():
                    data_visual_of_drilling_by_hour(filtered_data_by_datetime)

        elif visual_option == "Well B":
            st.header('Well B')
            well_path = os.path.abspath('WELL_B.csv')
            df = pd.read_csv(well_path)
            st.dataframe(df)
            stuck_percentage(df)

            #set datetime
            start_datetime, end_datetime, selected_datetime = time_range_filter()

            if start_datetime is not None and end_datetime is not None and selected_datetime is not None:
                st.write('Start Time : ',start_datetime,)
                st.write('Start Time : ',end_datetime)

                #filter data by datetime
                filtered_data_by_datetime = data_filter_by_time(df, selected_datetime, start_datetime.hour, end_datetime.hour)
                st.dataframe(filtered_data_by_datetime)

                # data visualization of drilling
                if not filtered_data_by_datetime['Date-Time'].isna().all():
                    data_visual_of_drilling_by_hour(filtered_data_by_datetime)

        if visual_option == "Well C":
            st.header('Well C')
            well_path = os.path.abspath('WELL_C.csv')
            df = pd.read_csv(well_path)
            st.dataframe(df)
            stuck_percentage(df)

            #set datetime
            start_datetime, end_datetime, selected_datetime = time_range_filter()

            if start_datetime is not None and end_datetime is not None and selected_datetime is not None:
                st.write('Start Time : ',start_datetime,)
                st.write('Start Time : ',end_datetime)

                #filter data by datetime
                filtered_data_by_datetime = data_filter_by_time(df, selected_datetime, start_datetime.hour, end_datetime.hour)
                st.dataframe(filtered_data_by_datetime)

                # data visualization of drilling
                if not filtered_data_by_datetime['Date-Time'].isna().all():
                    data_visual_of_drilling_by_hour(filtered_data_by_datetime)

           
    if selected == 'Visualization':
          st.write('Below is a visualization based on drilling data')

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
    st.sidebar.error('Your username or password is incorrect')
    beranda()
    register()
    
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
    beranda()
    register()