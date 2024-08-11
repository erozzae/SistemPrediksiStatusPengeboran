import streamlit as st
import streamlit_authenticator as stauth
import yaml

        
import os
import joblib

import pandas as pd
import numpy as np
import altair as alt
from altair import datum
from yaml.loader import SafeLoader
from streamlit_option_menu import option_menu
import json
from datetime import datetime, time, timedelta

from sqlalchemy import create_engine


st.set_page_config(
    page_title="Aplikasi Prediksi Status Pengeboran",
    page_icon="🎯",
)

# db_host = "integral-education.pudingbesar.com"
# db_database = "u483345376_drillingdb"
# db_username = "u483345376_eros"
# db_password = "Eros12345_"

with open('auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
)
    
# Fungsi untuk membuat koneksi ke database MySQL
def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        # Buat koneksi menggunakan sqlalchemy create_engine
        engine = create_engine(f'mysql+mysqlconnector://{user_name}:{user_password}@{host_name}/{db_name}')
        connection = engine.connect()
        print("Connection to MySQL DB successful")
    except Exception as e:
        print(f"The error '{e}' occurred")
    return connection

# Fungsi untuk menjalankan query dan mengembalikan hasil sebagai DataFrame
def read_query(engine, query):
    result = None
    try:
        result = pd.read_sql_query(query, engine)
    except Exception as e:
        print(f"The error '{e}' occurred")
    return result

def get_date_range(engine, table_name):
    result = None
    query = f"SELECT MIN(Date_Time) AS Start_Date, MAX(Date_Time) AS End_Date FROM {table_name}"
    try:
        result = pd.read_sql_query(query, engine)
    except Exception as e:
        print(f"The error '{e}' occurred")
    return result

# Inisialisasi state
if 'rerun_called' not in st.session_state:
    st.session_state.rerun_called = False

if 'data_frame' not in st.session_state:
    st.session_state['data_frame'] = None
    
if 'data_visual_1' not in st.session_state:
    st.session_state['data_visual_1'] = None

if 'data_visual_2' not in st.session_state:
    st.session_state['data_visual_2'] = None
    
if 'data_visual_2' not in st.session_state:
    st.session_state['data_visual_2'] = None
    
if 'data_visual_2' not in st.session_state:
    st.session_state['data_visual_2'] = None

# Fungsi untuk memanggil st._rerun() sekali saja
def run_once():
    if not st.session_state.rerun_called:
        st.session_state.rerun_called = True
        st.rerun()

def beranda():
    st.title('Welcome!!')
    st.write('In the Drilling Status Prediction System Application of PT. Parama Data Unit')
    st.image("drilling.png", width=200)

def data_visual_of_drilling(filtered_data_by_datetime):
    # Pastikan kolom 'Date-Time' dalam format datetime
    filtered_data_by_datetime.loc[:, 'Date_Time'] = pd.to_datetime(filtered_data_by_datetime.loc[:, 'Date_Time'])

    # Judul aplikasi
    st.title('Data Visualization')

    # Menambahkan jitter ke nilai Value
    jitter_amount = 0.2  # Adjust this value as needed
    def add_jitter(data, jitter_amount):
        data['Value'] += np.random.uniform(-jitter_amount, jitter_amount, size=len(data))
        return data
    
    # ambil semua kolom kecuali Date_Time
    
    # Visualisasi
    columns_g1 = filtered_data_by_datetime.columns[1:7] 
    columns_g2 = filtered_data_by_datetime.columns[7:13]
    columns_g3 = filtered_data_by_datetime.columns[13:18]
    columns_g4 = filtered_data_by_datetime.columns[18:23]
    
    
    # Membuat layout grid dengan 2 kolom dan 2 baris
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    
    # Inisialisasi chart dengan Altair
    with col1:
        st.write('Graph 1')
        selected_columns_1 = st.multiselect('Select columns for Graph 1', columns_g1, default=columns_g1.tolist())
        if selected_columns_1:
            datas_1 = add_jitter(filtered_data_by_datetime.melt('Date_Time', var_name='Variable', value_name='Value', value_vars=selected_columns_1), jitter_amount)
            st.session_state['data_visual_1'] = datas_1
            st.write('Graph 1')
            chart = alt.Chart(st.session_state.data_visual_1).mark_line().encode(
                x='Value:Q',  # Menggunakan kolom 'Date_Time' sebagai sumbu x
                y='Date_Time:T',  # Menggunakan nilai dari kolom yang dipilih sebagai sumbu y
                color='Variable:N',  # Warna berdasarkan kolom yang dipilih
                tooltip=['Date_Time', 'Variable', 'Value']  # Menampilkan tooltip dengan kolom yang relevan
            ).properties(
                width=800,
                height=400
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
    with col2:
        st.write('Graph 2')
        selected_columns_2 = st.multiselect('Select columns for Graph 2', columns_g2, default=columns_g2.tolist())
        if selected_columns_2:
            datas_2 = add_jitter(filtered_data_by_datetime.melt('Date_Time', var_name='Variable', value_name='Value', value_vars=selected_columns_2), jitter_amount)
            st.session_state['data_visual_2'] = datas_2
            st.write('Graph 2')
            chart = alt.Chart(st.session_state.data_visual_2).mark_line().encode(
                x='Value:Q',  # Menggunakan kolom 'Date_Time' sebagai sumbu x
                y='Date_Time:T',  # Menggunakan nilai dari kolom yang dipilih sebagai sumbu y
                color='Variable:N',  # Warna berdasarkan kolom yang dipilih
                tooltip=['Date_Time', 'Variable', 'Value']  # Menampilkan tooltip dengan kolom yang relevan
            ).properties(
                width=800,
                height=400
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

    with col3:
        st.write('Graph 3')
        selected_columns_3 = st.multiselect('Select columns for Graph 3', columns_g3, default=columns_g3.tolist())
        if selected_columns_3:
            datas_3 = add_jitter(filtered_data_by_datetime.melt('Date_Time', var_name='Variable', value_name='Value', value_vars=selected_columns_3), jitter_amount)
            st.session_state['data_visual_3'] = datas_3           
            st.write('Graph 3')
            chart = alt.Chart(st.session_state.data_visual_3).mark_line().encode(
                x='Value:Q',  # Menggunakan kolom 'Date_Time' sebagai sumbu x
                y='Date_Time:T',  # Menggunakan nilai dari kolom yang dipilih sebagai sumbu y
                color='Variable:N',  # Warna berdasarkan kolom yang dipilih
                tooltip=['Date_Time', 'Variable', 'Value']  # Menampilkan tooltip dengan kolom yang relevan
            ).properties(
                width=800,
                height=400
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        
    with col4:
        st.write('Graph 4')
        selected_columns_4 = st.multiselect('Select columns for Graph 4', columns_g4, default=columns_g4.tolist())
        if selected_columns_4:
            datas_4 = add_jitter(filtered_data_by_datetime.melt('Date_Time', var_name='Variable', value_name='Value', value_vars=selected_columns_4), jitter_amount)
            st.session_state['data_visual_4'] = datas_4
            st.write('Graph 2')
            chart = alt.Chart(st.session_state.data_visual_4).mark_line().encode(
                x='Value:Q',  # Menggunakan kolom 'Date_Time' sebagai sumbu x
                y='Date_Time:T',  # Menggunakan nilai dari kolom yang dipilih sebagai sumbu y
                color='Variable:N',  # Warna berdasarkan kolom yang dipilih
                tooltip=['Date_Time', 'Variable', 'Value']  # Menampilkan tooltip dengan kolom yang relevan
            ).properties(
                width=800,
                height=400
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

def data_filter_by_time(connection, table_name, start_date, end_date, start_hour, end_hour):

    # Mengonversi tanggal dan waktu mulai
    start_datetime = pd.to_datetime(start_date) + pd.to_timedelta(start_hour, unit='h')

    # Mengonversi tanggal dan waktu akhir
    end_datetime = pd.to_datetime(end_date) + pd.to_timedelta(end_hour, unit='h')

    # Membuat query SQL untuk mengambil data berdasarkan rentang waktu
    query = f"SELECT * FROM {table_name} WHERE Date_Time >= '{start_datetime}' AND Date_Time < '{end_datetime}'"

    # Mengeksekusi query dan mendapatkan hasil dalam bentuk DataFrame
    df = pd.read_sql(query, connection)

    return df

def time_range_filter():
    st.title('Time Filter For Visualization')

    # Filter Tanggal (rentang tanggal)
    selected_start_date = st.date_input('Select Start Date', key='start_date')
    selected_end_date = st.date_input('Select End Date', key='end_date')

    # Filter Jam Awal
    selected_start_hour = st.selectbox(
        'Select Start Time:',
        [None] + [time(hour) for hour in range(24)],
        format_func=lambda x: x.strftime('%H:%M') if x else 'Select Time'
    )

    # Filter Jam Akhir
    selected_end_hour = st.selectbox(
        'Select End Time:',
        [None] + [time(hour) for hour in range(24)],
        format_func=lambda x: x.strftime('%H:%M') if x else 'Select Time'
    )

    # Validasi input
    if selected_start_hour is None or selected_end_hour is None:
        st.error('Please select both start and end times.')
        return None, None, None, None

    if selected_start_date and selected_end_date:
        if selected_end_date < selected_start_date:
            st.error('The ending date must be after the starting date.')
            return None, None, None, None
        
        elif selected_end_date == selected_start_date:
            if selected_end_hour < selected_start_hour:
                st.error('The ending time must be after the starting time.')
                return None, None, None, None
            
        # Validasi rentang waktu maksimal 3 hari
        max_range = timedelta(days=3)
        if selected_end_date - selected_start_date > max_range:
            st.error('The date range must not exceed 3 days.')
            return None, None, None, None

        start_datetime = datetime.combine(selected_start_date, selected_start_hour)
        end_datetime = datetime.combine(selected_end_date, selected_end_hour)
        return start_datetime, end_datetime, selected_start_date, selected_end_date

    return None, None, None, None

def beranda_logged_in():
    connection = create_connection(st.secrets["db_host"], st.secrets["db_username"], st.secrets["db_password"], st.secrets["db_database"])
    with st.sidebar:
        selected=option_menu(
            menu_title='Menu',
            options=['Prediction','Data Source'],
            default_index=0,
        )
    if selected == "Prediction":
        st.write('Please make predictions about drilling status')
        model_path = os.path.abspath('dt_model.pkl')
        loaded_model = joblib.load(model_path)

        drilling_data_json = st.text_area("Input drilling data (JSON)", height=300)
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

            else:
                st.info("Please enter the last five minutes of drilling data (30 data sets)")

            if st.session_state['data_frame'] is not None:
                try:
                    # Meenyeleksi kolom
                    selected_data = st.session_state['data_frame'][['Scfm','Hkld','BitDepth','BVDepth','LogDepth']]

                    # Data Preprocessing
                    selected_data_np = np.array(selected_data)
                    selected_data_float = selected_data_np.astype(float)
                    
                    # PH = 5 menit
                    X_Test = selected_data_float.reshape((1, selected_data_float.shape[0]*selected_data_float.shape[1]))

                    # Menggunakan model yang dimuat untuk membuat prediksi
                    predictions = loaded_model.predict(X_Test)

                    st.write('Previously uploaded data :')

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
        
        with st.sidebar:
            visual_option = st.selectbox(
                "Select a data source",
                ["Well A", "Well B", "Well C"]
            )

        # selected_data = st.session_state.get('data_frame')
        
        if visual_option == "Well A":
            #avilable date range 
            avlb_date = get_date_range(connection,"well_a")
            if avlb_date is not None :
                st.info('Data Availability')
                st.success(f"Available Start Time from Data Sources: {avlb_date.iloc[0, 0]}")
                st.success(f"Available End Time from Data Sources: {avlb_date.iloc[0, 1]}")

            #set datetime
            start_datetime, end_datetime, selected_start_date, selected_end_date = time_range_filter()

            if start_datetime is not None and end_datetime is not None and selected_start_date is not None and selected_end_date is not None:
                st.write('Start Time : ', start_datetime)
                st.write('End Time : ', end_datetime)

                # #filter data by datetime
                filtered_data_by_datetime = data_filter_by_time(connection, "well_a", selected_start_date, selected_end_date, start_datetime.hour, end_datetime.hour)
                st.dataframe(filtered_data_by_datetime)

                # data visualization of drilling
                if not filtered_data_by_datetime['Date_Time'].isna().all():
                    data_visual_of_drilling(filtered_data_by_datetime)

        elif visual_option == "Well B":
            #avilable date range 
            avlb_date = get_date_range(connection,"well_b")
            if avlb_date is not None :
                st.info('Data Availability')
                st.success(f"Available Start Time from Data Sources: {avlb_date.iloc[0, 0]}")
                st.success(f"Available End Time from Data Sources: {avlb_date.iloc[0, 1]}")
            #set datetime
            start_datetime, end_datetime, selected_start_date, selected_end_date = time_range_filter()

            if start_datetime is not None and end_datetime is not None and selected_start_date is not None and selected_end_date is not None:
                st.write('Start Time : ', start_datetime)
                st.write('End Time : ', end_datetime)

                # #filter data by datetime
                filtered_data_by_datetime = data_filter_by_time(connection, "well_b", selected_start_date, selected_end_date, start_datetime.hour, end_datetime.hour)
                st.dataframe(filtered_data_by_datetime)

                # data visualization of drilling
                if not filtered_data_by_datetime['Date_Time'].isna().all():
                    data_visual_of_drilling(filtered_data_by_datetime)
                    
        elif visual_option == "Well C":
            #avilable date range 
            avlb_date = get_date_range(connection,"well_c")
            if avlb_date is not None :
                st.info('Data Availability')
                st.success(f"Available Start Time from Data Sources: {avlb_date.iloc[0, 0]}")
                st.success(f"Available End Time from Data Sources: {avlb_date.iloc[0, 1]}")

            #set datetime
            start_datetime, end_datetime, selected_start_date, selected_end_date = time_range_filter()

            if start_datetime is not None and end_datetime is not None and selected_start_date is not None and selected_end_date is not None:
                st.write('Start Time : ', start_datetime)
                st.write('End Time : ', end_datetime)

                # #filter data by datetime
                filtered_data_by_datetime = data_filter_by_time(connection, "well_c", selected_start_date, selected_end_date, start_datetime.hour, end_datetime.hour)
                st.dataframe(filtered_data_by_datetime)

                # data visualization of drilling
                if not filtered_data_by_datetime['Date_Time'].isna().all():
                    data_visual_of_drilling(filtered_data_by_datetime)


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

elif st.session_state["authentication_status"] is False:
    st.sidebar.error('Your username or password is incorrect')
    beranda()
    # register()
    
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')
    beranda()
    # register()