import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import base64

# OpenWeatherMap API Key
WEATHER_API_KEY = '52c94fc20a582a91a0994782b8ca2acc'

# Function to convert image to base64
def load_image(image_file):
    with open(image_file, "rb") as image:
        return base64.b64encode(image.read()).decode()

# Set background image
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            color: white;
        }}
        </style>
        """, unsafe_allow_html=True
    )
# Dummy user credentials
USER_CREDENTIALS = {
    'admin': 'password123',
    'user1': 'testpass'
}

# Function to check login
def check_login(username, password):
    return USER_CREDENTIALS.get(username) == password

# Login page
def login_page():
    set_background('https://media.istockphoto.com/id/1066762630/photo/oil-refinery-factory-with-beautiful-sky-at-dusk-for-energy-or-gas-industry-or-transportation.jpg?s=612x612&w=0&k=20&c=B__qBMMT7uSCF0Sp0P-aJLGuBtvzFSskvCGz25fu4q4=')
    st.markdown(
        """
        <style>
        .stApp
        {
            background-color:#66cdaa;
            color:white;
            display:flex;
            justify-content:center;
            align-content:center;
        }
        </style>

        """, unsafe_allow_html=True
    )

    st.markdown(
    """
    <div style='text-align: center; font-size: 36px; color: #000080; font-weight:bold;'>
        Threat Zone of oil and gas refineries
    </div>
    """,
    unsafe_allow_html=True)

    st.title("Login Page")

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        if check_login(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Incorrect username or password.")

# Main app content
def main_app():
    #set_background('https://media.istockphoto.com/id/182703770/photo/three-pumpjacks-at-dawn.jpg?s=612x612&w=0&k=20&c=qZhO-hnXBm0X8jlnBY01oi0nlkw98xvhWl8CIRIl0DM=')
    st.markdown(
        """
        <style>
        .stApp
        {
            background-color:#040720;
            color:white;
            display:flex;
            justify-content:center;
            align-content:center;
        }
        </style>

        """, unsafe_allow_html=True
    )

    st.title("Industry Threat Zone Stability Class and Weather Data")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        # Select row by index or industry name
        option = st.selectbox('Choose row selection method', ['By Index', 'By Industry Name'])

        if option == 'By Index':
            row_num = st.number_input("Enter row index", min_value=0, max_value=len(df) - 1, value=0)
            selected_row = df.iloc[row_num]
        else:
            industry_name = st.text_input("Enter Industry Name")
            selected_row = df[df['Industry Name'] == industry_name].iloc[0]

        if selected_row is not None:
            lat = selected_row['Latitude']
            lon = selected_row['Longitude']

            # Get weather data
            url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}'
            response = requests.get(url)
            weather_data = response.json()

            wind_speed = weather_data['wind']['speed']
            cloud_cover = weather_data['clouds']['all']
            sunset_time = weather_data['sys']['sunset']
            sunrise_time = weather_data['sys']['sunrise']

            # Calculate insolation and stability class
            stability_class = determine_stability_class(lat, lon, cloud_cover, wind_speed, sunrise_time, sunset_time)

            st.write(f"Industry: {selected_row['Industry Name']}")
            st.write(f"Latitude: {lat}, Longitude: {lon}")
            st.write(f"Wind Speed: {wind_speed} m/s")
            st.write(f"Cloud Cover: {cloud_cover}%")
            st.write(f"Stability Class: {stability_class}")
            st.write("---")

            # Explosion risk prediction
            predict_explosion_risk(selected_row)

# Function to calculate insolation
def calculate_insolation(lat, cloud_cover):
    today = date.today()
    N = pd.Period(today, freq='D').dayofyear
    delta = 23.45 * np.sin(np.deg2rad(360 * (N - 81) / 365))
    timeofdayinhrs = datetime.now().hour
    hourangle = (timeofdayinhrs - 12) * 15
    theta_not = np.arccos(np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(delta)) +
                          np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(hourangle)))
    d = (today - date(today.year, 1, 1)).days
    r_by_r0 = (1 + 0.0167 * np.cos(np.deg2rad(360 * (d - 3) / 365)))

    # Daytime insolation (Q)
    Q = 1361 * r_by_r0 ** 2 * np.cos(theta_not)

    if Q > 100:
        return Q, 'strong'
    elif 50 < Q <= 100:
        return Q, 'moderate'
    else:
        return Q, 'slight'

# Function to determine stability class
def determine_stability_class(lat, lon, cloud_cover, wind_speed, sunrise_time, sunset_time):
    Q, daytime_insolation_class = calculate_insolation(lat, cloud_cover)
    now = datetime.now().hour

    # Determine stability class based on conditions
    if now >= datetime.utcfromtimestamp(sunrise_time).hour + 1 and now <= datetime.utcfromtimestamp(sunset_time).hour - 1:
        if wind_speed < 2:
            return 'A' if daytime_insolation_class == 'strong' else 'B'
        elif wind_speed < 3:
            return 'B' if daytime_insolation_class == 'strong' else 'C'
        elif wind_speed < 4:
            return 'C' if daytime_insolation_class == 'moderate' else 'D'
        else:
            return 'D'
    else:
        return 'F' if wind_speed < 2 else 'E'

# Function to predict explosion risk
def predict_explosion_risk(selected_row):
    # Check if 'Gases' is present in the row and is a string
    if 'Gases' in selected_row and isinstance(selected_row['Gases'], str):
        # Split the gases by commas and handle missing values
        gases_list = selected_row['Gases'].split(', ')
        df_gases = pd.DataFrame([1] * len(gases_list), index=gases_list).T
    else:
        # If 'Gases' is not present or not a string, create an empty DataFrame
        df_gases = pd.DataFrame(columns=['Unknown'])

    # Ensure that df_gases is processed correctly
    df_gases.fillna('Unknown', inplace=True)

    # Check if 'Risk Category/Level' exists and create 'Explosion_Risk' column
    if 'Risk Category/Level' in selected_row:
        selected_row['Explosion_Risk'] = 1 if selected_row['Risk Category/Level'] == 'High' else 0
    else:
        selected_row['Explosion_Risk'] = 0  # Default risk to low if missing

    # Concatenate the gases data and explosion risk
    df_processed = pd.concat([df_gases, pd.DataFrame([selected_row['Explosion_Risk']], columns=['Explosion_Risk'])], axis=1)

    # Features (X) and target (y)
    X = df_processed.drop('Explosion_Risk', axis=1, errors='ignore')
    y = df_processed['Explosion_Risk']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_scaled, y)

    # Prediction
    y_pred = clf.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    st.write(f"Explosion Risk Prediction: {y_pred[0]}")
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")


# Session State to handle login status
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Show login page if the user is not logged in
if not st.session_state['logged_in']:
    login_page()
else:
    # Show the main app content if the user is logged in
    main_app()
