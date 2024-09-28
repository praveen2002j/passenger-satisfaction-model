import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('new_passengers.csv')
    return data

# Load the models
def load_model(model_name):
    return joblib.load(model_name)

# Define the Streamlit app
st.title("Airline Passenger Satisfaction Prediction System")

# Sidebar options
st.sidebar.header("Options")
selected_option = st.sidebar.selectbox("Choose an option", ["View Data", "Make Prediction"])

# Load data
data = load_data()

# Separate features and target variable
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use transform for test data

# View Data option
if selected_option == "View Data":
    st.subheader("Dataset: Airline Passenger Satisfaction")
    
    # Display data
    st.write(data.head())

    # Display dataset summary
    st.write("Summary Statistics:")
    st.write(data.describe())

    # Display charts for exploration
    st.subheader("Class Distribution")
    st.bar_chart(data['satisfaction'].value_counts())

# Prediction option
elif selected_option == "Make Prediction":
    st.subheader("Make a Prediction")
    
    # Load models (ensure these files exist in the same directory)
    model_catboost = load_model('catboost_model.joblib')
    model_lightgbm = load_model('lightgbm_model.joblib')
    
    # Feature inputs (based on the dataset)
    st.sidebar.subheader("Input Features")
    gender = st.sidebar.selectbox("Gender (0: Female, 1: Male)", [0, 1])
    age = st.sidebar.slider("Age", 18, 80, 30)
    flight_distance = st.sidebar.slider("Flight Distance", 0, 5000, 1000)
    inflight_service = st.sidebar.slider("Inflight Service (Rating 1-5)", 1, 5, 3)
    seat_comfort = st.sidebar.slider("Seat Comfort (Rating 1-5)", 1, 5, 3)
    cleanliness = st.sidebar.slider("Cleanliness (Rating 1-5)", 1, 5, 3)
    food_drink = st.sidebar.slider("Food & Drink (Rating 1-5)", 1, 5, 3)
    wifi_service = st.sidebar.slider("WiFi Service (Rating 1-5)", 1, 5, 3)
    entertainment = st.sidebar.slider("Entertainment (Rating 1-5)", 1, 5, 3)
    baggage_handling = st.sidebar.slider("Baggage Handling (Rating 1-5)", 1, 5, 3)
    departure_delay_minutes = st.sidebar.slider("Departure Delay Minutes", 0, 180, 0)
    online_boarding = st.sidebar.slider("Online Boarding (Rating 1-5)", 1, 5, 3)
    checkin_service = st.sidebar.slider("Check-in Service (Rating 1-5)", 1, 5, 3)
    gate = st.sidebar.slider("Gate (Rating 1-5)", 1, 5, 3)

    # User can select which model to use for prediction
    selected_model = st.selectbox("Choose a model", ["CatBoost", "LightGBM"])

    # Input features into a DataFrame for prediction
    input_data = pd.DataFrame({
        'gender': gender,  # Categorical variable, already encoded (1 for Male, 0 for Female)
        'age': age,  # Numeric value
        'customer_type': 0,  # Already encoded (0 for Regular, 1 for Loyal)
        'travel_type': 0,  # Already encoded (0 for Personal, 1 for Business)
        'class': 0,  # Already encoded (0 for Economy, 1 for Business)
        'distance': flight_distance,  # Numeric value
        'departure_delay_minutes': departure_delay_minutes,  # Numeric value
        'arrival_delay_minutes': 0,  # Placeholder; adjust as needed
        'dep_val_time_convenient': 0,  # Placeholder; adjust as needed
        'online_booking_service': 0,  # Placeholder; adjust as needed
        'onboard_service': inflight_service,  # Numeric value
        'seat_comfort': seat_comfort,  # Numeric value
        'leg_room_service': 0,  # Placeholder; adjust as needed
        'cleanliness': cleanliness,  # Numeric value
        'food_drink': food_drink,  # Numeric value
        'inflight_service': inflight_service,  # Numeric value
        'wifi_service': wifi_service,  # Numeric value
        'entertainment': entertainment,  # Numeric value
        'baggage_handling': baggage_handling,  # Numeric value
        'checkin_service': checkin_service,  # Numeric value
        'online_boarding': online_boarding,  # Numeric value
        'gate': gate,  # Numeric value
    }, index=[0])  # Ensure the input_data is a DataFrame with one row

    # Make sure the input_data has the same columns as the training data
    input_data = input_data[X.columns]  # Rearrange columns to match the training set

    # Apply scaling (use the same scaler as used in training)
    input_data_scaled = scaler.transform(input_data)

    # Perform the prediction
    if st.button("Predict Satisfaction"):
        if selected_model == "CatBoost":
            prediction = model_catboost.predict(input_data_scaled)
        else:
            prediction = model_lightgbm.predict(input_data_scaled)

        # Display the result
        st.write(f"Prediction: {'Satisfied' if prediction[0] == 1 else 'Not Satisfied'}")
