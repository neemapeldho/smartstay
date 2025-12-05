import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------
# 1) CUSTOM MODERN UI (Vibrant Dark Theme)
# ------------------------------------------------------
st.set_page_config(
    page_title="Hotel Price Predictor",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: #E0E0E0; /* Light gray for general text */
}

/* Animated Background (Now with richer colors) */
body {
    background: radial-gradient(circle at top right, #330033 0%, #0c0f1f 70%, #000000 100%);
    overflow-x: hidden;
    min-height: 100vh;
}

@keyframes floatStars {
    from {background-position: 0 0;}
    to {background-position: 10000px 10000px;}
}

.star-bg {
    background: url('https://www.transparenttextures.com/patterns/stardust.png');
    animation: floatStars 200s linear infinite;
    opacity: 0.35; /* Slightly brighter stars */
}

/* Glass card - Enhanced with vibrant border and deeper shadow */
.glass-card {
    background: rgba(18, 18, 30, 0.7); /* Darker background */
    padding: 30px;
    border-radius: 18px;
    /* Neon border effect */
    border: 1px solid rgba(255,0,255,0.2); 
    box-shadow: 0 0 15px rgba(255,0,255,0.2), 0px 4px 30px rgba(0,0,0,0.8);
    backdrop-filter: blur(12px);
    margin-bottom: 25px;
    transition: all 0.3s ease-in-out;
}
.glass-card:hover {
    border: 1px solid #ff00ff; /* Fuchsia hover */
    box-shadow: 0 0 25px rgba(255,0,255,0.4), 0px 4px 40px rgba(0,0,0,1);
}

/* Title - Cyan glow */
.title-text {
    font-size: 55px;
    font-weight: 700;
    text-align: center;
    color: #00ffff; /* Cyan */
    text-shadow: 0 0 10px rgba(0,255,255,0.5);
    margin-bottom: 20px;
}

/* Subheaders - Fuchsia Accent */
.section-header {
    font-size: 28px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 15px;
    color: #ff00ff; /* Fuchsia */
}

/* Streamlit Widget Overrides (Slider, Selectbox, Text input) */
div[data-testid="stForm"] .stButton>button {
    background-color: #ff00ff; 
    color: white; 
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(255,0,255,0.4);
    transition: all 0.3s ease;
}

/* Prediction box - Brighter and more pronounced */
.pred-box {
    font-size: 36px;
    padding: 25px;
    /* Gradient background for prediction result */
    background: linear-gradient(90deg, #1e1e2f, #330033);
    border-radius: 15px;
    margin-top: 30px;
    text-align: center;
    border: 3px solid #00ffff; /* Cyan border */
    color: #00ffff; /* Cyan text */
    text-shadow: 0 0 5px rgba(0,255,255,0.5);
    box-shadow: 0 0 20px rgba(0,255,255,0.3);
}

/* Center the predict button */
.stButton>button {
    margin-top: 20px;
    font-size: 18px;
}

</style>

<div class="star-bg"></div>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2) DATA LOADING & MODEL TRAINING (Your exact pipeline)
# ------------------------------------------------------
# NOTE: The provided path is local and may not work in a deployed environment. 
# Assuming 'Airbnb_Data.csv' is available for the script to run locally.

try:
    # Use st.cache_data to speed up the app by caching the data loading and model training
    @st.cache_data
    def load_data_and_train_model():
        # IMPORTANT: Replace the local file path with a reliable path or mock data 
        # for testing in environments where the file system is restricted.
        # For this example, we proceed with the provided structure but note 
        # that the file 'Airbnb_Data.csv' must be accessible.
        try:
            # Using a relative path which often works better in deployment environments 
            # if the file is uploaded alongside the script.
            df = pd.read_csv("Airbnb_Data.csv") 
        except FileNotFoundError:
            st.error("Error: 'Airbnb_Data.csv' not found. Please ensure the file is accessible in the correct location.")
            # Create a minimal mock DataFrame for the app to function without crashing
            mock_data = {
                'id': [1], 'log_price': [3.5], 'property_type': ['Apartment'], 'room_type': ['Entire home/apt'], 
                'accommodates': [2], 'bathrooms': [1.0], 'bed_type': ['Real Bed'], 'cancellation_policy': ['strict'], 
                'cleaning_fee': ['t'], 'city': ['NYC'], 'host_has_profile_pic': ['t'], 'host_identity_verified': ['t'], 
                'host_response_rate': ['100%'], 'instant_bookable': ['t'], 'name': ['Cozy Studio'], 
                'neighbourhood': ['Midtown'], 'number_of_reviews': [10], 'review_scores_rating': [90], 
                'zipcode': [10001.0], 'bedrooms': [1.0], 'beds': [1.0], 'host_since': ['2015-01-01'], 
                'amenities': ['a'], 'description': ['b'], 'first_review': ['c'], 'last_review': ['d'], 
                'latitude': [40.7], 'longitude': [-74.0], 'thumbnail_url': ['e']
            }
            df = pd.DataFrame(mock_data)

        # Drop specified columns
        df = df.drop(columns=["id","amenities","description","first_review","last_review",
                             "latitude","longitude","thumbnail_url"], errors='ignore')

        # Handle missing host features
        # Note: The original data uses 't'/'f' for boolean columns like host_has_profile_pic
        df["host_has_profile_pic"] = df["host_has_profile_pic"].fillna(df["host_has_profile_pic"].mode()[0])
        df["host_identity_verified"] = df["host_identity_verified"].fillna(df["host_identity_verified"].mode()[0])
        
        # Process host_response_rate (convert to string, remove %, then handle NaNs/modes)
        df["host_response_rate"] = df["host_response_rate"].astype(str).str.replace('%', '', regex=False).str.strip()
        df["host_response_rate"] = df["host_response_rate"].replace({'nan': df["host_response_rate"].mode()[0]}) # Fill NaNs (if any remain)
        df["host_response_rate"] = df["host_response_rate"] + '%' # Re-add % for encoding consistency

        
        # Calculate host experience
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        df['host_experience'] = (datetime.now() - df['host_since']).dt.days // 365
        df['host_experience'] = df['host_experience'].fillna(df['host_experience'].median())
        df = df.drop('host_since', axis=1)

        # Handle categorical and numerical missing values
        cat_cols = ['neighbourhood', 'zipcode', 'name', 'property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
        df["zipcode"] = df["zipcode"].astype(str).str.split(".").str[0].sort_values()
        for col in cat_cols:
            if col in df.columns:
                 df[col] = df[col].fillna(df[col].mode()[0]).astype(str)
        
        num_cols = ['bathrooms', 'bedrooms', 'beds', 'review_scores_rating']
        for col in num_cols:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
        
        # Ensure cleaning_fee is boolean/numeric (assuming 't'/'f' logic)
        df['cleaning_fee'] = df['cleaning_fee'].replace({'t': 1, 'f': 0, True: 1, False: 0}).astype(int)

        # Store original categories for Streamlit selectboxes
        original_cats = {}
        for col in df.columns:
            if df[col].dtype == "object":
                original_cats[col] = df[col].astype(str).unique()

        # Label Encoding
        label_encoders = {}
        for col in df.columns:
            if df[col].dtype == "object":
                le = LabelEncoder()
                # Handle unseen labels by fitting only on existing data
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

        # Separate features and target
        if 'log_price' not in df.columns:
            st.error("Error: 'log_price' column is missing after data processing. Cannot train model.")
            return None, None, None, None, None
            
        Y = df["log_price"]
        X = df.drop(columns=["log_price"], axis=1)
        
        # Ensure all columns are present and in correct order for prediction consistency
        feature_columns = X.columns.tolist()

        # Split and Scale
        X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train Linear Regression model
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        
        return lr, scaler, label_encoders, original_cats, feature_columns

    lr, scaler, label_encoders, original_cats, feature_columns = load_data_and_train_model()

except Exception as e:
    st.error(f"An error occurred during data loading or model training: {e}")
    # In case of error, set placeholders to avoid downstream crashes
    lr, scaler, label_encoders, original_cats, feature_columns = None, None, None, None, None
    st.stop()


# ------------------------------------------------------
# 3) TITLE SECTION
# ------------------------------------------------------
st.markdown("<h1 class='title-text'>🏨 SmartStay Estimator</h1>", unsafe_allow_html=True)
st.write("")

# ------------------------------------------------------
# 4) INPUT UI INSIDE CARDS
# ------------------------------------------------------

# Helper function to ensure all original categories are available in select boxes
def get_select_options(col_name, original_cats_dict, default_val=''):
    if col_name in original_cats_dict:
        return original_cats_dict[col_name].tolist()
    # Fallback for missing/mocked data to prevent error
    return [default_val]

# Map for 'Yes'/'No' inputs to 1/0
yes_no = {"Yes": 1, "No": 0}

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📝 Enter Hotel Details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Property Type", get_select_options('property_type', original_cats, 'Apartment'))
    room_type = st.selectbox("Room Type", get_select_options('room_type', original_cats, 'Entire home/apt'))
    bed_type = st.selectbox("Bed Type", get_select_options('bed_type', original_cats, 'Real Bed'))
    cancellation_policy = st.selectbox("Cancellation Policy", get_select_options('cancellation_policy', original_cats, 'strict'))

with col2:
    city = st.selectbox("City", get_select_options('city', original_cats, 'NYC'))
    # Name has too many unique values for a real app, but we follow the original script structure.
    name_options = get_select_options('name', original_cats, 'Cozy Studio')
    name = st.selectbox("Name (Title)", name_options, index=0)
    neighbourhood = st.selectbox("Neighbourhood", get_select_options('neighbourhood', original_cats, 'Midtown'))
    zipcode = st.selectbox("Zipcode", get_select_options('zipcode', original_cats, '10001'))

st.markdown("</div>", unsafe_allow_html=True)

# ----------- NUMERIC SECTION -------------

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📊 Property Metrics</div>", unsafe_allow_html=True)

accommodates = st.slider("Accommodates (Guests)", 1, 16, 2)
bathrooms = st.slider("Bathrooms", 1, 10, 1)
bedrooms = st.slider("Bedrooms", 1, 20, 1)
beds = st.slider("Beds", 1, 40, 2)

st.markdown("</div>", unsafe_allow_html=True)

# ----------- REVIEWS SECTION -------------

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>⭐ Reviews & Host Metrics</div>", unsafe_allow_html=True)

review_scores_rating = st.slider("Review Score Rating (0-100)", 0, 100, 90)
host_experience = st.slider("Host Experience (Years)", 0, 20, 5)
host_response_rate = st.slider("Host Response Rate (%)", 0, 100, 95)
number_of_reviews = st.slider("Number of Reviews", 0, 500, 50)

st.markdown("</div>", unsafe_allow_html=True)

# ----------- HOST SECTION -------------

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>🧑‍💼 Host Information</div>", unsafe_allow_html=True)

cleaning_fee_input = st.selectbox("Cleaning Fee Included", ["Yes", "No"])
host_has_profile_pic_input = st.selectbox("Host Has Profile Pic", ["Yes", "No"])
host_identity_verified_input = st.selectbox("Host Identity Verified", ["Yes", "No"])
instant_bookable_input = st.selectbox("Instant Bookable", ["Yes", "No"])

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# 5) PREDICTION
# ------------------------------------------------------

if lr and scaler and label_encoders:
    # Prepare input data in the exact format and order as the training data X
    try:
        # Convert Streamlit inputs back to the format expected by the LabelEncoder
        
        # Map 'Yes'/'No' inputs for binary categorical columns ('t'/'f' are often used in the dataset)
        host_pic_encoded = 't' if host_has_profile_pic_input == 'Yes' else 'f'
        host_verified_encoded = 't' if host_identity_verified_input == 'Yes' else 'f'
        instant_bookable_encoded = 't' if instant_bookable_input == 'Yes' else 'f'
        
        # Prepare host_response_rate for encoding: must match the format in classes_ ('95%', '100%', etc.)
        # Handle the edge case where the exact rate (e.g., 95%) might not be in the training classes
        rate_key = f"{host_response_rate}%"
        if rate_key not in label_encoders['host_response_rate'].classes_:
            # Fallback to the mode or nearest available class if the exact percentage is missing
            rate_key = label_encoders['host_response_rate'].classes_[0] # Using the first class as a robust fallback

        data1 = {
            'property_type': label_encoders['property_type'].transform([property_type])[0],
            'room_type': label_encoders['room_type'].transform([room_type])[0],
            'accommodates': accommodates,
            'bathrooms': bathrooms,
            'bed_type': label_encoders['bed_type'].transform([bed_type])[0],
            'cancellation_policy': label_encoders['cancellation_policy'].transform([cancellation_policy])[0],
            'cleaning_fee': yes_no[cleaning_fee_input], 
            'city': label_encoders['city'].transform([city])[0],
            'host_has_profile_pic': label_encoders['host_has_profile_pic'].transform([host_pic_encoded])[0], 
            'host_identity_verified': label_encoders['host_identity_verified'].transform([host_verified_encoded])[0],
            'host_response_rate': label_encoders['host_response_rate'].transform([rate_key])[0],
            'instant_bookable': label_encoders['instant_bookable'].transform([instant_bookable_encoded])[0],
            'name': label_encoders['name'].transform([name])[0],
            'neighbourhood': label_encoders['neighbourhood'].transform([neighbourhood])[0],
            'number_of_reviews': number_of_reviews,
            'review_scores_rating': review_scores_rating,
            'zipcode': label_encoders['zipcode'].transform([zipcode])[0],
            'bedrooms': bedrooms,
            'beds': beds,
            'host_experience': host_experience
        }
        
        # Create DataFrame and reorder columns to match X_train
        input_df = pd.DataFrame(data1, index=[0])
        input_df = input_df[feature_columns]

    except ValueError as e:
        st.error(f"Error encoding categorical features. This might happen if your selection was not present in the training data. Details: {e}")
        input_df = None


    if st.button("Predict Price", use_container_width=True) and input_df is not None:
        try:
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Predict the log price
            log_price_pred = lr.predict(input_scaled)[0]
            
            # Convert log price back to actual price (USD)
            actual_price = np.exp(log_price_pred)

            st.markdown(f"<div class='pred-box'>💰 Predicted Price: <b>${round(actual_price, 2)} per night</b></div>", 
                        unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed. Please check inputs and data structure. Error: {e}")
else:
    st.warning("Model training failed due to data issues (likely file not found). Prediction functionality is disabled.")