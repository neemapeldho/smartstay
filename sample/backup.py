import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------
# 🔥 MODERN UI — Dark Neon Theme
# ----------------------------------------------------
st.set_page_config(page_title="SmartStay Estimator", page_icon="🏨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: #E8E8E8;
}

/* Background */
body {
    background: linear-gradient(135deg, #0a0a12 0%, #1a0f1f 40%, #000000 100%);
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 25px rgba(0,255,255,0.15);
    margin-bottom: 25px;
}

/* Title */
.title {
    font-size: 55px;
    font-weight: 700;
    color: #00ffff;
    text-shadow: 0 0 15px rgba(0,255,255,0.8);
    text-align: center;
}

/* Section Header */
.section-header {
    font-size: 26px;
    color: #ff00ff;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff00ff, #00ffff);
    border: none;
    padding: 12px 25px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0 0 15px rgba(255,0,255,0.4);
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Prediction box */
.pred-box {
    padding: 20px;
    font-size: 34px;
    color: #00ffff;
    text-align: center;
    border-radius: 15px;
    background: rgba(0,0,0,0.5);
    border: 2px solid #00ffff;
    text-shadow: 0 0 8px rgba(0,255,255,0.8);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🏨 SmartStay Estimator</h1>", unsafe_allow_html=True)

# ----------------------------------------------------
# 📌 DATA LOADING
# ----------------------------------------------------
df = pd.read_csv("C:/Users/Neema P Eldho/OneDrive/Desktop/smartstay/sample/Airbnb_Data.csv")

df = df.drop(columns=[
    "id","amenities","description","first_review","last_review",
    "latitude","longitude","thumbnail_url"
], errors="ignore")

df["host_has_profile_pic"] = df["host_has_profile_pic"].fillna(df["host_has_profile_pic"].mode()[0])
df["host_identity_verified"] = df["host_identity_verified"].fillna(df["host_identity_verified"].mode()[0])
df["host_response_rate"] = df["host_response_rate"].fillna(df["host_response_rate"].mode()[0])

df["host_since"] = pd.to_datetime(df["host_since"])
df["host_experience"] = (datetime.now() - df["host_since"]).dt.days // 365
df["host_experience"] = df["host_experience"].fillna(df["host_experience"].median())
df = df.drop("host_since", axis=1)

df["zipcode"] = df["zipcode"].astype(str).str.split(".").str[0]
df["zipcode"] = df["zipcode"].fillna(df["zipcode"].mode()[0])

for c in ["neighbourhood","zipcode"]:
    df[c] = df[c].fillna(df[c].mode()[0])

for c in ['bathrooms','bedrooms','beds','review_scores_rating']:
    df[c] = df[c].fillna(df[c].mean())

original_cats = {}
for col in df.columns:
    if df[col].dtype == "object":
        original_cats[col] = sorted(df[col].astype(str).unique())

label_encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

Y = df["log_price"]
X = df.drop(columns=["log_price"])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------------
# 📝 STREAMLIT INPUTS
# ----------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Enter Hotel Details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Property Type", original_cats['property_type'])
    room_type = st.selectbox("Room Type", original_cats['room_type'])
    bed_type = st.selectbox("Bed Type", original_cats['bed_type'])
    cancellation_policy = st.selectbox("Cancellation Policy", original_cats['cancellation_policy'])

with col2:
    city = st.selectbox("City", original_cats['city'])
    name = st.selectbox("Name", original_cats['name'])
    neighbourhood = st.selectbox("Neighbourhood", original_cats['neighbourhood'])
    zipcode = st.selectbox("Zipcode", original_cats['zipcode'])

st.markdown("</div>", unsafe_allow_html=True)

# Metrics
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Property Metrics</div>", unsafe_allow_html=True)

accommodates = st.slider("Accommodates", 1, 16, 2)
bathrooms = st.slider("Bathrooms", 1, 16, 1)
bedrooms = st.slider("Bedrooms", 1, 16, 1)
beds = st.slider("Beds", 1, 30, 1)

st.markdown("</div>", unsafe_allow_html=True)

# Reviews + Host
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Host & Review Metrics</div>", unsafe_allow_html=True)

review_scores_rating = st.slider("Review Score", 0, 100, 85)
host_experience = st.slider("Host Experience (Years)", 0, 40, 5)
host_response_rate = st.slider("Host Response Rate (%)", 0, 100, 95)
number_of_reviews = st.slider("Reviews Count", 0, 1000, 50)

cleaning_fee = st.selectbox("Cleaning Fee", ["Yes","No"])
host_pic = st.selectbox("Host Has Profile Pic", ["Yes","No"])
host_verified = st.selectbox("Host Identity Verified", ["Yes","No"])
instant = st.selectbox("Instant Bookable", ["Yes","No"])

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 🔮 PREDICTION
# ----------------------------------------------------
if st.button("Predict Price", use_container_width=True):

    yes_no = {"Yes":1, "No":0}

    data = {
        'property_type': label_encoders['property_type'].transform([property_type])[0],
        'room_type': label_encoders['room_type'].transform([room_type])[0],
        'accommodates': accommodates,
        'bathrooms': bathrooms,
        'bed_type': label_encoders['bed_type'].transform([bed_type])[0],
        'cancellation_policy': label_encoders['cancellation_policy'].transform([cancellation_policy])[0],
        'cleaning_fee': yes_no[cleaning_fee],
        'city': label_encoders['city'].transform([city])[0],
        'host_has_profile_pic': yes_no[host_pic],
        'host_identity_verified': yes_no[host_verified],
        'host_response_rate': host_response_rate,
        'instant_bookable': yes_no[instant],
        'name': label_encoders['name'].transform([name])[0],
        'neighbourhood': label_encoders['neighbourhood'].transform([neighbourhood])[0],
        'number_of_reviews': number_of_reviews,
        'review_scores_rating': review_scores_rating,
        'zipcode': label_encoders['zipcode'].transform([zipcode])[0],
        'bedrooms': bedrooms,
        'beds': beds,
        'host_experience': host_experience
    }

    input_df = pd.DataFrame(data, index=[0])

    scaled = scaler.transform(input_df)
    log_pred = model.predict(scaled)[0]
    price = np.exp(log_pred)

    st.markdown(f"<div class='pred-box'>💰 Predicted Price: <b>₹{round(price * 89,2)}</b></div>", unsafe_allow_html=True)

