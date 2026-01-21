import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button:hover { background-color: #0b5ed7; }
    h1 { color: #212529; text-align: center; }
    .stSuccess { background-color: #d1e7dd; color: #0f5132; }
    .stError { background-color: #f8d7da; color: #842029; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_files():
    try:
      
        model = joblib.load('models/RandomForest/churn_pred_rf_model.pkl')
        label_enc = joblib.load('models/RandomForest/random_label.pkl')
 
        return model, label_enc
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, label_enc = load_files()


st.title("📉 Customer Churn Prediction System")
st.markdown("### Enter Customer Details to Predict Churn")
st.write("---")

if model is not None:
    with st.form("churn_form"):
       
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Personal Info")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            geography = st.selectbox("Geography (Country)", ["France", "Germany", "Spain"])
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

        with col2:
            st.subheader("🏦 Bank Details")
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
            balance = st.number_input("Balance", min_value=0.0, value=10000.0)
            num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
            tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)

        with col3:
            st.subheader("⚙️ Activity & Score")
            has_crcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
            complain = st.selectbox("Any Complain?", ["Yes", "No"]) 
            satisfaction_score = st.number_input("Satisfaction Score", min_value=0, value=3)
            point_earned = st.number_input("Points Earned", min_value=0, value=500)

        st.markdown("---")
        submit_btn = st.form_submit_button("🚀 Predict Result")


    if submit_btn:
        
 
            try:
                gender_encoded = label_enc.transform([gender])[0]
            except:
                gender_encoded = 1 if gender == "Male" else 0
            
            has_crcard_val = 1 if has_crcard == "Yes" else 0
            is_active_val = 1 if is_active_member == "Yes" else 0
            complain_val = 1 if complain == "Yes" else 0 

            geo_france = 1 if geography == "France" else 0
            geo_germany = 1 if geography == "Germany" else 0
            geo_spain = 1 if geography == "Spain" else 0

            # 4. Create DataFrame
            input_data = pd.DataFrame([[
                credit_score,
                gender_encoded,
                age,
                tenure,
                balance,
                num_of_products,
                has_crcard_val,
                is_active_val,
                estimated_salary,
                complain_val,
                satisfaction_score,
                point_earned,
                geo_france,
                geo_germany,
                geo_spain
            ]], columns=[
                'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Complain', 
                'Satisfaction Score', 'Point Earned', 
                'Geography_France', 'Geography_Germany', 'Geography_Spain'
            ])
            
            # Predict
            prediction = model.predict(input_data)
            
            # Display Result
            st.markdown("---")
            if prediction[0] == 1:
                st.error("🚨 Prediction: Customer will CHURN (Leave the Bank)")
                st.write("Suggestion: Please offer them a better plan or discount.")
            else:
                st.success("✅ Prediction: Customer will STAY")

     




