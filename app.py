import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Customer Churn Prediction App")
st.markdown("""
This application predicts whether a customer will churn (leave) a subscription-based service based on their 
profile and usage data. The model was trained on the Kaggle Telco Customer Churn Dataset.
""")

# Load the model and feature info
@st.cache_resource
def load_model():
    with open('churn_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('feature_info.pkl', 'rb') as file:
        feature_info = pickle.load(file)
        
    return model, feature_info

try:
    model, feature_info = load_model()
    categorical_features = feature_info['categorical_features']
    numerical_features = feature_info['numerical_features']
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "About the Model"])

# Prediction page
if page == "Prediction":
    st.header("Customer Information")
    
    if model_loaded:
        user_input = {}
        
        st.subheader("Personal Information")
        user_input['gender'] = st.selectbox("Gender", options=["Male", "Female"])
        user_input['SeniorCitizen'] = st.selectbox("Senior Citizen", options=[0, 1])
        user_input['Partner'] = st.selectbox("Has Partner", options=["Yes", "No"])
        user_input['Dependents'] = st.selectbox("Has Dependents", options=["Yes", "No"])
        
        # Account Information
        st.subheader("Account Information")
        user_input['tenure'] = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        user_input['Contract'] = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])
        
        # Billing
        st.subheader("Billing Information")
        user_input['MonthlyCharges'] = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=5.0)
        user_input['TotalCharges'] = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=user_input['MonthlyCharges'] * user_input['tenure'])
        
        input_df = pd.DataFrame([user_input])
        
        if st.button("Predict Churn"):
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Display results
            st.header("Prediction Result")
            
            # Create columns for result display
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction[0] == 1:
                    st.error("⚠️ Customer is likely to churn!")
                else:
                    st.success("✅ Customer is likely to stay!")
                
                st.write(f"Confidence: {prediction_proba[0][prediction[0]]:.2%}")
            
            with res_col2:
                # Display churn probability
                churn_proba = prediction_proba[0][1]
                st.metric(label="Churn Probability", value=f"{churn_proba:.2%}")
                
                st.progress(churn_proba)
            
            st.subheader("Key Factors to Consider")
            if user_input['Contract'] == "Month-to-month":
                st.write("📝 Month-to-month contracts have higher churn rates")
            
            if user_input['tenure'] < 12:
                st.write("⏱️ Customers with less than 12 months tenure are more likely to churn")
            
            st.subheader("Recommendation")
            if prediction[0] == 1:
                st.write("""
                Consider these retention strategies:
                - Offer a contract upgrade with incentives
                - Provide additional services at a discount
                - Implement a personalized loyalty program
                """)
            else:
                st.write("""
                To maintain customer loyalty:
                - Continue providing excellent service
                - Consider cross-selling complementary services
                - Periodically check in to ensure satisfaction
                """)
    else:
        st.warning("The prediction model could not be loaded. Please make sure the model file exists.")

elif page == "About the Model":
    st.header("About the Customer Churn Prediction Model")
    
    st.subheader("Data Source")
    st.write("""
    This model was trained on the [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 
    which contains information about customers of a fictional telecommunications company, including whether they left within the last month.
    """)
    
    st.subheader("Model Information")
    st.write("""
    We trained and compared several machine learning models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Neural Network (MLP)
    
    The best performing model was selected based on accuracy and saved for predictions.
    """)
    
    st.subheader("Features Used")
    if model_loaded:
        st.write("**Categorical Features:**")
        st.write(", ".join(categorical_features))
        
        st.write("**Numerical Features:**")
        st.write(", ".join(numerical_features))
        
st.markdown("---")
st.markdown("Customer Churn Prediction App | Created with Streamlit")