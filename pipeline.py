import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Configure Streamlit page layouts
st.set_page_config(page_title="Churn & NLP Analytics Portal", layout="wide")

# ==========================================
# CACHED CORE PIPELINE BACKEND LOGIC
# ==========================================
@st.cache_resource
def train_pipeline_backend():
    """Simulates dataset records, vectorizes text feedback data, and trains the model."""
    np.random.seed(42)
    data_size = 200
    
    raw_data = {
        'CustomerID': range(1001, 1001 + data_size),
        'TenureMonths': np.random.randint(1, 72, size=data_size),
        'MonthlyCharges': np.random.uniform(20.0, 120.0, size=data_size),
        'CustomerFeedback': np.random.choice([
            "Great service, loving the stable system connection",
            "Terrible interface, service drops constantly and high billing",
            "Average platform performance but customer support is too slow",
            "Completely overcharged. Moving to an alternative provider soon"
        ], size=data_size),
        'Churn': np.random.choice([0, 1], p=[0.7, 0.3], size=data_size)
    }
    
    df = pd.DataFrame(raw_data)
    
    # NLP Vectorization
    tfidf = TfidfVectorizer(max_features=5, stop_words='english')
    feedback_features = tfidf.fit_transform(df['CustomerFeedback']).toarray()
    feedback_df = pd.DataFrame(feedback_features, columns=[f"nlp_{w}" for w in tfidf.get_feature_names_out()])
    
    X_features = pd.concat([df[['TenureMonths', 'MonthlyCharges']], feedback_df], axis=1)
    y_target = df['Churn']
    
    # Train-Test Segmentation
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.25, random_state=42, stratify=y_target)
    
    # Scale Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Model Learning Execution
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    classifier.fit(X_train_scaled, y_train)
    
    return classifier, scaler, tfidf, df

# Execute training and store backend instances
model, data_scaler, text_vectorizer, source_df = train_pipeline_backend()

# ==========================================
# INTERACTIVE USER INTERFACE DESIGN LAYOUT
# ==========================================
st.title("📊 Enterprise Customer Churn & Text Analytics Dashboard")
st.markdown("This interactive portal executes predictive modeling classifications and parses textual review inputs in real-time.")

# Sidebar Controls for Real-Time Predictions
st.sidebar.header("🔮 Simulate a Single Customer Profile")
input_tenure = st.sidebar.slider("Tenure Grid Location (Months)", min_value=1, max_value=72, value=24)
input_charges = st.sidebar.slider("Monthly Service Billing Charges ($)", min_value=20.0, max_value=120.0, value=65.0)
input_feedback = st.sidebar.text_area("Customer Text Review Feedback Tag", value="Average platform performance but customer support is too slow")

if st.sidebar.button("Run Predictive Diagnostics Pipeline"):
    # 1. Transform raw text input parameters using the trained TF-IDF instance
    user_text_vec = text_vectorizer.transform([input_feedback]).toarray()
    user_text_df = pd.DataFrame(user_text_vec, columns=[f"nlp_{w}" for w in text_vectorizer.get_feature_names_out()])
    
    # 2. Structure unified profile data vectors
    user_tabular = pd.DataFrame([[input_tenure, input_charges]], columns=['TenureMonths', 'MonthlyCharges'])
    user_profile_combined = pd.concat([user_tabular, user_text_df], axis=1)
    
    # 3. Standardize dimensions using the trained scaler instance
    user_scaled = data_scaler.transform(user_profile_combined)
    
    # 4. Generate classifications
    prediction = model.predict(user_scaled)[0]
    prediction_proba = model.predict_proba(user_scaled)[0]
    
    st.subheader("🎯 Real-Time Profile Diagnostic Output")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ Prediction System Result: HIGH RISK CHURN PROFILE")
        else:
            st.success("✅ Prediction System Result: RETAINED PROFILE (LOW CHURN RISK)")
            
    with col2:
        st.metric(label="Calculated Probability Matrix Score", value=f"{prediction_proba[prediction] * 100:.2f}% Confidence")

# Main Page Core Analytical Data Framework Views
st.subheader("📈 Raw Warehouse Customer Dataset Insights Overview")
st.dataframe(source_df.head(10), use_container_width=True)
