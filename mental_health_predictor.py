"""
Mental Health Treatment Prediction - Streamlit App
Deploy this app using: streamlit run app.py
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .treatment-yes {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .treatment-no {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    h1 {
        color: #1976D2;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    try:
        with open('mental_health_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("Please make sure all .pkl files are in the same directory as this script.")
        st.stop()

model, scaler, label_encoders, feature_names = load_model_artifacts()

# Title and description
st.title("🧠 Mental Health Treatment Prediction")
st.markdown("""
This application predicts whether an individual in the tech industry is likely to seek treatment 
for mental health issues based on various workplace and personal factors.
""")

st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Personal Information")
    
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        help="Your current age"
    )
    
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female", "Other"],
        help="Select your gender identity"
    )
    
    country = st.selectbox(
        "Country",
        options=["United States", "United Kingdom", "Canada", "Germany", "Netherlands", 
                 "Australia", "India", "Other"],
        help="Country where you work"
    )
    
    self_employed = st.selectbox(
        "Self Employed",
        options=["No", "Yes"],
        help="Are you self-employed?"
    )
    
    family_history = st.selectbox(
        "Family History of Mental Illness",
        options=["No", "Yes"],
        help="Do you have a family history of mental illness?"
    )
    
    work_interfere = st.selectbox(
        "Does Mental Health Interfere with Work?",
        options=["Never", "Rarely", "Sometimes", "Often", "Not applicable"],
        help="If you have a mental health condition, how often does it interfere with work?"
    )
    
    st.subheader("🏢 Company Information")
    
    no_employees = st.selectbox(
        "Number of Employees",
        options=["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"],
        help="How many employees does your company have?"
    )
    
    remote_work = st.selectbox(
        "Remote Work",
        options=["No", "Yes"],
        help="Do you work remotely?"
    )
    
    tech_company = st.selectbox(
        "Tech Company",
        options=["Yes", "No"],
        help="Is your employer primarily a tech company/organization?"
    )

with col2:
    st.subheader("💼 Workplace Benefits & Policies")
    
    benefits = st.selectbox(
        "Mental Health Benefits",
        options=["Yes", "No", "Don't know"],
        help="Does your employer provide mental health benefits?"
    )
    
    care_options = st.selectbox(
        "Awareness of Care Options",
        options=["Yes", "No", "Not sure"],
        help="Do you know the options for mental health care your employer provides?"
    )
    
    wellness_program = st.selectbox(
        "Wellness Program",
        options=["Yes", "No", "Don't know"],
        help="Has your employer discussed mental health as part of a wellness program?"
    )
    
    seek_help = st.selectbox(
        "Seek Help Resources",
        options=["Yes", "No", "Don't know"],
        help="Does your employer provide resources to learn more about mental health?"
    )
    
    anonymity = st.selectbox(
        "Anonymity Protected",
        options=["Yes", "No", "Don't know"],
        help="Is your anonymity protected if you take mental health treatment?"
    )
    
    leave = st.selectbox(
        "Ease of Taking Medical Leave",
        options=["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"],
        help="How easy is it to take medical leave for mental health?"
    )
    
    st.subheader("💭 Workplace Attitudes")
    
    mental_health_consequence = st.selectbox(
        "Mental Health Discussion Consequences",
        options=["No", "Yes", "Maybe"],
        help="Do you think discussing mental health issues would have negative consequences?"
    )
    
    phys_health_consequence = st.selectbox(
        "Physical Health Discussion Consequences",
        options=["No", "Yes", "Maybe"],
        help="Do you think discussing physical health issues would have negative consequences?"
    )
    
    coworkers = st.selectbox(
        "Discuss with Coworkers",
        options=["Yes", "No", "Some of them"],
        help="Would you discuss mental health with coworkers?"
    )
    
    supervisor = st.selectbox(
        "Discuss with Supervisor",
        options=["Yes", "No", "Some of them"],
        help="Would you discuss mental health with your supervisor?"
    )
    
    mental_health_interview = st.selectbox(
        "Discuss in Interview (Mental Health)",
        options=["Yes", "No", "Maybe"],
        help="Would you bring up mental health in an interview?"
    )
    
    phys_health_interview = st.selectbox(
        "Discuss in Interview (Physical Health)",
        options=["Yes", "No", "Maybe"],
        help="Would you bring up physical health in an interview?"
    )
    
    mental_vs_physical = st.selectbox(
        "Mental vs Physical Health Importance",
        options=["Yes", "No", "Don't know"],
        help="Do you feel your employer takes mental health as seriously as physical health?"
    )
    
    obs_consequence = st.selectbox(
        "Observed Consequences",
        options=["No", "Yes"],
        help="Have you observed negative consequences for coworkers with mental health issues?"
    )

st.markdown("---")

# Prediction button
if st.button("🔮 Predict Treatment Likelihood"):
    # Prepare input data
    input_data = {
        'Age': age,
        'Gender': gender,
        'Country': country,
        'self_employed': self_employed,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'no_employees': no_employees,
        'remote_work': remote_work,
        'tech_company': tech_company,
        'benefits': benefits,
        'care_options': care_options,
        'wellness_program': wellness_program,
        'seek_help': seek_help,
        'anonymity': anonymity,
        'leave': leave,
        'mental_health_consequence': mental_health_consequence,
        'phys_health_consequence': phys_health_consequence,
        'coworkers': coworkers,
        'supervisor': supervisor,
        'mental_health_interview': mental_health_interview,
        'phys_health_interview': phys_health_interview,
        'mental_vs_physical': mental_vs_physical,
        'obs_consequence': obs_consequence
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except:
                # If value not seen during training, use most common value
                input_df[col] = 0
    
    # Ensure correct feature order
    input_df = input_df[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("## 📊 Prediction Results")
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability[1] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Treatment Likelihood", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkred" if probability[1] > 0.5 else "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 70], 'color': '#fff9c4'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction box
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box treatment-yes">
            <h2>⚠️ Likely to Seek Treatment</h2>
            <h3>Confidence: {probability[1]*100:.1f}%</h3>
            <p>Based on the provided information, the model predicts that this individual is likely to seek treatment for mental health issues.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box treatment-no">
            <h2>✅ Unlikely to Seek Treatment</h2>
            <h3>Confidence: {probability[0]*100:.1f}%</h3>
            <p>Based on the provided information, the model predicts that this individual is unlikely to seek treatment for mental health issues.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed probabilities
    st.markdown("### 📈 Detailed Probabilities")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="No Treatment",
            value=f"{probability[0]*100:.2f}%",
            delta=f"{(probability[0]-0.5)*100:.2f}%" if probability[0] > 0.5 else f"{(probability[0]-0.5)*100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Seeks Treatment",
            value=f"{probability[1]*100:.2f}%",
            delta=f"{(probability[1]-0.5)*100:.2f}%" if probability[1] > 0.5 else f"{(probability[1]-0.5)*100:.2f}%"
        )
    
    # Model information
    with st.expander("ℹ️ About This Prediction"):
        st.write("""
        **Model Details:**
        - Algorithm: Random Forest Classifier
        - Accuracy: 83%
        - ROC-AUC Score: 0.8949
        - Training Data: Mental Health in Tech Survey (OSMI)
        
        **Important Note:**
        This prediction is based on machine learning analysis of survey data and should not be used as a substitute 
        for professional mental health assessment. If you're experiencing mental health concerns, please consult 
        with a qualified healthcare professional.
        """)

# Sidebar information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=100)
    st.markdown("## About")
    st.info("""
    This application uses machine learning to predict the likelihood of seeking mental health treatment 
    based on workplace and personal factors in the tech industry.
    
    **Model Performance:**
    - Accuracy: 83%
    - ROC-AUC: 0.89
    - Algorithm: Random Forest
    """)
    
    st.markdown("## 📚 Resources")
    st.markdown("""
    - [Mental Health America](https://www.mhanational.org/)
    - [NAMI](https://www.nami.org/)
    - [Crisis Text Line](https://www.crisistextline.org/): Text HOME to 741741
    - [National Suicide Prevention Lifeline](https://988lifeline.org/): 988
    """)
    
    st.markdown("---")
    st.markdown("**Developed with ❤️ using Streamlit**")
    st.markdown("Data Source: OSMI Mental Health in Tech Survey")