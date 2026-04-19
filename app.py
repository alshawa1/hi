import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Page Config ---
st.set_page_config(
    page_title="NovaTrust - AI Targeting System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Light Mode adaptive)
st.markdown("""
<style>
.main-header { font-size: 40px; color: #1E88E5; font-weight: bold; text-align: center; margin-bottom: -10px; }
.sub-header { font-size: 20px; color: #607D8B; text-align: center; margin-bottom: 30px; }
.card-success { background-color: rgba(76, 175, 80, 0.1); border-left: 5px solid #4CAF50; padding: 20px; border-radius: 5px; }
.card-danger { background-color: rgba(244, 67, 54, 0.1); border-left: 5px solid #F44336; padding: 20px; border-radius: 5px; }
.team-section { background: rgba(158, 158, 158, 0.1); padding: 30px; border-radius: 15px; margin-top: 50px; text-align: center; }
.team-title { font-family: 'Courier New', Courier, monospace; font-size: 28px; font-weight: bold; color: #E91E63; }
.team-member { font-size: 16px; margin: 5px 0; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --- Load Model Artifacts or Train on the Fly ---
@st.cache_resource
def load_or_train_model():
    # If pkl files exist, load them (ultra fast)
    try:
        model = joblib.load('log_model.pkl')
        scaler = joblib.load('scaler.pkl')
        expected_cols = joblib.load('expected_columns.pkl')
        return model, scaler, expected_cols
    except Exception:
        # If not on PC/GitHub, train it live (takes ~2 seconds on Streamlit Cloud)
        try:
            df = pd.read_csv('bank-additional-full (1).csv', sep=';')
            if 'duration' in df.columns:
                df = df.drop('duration', axis=1)
            df['target'] = df['y'].map({'yes': 1, 'no': 0})
            df = df.drop('y', axis=1)
            
            for col in ['job', 'marital', 'housing', 'loan']:
                mode_val = df[df[col] != 'unknown'][col].mode()[0]
                df[col] = df[col].replace('unknown', mode_val)

            df['contacted_before'] = (df['pdays'] != 999).astype(int)
            df.loc[df['pdays'] == 999, 'pdays'] = 0
            
            edu_map = {'illiterate':0, 'unknown':1, 'basic.4y':2, 'basic.6y':3, 'basic.9y':4, 'high.school':5, 'professional.course':6, 'university.degree':7}
            df['education_level'] = df['education'].map(edu_map)
            df = df.drop('education', axis=1)
            
            df.loc[df['campaign'] > 15, 'campaign'] = 15
            df = df.drop(['emp.var.rate', 'nr.employed'], axis=1, errors='ignore')
            
            df_ml = df.drop(columns=['age_group'], errors='ignore')
            categorical_cols = df_ml.select_dtypes(include=['object']).columns
            df_encoded = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True)
            
            X = df_encoded.drop('target', axis=1)
            y = df_encoded['target']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
            model.fit(X_scaled, y)
            
            return model, scaler, list(X.columns)
        except Exception as deep_e:
            st.error(f"Failed to find model and failed to read the CSV file! Error: {deep_e}")
            return None, None, None

with st.spinner("Initializing AI Model..."):
    model, scaler, expected_cols = load_or_train_model()

# --- Main App ---
st.markdown("<div class='main-header'>🏢 NovaTrust Bank</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-Powered Term Deposit Prediction System</div>", unsafe_allow_html=True)

if model:
    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("👤 Client Demographics")
        age = st.slider("Age", 18, 100, 35)
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox("Marital Status", ['single', 'married', 'divorced', 'unknown'])
        education = st.selectbox("Education", ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree', 'unknown'])
        
        st.header("💳 Financial Status")
        default = st.selectbox("Credit Default?", ['no', 'yes', 'unknown'])
        housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])
        
        st.header("📞 Campaign Info")
        contact = st.selectbox("Contact Method", ['cellular', 'telephone'])
        month = st.selectbox("Month of Call", ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign = st.slider("Number of Calls in this campaign", 1, 15, 1)
        
        st.header("⏳ History & Economy")
        never_contacted = st.checkbox("Never Contacted Before?", value=True)
        pdays = 999 if never_contacted else st.slider("Days since last contact", 0, 30, 5)
        previous = st.slider("Number of past contacts", 0, 7, 0)
        poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])
        
        st.markdown("---")
        euribor3m = st.number_input("Euribor 3M Rate (%)", value=1.0)
        cons_price_idx = st.number_input("Consumer Price Index", value=93.2)
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)

    # --- Processing Vector ---
    input_data = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m
    }
    
    df_in = pd.DataFrame([input_data])
    
    df_in['contacted_before'] = (df_in['pdays'] != 999).astype(int)
    df_in.loc[df_in['pdays'] == 999, 'pdays'] = 0
    
    edu_map = {'illiterate':0, 'unknown':1, 'basic.4y':2, 'basic.6y':3, 'basic.9y':4, 'high.school':5, 'professional.course':6, 'university.degree':7}
    df_in['education_level'] = df_in['education'].map(edu_map)
    df_in = df_in.drop('education', axis=1)

    cat_cols = df_in.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df_in, columns=cat_cols)
    
    df_final = df_encoded.reindex(columns=expected_cols, fill_value=0)
    X_scaled = scaler.transform(df_final)

    # --- Prediction Panel ---
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.write("### AI Customer Assessment:")
        if st.button("🔮 P R E D I C T", use_container_width=True):
            prob = model.predict_proba(X_scaled)[0][1]
            
            st.write("---")
            if prob >= 0.5:
                st.markdown(f"<div class='card-success'>✅ <b>High Potential!</b><br>Probability of subscribing: <b>{prob*100:.1f}%</b><br>Action: <b>CALL THIS CLIENT</b></div>", unsafe_allow_html=True)
                st.balloons()
            elif prob >= 0.2:
                st.info(f"🤔 **Moderate Potential**<br>Probability: **{prob*100:.1f}%**<br>Action: Worth a try if budget allows.", icon="📈")
            else:
                st.markdown(f"<div class='card-danger'>⛔ <b>Very Low Potential</b><br>Probability of subscribing: <b>{prob*100:.1f}%</b><br>Action: <b>SKIP CALL - SAVE MONEY (€6)</b></div>", unsafe_allow_html=True)

# --- Team Section ---
st.markdown("<div class='team-section'>", unsafe_allow_html=True)
st.markdown("<div class='team-title'>🛡️ لا تراجع ولا استسلام 🛡️</div><br>", unsafe_allow_html=True)
st.markdown("<b>Designed & Engineered By:</b>", unsafe_allow_html=True)

image_loaded = False
for img_name in ["teamna.jpeg", "صوره التيم.jpeg", "صورة التيم.jpeg", "team.jpg", "team.png", "صوره التيم.jpg"]:
    if os.path.exists(img_name):
        try:
            with open(img_name, "rb") as f:
                st.image(f.read(), use_column_width=True)
            image_loaded = True
            break
        except:
            pass

if not image_loaded:
    st.warning("⚠️ يرجى التأكد من تغيير اسم الصورة إلى team.jpg ليتم عرضها هنا.")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("<p class='team-member'>👨‍💻 Omar Khorshed<br><small>224225@eru.edu.eg</small></p>", unsafe_allow_html=True)
    st.markdown("<p class='team-member'>👨‍💻 Omar Gamal<br><small>224037@eru.edu.eg</small></p>", unsafe_allow_html=True)
with col_t2:
    st.markdown("<p class='team-member'>👩‍💻 Mariam Tamer<br><small>224015@eru.edu.eg</small></p>", unsafe_allow_html=True)
    st.markdown("<p class='team-member'>👩‍💻 Alhosna Ezzat<br><small>224222@eru.edu.eg</small></p>", unsafe_allow_html=True)

st.markdown("<br><hr style='opacity: 0.3;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 20px; color: #4CAF50; font-weight: bold;'>Under the Supervision of:</p>", unsafe_allow_html=True)
st.markdown("<p class='team-member'>🎓 Dr. Rowaida Ali</p>", unsafe_allow_html=True)
st.markdown("<p class='team-member'>👩‍🏫 TA. Mariam Abdelhamid</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
