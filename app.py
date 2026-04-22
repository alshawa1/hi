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
.card-warning { background-color: rgba(255, 193, 7, 0.1); border-left: 5px solid #FFC107; padding: 20px; border-radius: 5px; }
.card-danger { background-color: rgba(244, 67, 54, 0.1); border-left: 5px solid #F44336; padding: 20px; border-radius: 5px; }
.team-section { background: rgba(158, 158, 158, 0.1); padding: 30px; border-radius: 15px; margin-top: 50px; text-align: center; }
.team-title { font-family: 'Courier New', Courier, monospace; font-size: 28px; font-weight: bold; color: #E91E63; }
.team-member { font-size: 16px; margin: 5px 0; font-weight: 500; }
.strategy-header { background: linear-gradient(90deg, #1e88e5, #1565c0); color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- Logic: Prescriptive Strategy Function ---
def get_strategy_label(prob, contact_history, month_val):
    if prob >= 0.85:
        return "🔥 أولوية قصوى: اتصل الآن!"
    elif 0.60 <= prob < 0.85:
        if contact_history == 1:
            return "🤝 عميل وفيّ: ذكّره بنجاحاته السابقة."
        elif str(month_val).lower() in ['mar', 'sep', 'oct', 'dec']:
            return f"📅 توقيت ذهبي: استغل شهر {month_val}."
        else:
            return "⏳ توصية: عميل مهتم، انتظر الوقت الأمثل."
    elif 0.30 <= prob < 0.60:
        return "⚠️ نصيحة: احتمالية متوسطة، مكالمتين كحد أقصى."
    else:
        return "❌ استبعاد: وفر مجهودك وتكلفة المكالمة."

# --- Load Model Artifacts or Train on the Fly ---
@st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load('log_model.pkl')
        scaler = joblib.load('scaler.pkl')
        expected_cols = joblib.load('expected_columns.pkl')
        return model, scaler, expected_cols
    except Exception:
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

# --- Page 1: Individual Prediction ---
def predictive_page():
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
        df_in_p = df_in.drop('education', axis=1)

        cat_cols = df_in_p.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df_in_p, columns=cat_cols)
        df_final = df_encoded.reindex(columns=expected_cols, fill_value=0)
        X_scaled = scaler.transform(df_final)

        # --- Prediction Panel ---
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.write("### 🤖 نظام تحليل ودعم القرار (AI Support):")
            if st.button("🔮 P R E D I C T", use_container_width=True):
                prob = model.predict_proba(X_scaled)[0][1]
                
                contact_history = df_in['contacted_before'].iloc[0]
                month_val = df_in['month'].iloc[0]
                
                strategy_label = get_strategy_label(prob, contact_history, month_val)
                
                if prob >= 0.85:
                    card_class, status_icon = "card-success", "🔥"
                elif prob >= 0.60:
                    card_class, status_icon = "card-success", "🤝"
                elif prob >= 0.30:
                    card_class, status_icon = "card-warning", "⚠️"
                else:
                    card_class, status_icon = "card-danger", "⛔"
                
                st.write("---")
                st.markdown(f"""
                <div class='{card_class}'>
                    <div style='font-size: 22px; font-weight: bold;'>📊 التحليل التوقعي (Predictive Analysis): {prob*100:.1f}%</div>
                    <hr style='margin: 10px 0; border-top: 1px solid rgba(0,0,0,0.1);'>
                    <div style='font-size: 18px; line-height: 1.6;'>
                        {status_icon} <b>التوجيه الاستراتيجي (Prescriptive Strategy):</b><br>
                        {strategy_label}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if prob >= 0.6:
                    st.balloons()

# --- Page 2: Prescriptive Strategy (Batch Processing) ---
@st.cache_data
def get_prescriptive_data():
    # Load and Preprocess for batch
    df = pd.read_csv('bank-additional-full (1).csv', sep=';')
    df_raw = df.copy()
    
    if 'duration' in df.columns:
        df = df.drop('duration', axis=1)
    
    for col in ['job', 'marital', 'housing', 'loan']:
        mode_val = df[df[col] != 'unknown'][col].mode()[0]
        df[col] = df[col].replace('unknown', mode_val)

    df['contacted_before'] = (df['pdays'] != 999).astype(int)
    df.loc[df['pdays'] == 999, 'pdays'] = 0
    
    edu_map = {'illiterate':0, 'unknown':1, 'basic.4y':2, 'basic.6y':3, 'basic.9y':4, 'high.school':5, 'professional.course':6, 'university.degree':7}
    df['education_level'] = df['education'].map(edu_map)
    df = df.drop('education', axis=1)
    df.loc[df['campaign'] > 15, 'campaign'] = 15
    df = df.drop(['emp.var.rate', 'nr.employed', 'y'], axis=1, errors='ignore')
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_final = df_encoded.reindex(columns=expected_cols, fill_value=0)
    
    X_scaled = scaler.transform(df_final)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    # Merge results
    df_raw['Probability'] = probs
    df_raw['contacted_before'] = df['contacted_before']
    
    # Apply Strategy Label
    df_raw['Recommended_Action'] = df_raw.apply(lambda r: get_strategy_label(r['Probability'], r['contacted_before'], r['month']), axis=1)
    
    # Sort and take top 500
    top_500 = df_raw.sort_values(by='Probability', ascending=False).head(500)
    return top_500[['age', 'job', 'marital', 'month', 'Probability', 'Recommended_Action']].rename(columns={
        'age': 'Age', 'job': 'Job', 'marital': 'Marital', 'month': 'Month', 'Recommended_Action': 'Strategic_Action'
    })

def prescriptive_page():
    st.markdown("<div class='main-header'>🎯 Smart Call List (Prescriptive Strategy)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Top 500 High-Potential Clients Identified by AI</div>", unsafe_allow_html=True)
    
    with st.spinner("Analyzing full database..."):
        top_clients = get_prescriptive_data()
    
    st.markdown("""
    <div class='strategy-header'>
        💡 <b>How to use this list:</b> This table lists 500 customers with the highest likelihood of conversion. 
        Staff should prioritize the top rows to maximize the bank's ROI.
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Table
    st.dataframe(
        top_clients.style.format({'Probability': "{:.2%}"})
        .background_gradient(subset=['Probability'], cmap='GnBu'),
        use_container_width=True,
        height=600
    )
    
    # Export
    csv = top_clients.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 Download Smart Call List (CSV)",
        data=csv,
        file_name='NovaTrust_Smart_Call_List.csv',
        mime='text/csv',
    )

# --- Navigation & Shared Components ---
menu = st.sidebar.radio("Navigation", ["🔍 Individual Prediction", "📋 Prescriptive Strategy List"])

if menu == "🔍 Individual Prediction":
    predictive_page()
else:
    prescriptive_page()

# --- Team Section (Shared) ---
st.markdown("<div class='team-section'>", unsafe_allow_html=True)
st.markdown("<div class='team-title'>🛡️ لا تراجع ولا استسلام 🛡️</div><br>", unsafe_allow_html=True)
st.markdown("<b>Designed & Engineered By:</b>", unsafe_allow_html=True)

image_loaded = False
for img_name in ["teamna.jpeg", "team.jpg", "team.png"]:
    if os.path.exists(img_name):
        st.image(img_name, use_column_width=True)
        image_loaded = True
        break

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
