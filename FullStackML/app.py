import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                             accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, roc_auc_score, roc_curve)

# Page Config
st.set_page_config(page_title="Regression vs Classification Pro", layout="wide")

# Load CSS (Added error handling)
current_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_dir, "style.css")
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # This will help you debug on the cloud
        st.sidebar.error(f"CSS file not found at: {file_path}")
load_css(css_path)

# --- TITLE SECTION ---
st.markdown("""
<div class="title-card">
    <h1>Penguin Species Analysis</h1>
    <p>Predicting <b>Sex</b> (Male/Female) using Regression and Classification</p>
</div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    # Loading penguins dataset
    data = sns.load_dataset("penguins").dropna()
    # Map sex to binary: Male=1, Female=0
    data["sex"] = data["sex"].map({"Male": 1, "Female": 0})
    return data

df = load_data()

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# --- MODEL TRAINING ---

# 1. SLR (Simple Linear Regression)
X_s = df[['body_mass_g']]
y_s = df['sex']
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

scaler_slr = StandardScaler()
X_tr_s_scaled = scaler_slr.fit_transform(X_tr_s)
X_te_s_scaled = scaler_slr.transform(X_te_s)
slr_model = LinearRegression().fit(X_tr_s_scaled, y_tr_s)

# 2. MLR (Multiple Linear Regression)
features = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']
X_m = df[features]
y_m = df['sex']
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

scaler_mlr = StandardScaler()
X_tr_m_scaled = scaler_mlr.fit_transform(X_tr_m)
X_te_m_scaled = scaler_mlr.transform(X_te_m)
mlr_model = LinearRegression().fit(X_tr_m_scaled, y_tr_m)

# 3. Logistic Regression
X_l = df[['body_mass_g']]
y_l = df['sex']
X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(X_l, y_l, test_size=0.2, random_state=42)

scaler_log = StandardScaler()
X_tr_l_scaled = scaler_log.fit_transform(X_tr_l)
X_te_l_scaled = scaler_log.transform(X_te_l)
lr_model = LogisticRegression().fit(X_tr_l_scaled, y_tr_l)

# --- VISUALIZATION ---

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### SLR: Body Mass vs Sex")
    y_pred_s = slr_model.predict(X_te_s_scaled)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mean_absolute_error(y_te_s, y_pred_s):.2f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_te_s, y_pred_s)):.2f}")
    m3.metric("R²", f"{r2_score(y_te_s, y_pred_s):.2f}")

    fig1, ax1 = plt.subplots()
    ax1.scatter(df['body_mass_g'], df['sex'], alpha=0.3, color='gray')
    
    # Regression line calculation
    x_range = np.linspace(df['body_mass_g'].min(), df['body_mass_g'].max(), 100).reshape(-1, 1)
    y_line = slr_model.predict(scaler_slr.transform(x_range))
    ax1.plot(x_range, y_line, color="red", linewidth=3)
    ax1.set_xlabel("Body Mass (g)")
    ax1.set_ylabel("Sex (Probability)")
    st.pyplot(fig1)

with col_right:
    st.markdown("### MLR: Feature Correlation")
    y_pred_m = mlr_model.predict(X_te_m_scaled)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mean_absolute_error(y_te_m, y_pred_m):.2f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_te_m, y_pred_m)):.2f}")
    m3.metric("R²", f"{r2_score(y_te_m, y_pred_m):.2f}")

    fig2, ax2 = plt.subplots()
    sns.heatmap(df[features].corr(), annot=True, cmap="Blues", ax=ax2)
    st.pyplot(fig2)

st.divider()

# Logistic Regression Section
st.markdown("### Logistic Regression: Performance Metrics")
y_pred_l = lr_model.predict(X_te_l_scaled)
y_prob_l = lr_model.predict_proba(X_te_l_scaled)[:, 1]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy_score(y_te_l, y_pred_l):.2f}")
m2.metric("Precision", f"{precision_score(y_te_l, y_pred_l):.2f}")
m3.metric("Recall", f"{recall_score(y_te_l, y_pred_l):.2f}")
m4.metric("F1 Score", f"{f1_score(y_te_l, y_pred_l):.2f}")

# Confusion Matrix and ROC Curve
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_te_l, y_pred_l)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

with col_plot2:
    st.write("**ROC Curve**")
    fpr, tpr, _ = roc_curve(y_te_l, y_prob_l)
    auc_val = roc_auc_score(y_te_l, y_prob_l)
    
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()

    st.pyplot(fig_roc)
