import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# CARGA DE MODELOS Y SCALERS
# ================================
logistic_model = joblib.load("modelos/logistic_model.pkl")
knn_model = joblib.load("modelos/knn_model.pkl")
kmeans_model = joblib.load("modelos/kmeans_model.pkl")

scaler_telco = joblib.load("modelos/scaler_telco.pkl")
scaler_cc = joblib.load("modelos/scaler_cc.pkl")

telco_columns = joblib.load("modelos/telco_columns.pkl")
cc_columns = joblib.load("modelos/cc_columns.pkl")   # Nuevo


# ================================
# FUNCIONES DE PROCESAMIENTO
# ================================
def preprocess_telco(input_dict):
    df_input = pd.DataFrame([input_dict])
    df_input = pd.get_dummies(df_input, drop_first=True)
    df_input = df_input.reindex(columns=telco_columns, fill_value=0)
    df_scaled = scaler_telco.transform(df_input)
    return df_scaled


def preprocess_creditcard(balance, purchases, cash, credit_limit, payments):
    df_input = pd.DataFrame(columns=cc_columns)
    df_input.loc[0] = 0

    df_input.loc[0, "BALANCE"] = balance
    df_input.loc[0, "PURCHASES"] = purchases
    df_input.loc[0, "CASH_ADVANCE"] = cash
    df_input.loc[0, "CREDIT_LIMIT"] = credit_limit
    df_input.loc[0, "PAYMENTS"] = payments

    df_scaled = scaler_cc.transform(df_input)
    return df_scaled


def describe_cluster(c):
    if c == 0:
        return "Cluster 0: Clientes de bajo uso, comportamiento estable."
    if c == 1:
        return "Cluster 1: Clientes premium, alto gasto y muy activos."
    if c == 2:
        return "Cluster 2: Clientes con alto uso de adelantos en efectivo (riesgo alto)."
    if c == 3:
        return "Cluster 3: Clientes casi inactivos, bajo valor."
    return "Descripción no disponible."


# ================================
# ESTILOS VISUALES
# ================================
st.set_page_config(page_title="Parcial 3 ML", layout="centered")

st.markdown("""
    <style>
        h1 { text-align: center; font-size: 42px; color: #1E88E5; }
        .title { text-align: center; font-size: 22px; color: #555; }
        .section { margin-top: 35px; padding: 20px; border-radius: 12px; background-color: #f8f9fa; }
        .footer { text-align:center; margin-top:50px; font-size: 14px; color:#777; }
    </style>
""", unsafe_allow_html=True)


# ================================
# TÍTULO PRINCIPAL
# ================================
st.markdown("<h1>Aplicación Web — Modelos de Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<p class='title'>Proyecto Académico — Clasificación y Clustering</p>", unsafe_allow_html=True)

st.sidebar.title("Navegación")
menu = st.sidebar.radio("Selecciona un modelo:", ["Regresión Logística", "KNN", "K-Means (Clustering)"])


# ================================
# SECCIÓN — REGRESIÓN LOGÍSTICA
# ================================
if menu == "Regresión Logística":

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Predicción de Churn — Regresión Logística")

    tenure = st.number_input("Tenure (Meses)", min_value=0, max_value=200)
    monthly = st.number_input("Monthly Charges", min_value=0.0)
    total = st.number_input("Total Charges", min_value=0.0)

    if st.button("Predecir Churn"):
        input_dict = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        X_scaled = preprocess_telco(input_dict)
        prob = logistic_model.predict_proba(X_scaled)[0][1]
        pred = logistic_model.predict(X_scaled)[0]

        st.success(f"Probabilidad de Churn: {prob:.4f}")
        st.write("Predicción final:", "Yes" if pred == 1 else "No")

    st.markdown("</div>", unsafe_allow_html=True)


# ================================
# SECCIÓN — KNN
# ================================
if menu == "KNN":

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Predicción de Churn — Modelo KNN")

    tenure = st.number_input("Tenure (Meses)", min_value=0, max_value=200)
    monthly = st.number_input("Monthly Charges", min_value=0.0)
    total = st.number_input("Total Charges", min_value=0.0)

    if st.button("Predecir con KNN"):
        input_dict = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        X_scaled = preprocess_telco(input_dict)
        pred = knn_model.predict(X_scaled)[0]

        st.success("Predicción final: " + ("Yes" if pred == 1 else "No"))

    st.markdown("</div>", unsafe_allow_html=True)


# ================================
# SECCIÓN — K-MEANS
# ================================
if menu == "K-Means (Clustering)":

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Asignación de Cluster – Tarjetas de Crédito")

    balance = st.number_input("Balance")
    purchases = st.number_input("Purchases")
    cash = st.number_input("Cash Advance")
    credit_limit = st.number_input("Credit Limit")
    payments = st.number_input("Payments")

    if st.button("Asignar Cluster"):
        X_scaled = preprocess_creditcard(balance, purchases, cash, credit_limit, payments)
        cluster = kmeans_model.predict(X_scaled)[0]

        st.success(f"Cluster asignado: {cluster}")
        st.info(describe_cluster(cluster))

    st.markdown("</div>", unsafe_allow_html=True)


# ================================
# FOOTER
# ================================
st.markdown("<p class='footer'>Proyecto académico desarrollado por Jorge Galvis y Miguel Lerma</p>", unsafe_allow_html=True)
