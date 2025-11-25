# Proyecto Parcial 3 – Machine Learning  
**Autores:** Jorge Enrique Galvis Sáenz & Miguel Lerma  
Universidad Católica – Ingeniería de Sistemas  
2024–2025  
📌 1. Objetivo General

Desarrollar, entrenar y comparar modelos de Machine Learning supervisado (clasificación) y no supervisado (clustering) utilizando datasets reales. Finalmente, integrar los modelos entrenados en una aplicación web interactiva donde cualquier usuario puede realizar predicciones.

📂 2. Datasets usados
A. Dataset Supervisado (Clasificación)

Telco Customer Churn
Fuente: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Target obligatorio: Churn (Yes/No)

Variables utilizadas: 30 columnas finales tras preprocesamiento

B. Dataset No Supervisado (Clustering)

Credit Card Dataset for Clustering
Fuente: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

Se utilizaron únicamente features numéricas

Modelo: K-Means

🤖 3. Modelos Implementados
🔹 Regresión Logística (Supervisado)

Métricas evaluadas:

ROC Curve

AUC

Accuracy

Precision

Recall

F1-Score

🔹 K-Nearest Neighbors – KNN (Supervisado)

Comparado bajo las mismas métricas que la regresión logística

Mismo preprocesamiento y mismas 30 columnas

🔹 K-Means (No Supervisado)

Selección de número de clusters mediante:

Elbow Method

Silhouette Score

Clustering final con k = 4

Interpretación de perfiles de cluster

🧪 4. Notebooks incluidos

Todos los notebooks están ubicados en:

notebooks/

Notebook	Descripción
telco_logistic.ipynb	Preprocesamiento + entrenamiento + evaluación + exportación de modelos Logísticos
telco_knn.ipynb	Entrenamiento y evaluación del modelo KNN bajo las mismas 30 columnas
creditcard_kmeans.ipynb	Clustering K-Means + análisis Elbow + Silhouette + exportación de modelos
📦 5. Modelos Exportados

Los archivos .pkl se encuentran en:

modelos/

Archivo	Descripción
logistic_model.pkl	Modelo de regresión logística entrenado
knn_model.pkl	Modelo KNN entrenado
kmeans_model.pkl	Modelo K-Means entrenado
scaler_telco.pkl	Scaler utilizado para Telco
scaler_cc.pkl	Scaler utilizado para clustering
telco_columns.pkl	Lista de columnas exactas utilizadas en el entrenamiento
🌐 6. Aplicación Web — Streamlit

La app está ubicada en:

app/app.py

Funcionalidades:
Modelo	Función en la Web
Regresión Logística	Predicción de churn + probabilidad
KNN	Predicción de churn usando vecino más cercano
K-Means	Asignación de cluster + descripción interpretada
▶️ 7. Ejecutar la Aplicación Web
Paso 1 — Instalar dependencias

Desde la raíz del proyecto:

pip install -r requirements.txt

Paso 2 — Ejecutar Streamlit
streamlit run app/app.py

La app se abre en tu navegador en:
http://localhost:8501/

🏗️ 8. Estructura del Proyecto
Parcial3-ML/
├── app/
│   └── app.py
├── data/
│   ├── CC GENERAL.csv
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── modelos/
│   ├── logistic_model.pkl
│   ├── knn_model.pkl
│   ├── kmeans_model.pkl
│   ├── scaler_telco.pkl
│   ├── scaler_cc.pkl
│   └── telco_columns.pkl
├── notebooks/
│   ├── telco_logistic.ipynb
│   ├── telco_knn.ipynb
│   └── creditcard_kmeans.ipynb
├── requirements.txt
└── README.md

🧩 9. Cómo reentrenar los modelos

Modificar cualquiera de los notebooks

Ejecutar Run All

Los nuevos .pkl serán generados en /modelos

La app web los cargará automáticamente