import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
import os
import sys

# Configuration de l'environnement
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ["HADOOP_HOME"] = r"C:\\hadoop"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Churn Bancaire",
    layout="wide"
)

# Initialisation de Spark
@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("BankChurnPrediction") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return spark

# Chargement du modèle et du scaler
@st.cache_resource
def load_model(_spark):
    from pyspark.ml.feature import StandardScalerModel
    model_path = "C:/Users/abirm/Projects/BankChurnPredict/models/model"
    scaler_path = "C:/Users/abirm/Projects/BankChurnPredict/models/scaler"
    model = RandomForestClassificationModel.load(model_path)
    scaler = StandardScalerModel.load(scaler_path)
    return model, scaler

# Interface utilisateur
st.title(" Prédiction de Churn Bancaire")
st.markdown("---")

# Sidebar pour les informations
st.sidebar.header("📊 À propos")
st.sidebar.info(
    "Cette application prédit la probabilité qu'un client quitte la banque "
    "en utilisant un modèle Random Forest entraîné avec PySpark."
)

# Formulaire de saisie
st.header("Entrez les informations du client")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Score de Crédit", min_value=300, max_value=850, value=650)
    age = st.number_input("Âge", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Ancienneté (années)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Solde du Compte", min_value=0.0, value=50000.0, step=1000.0)

with col2:
    num_products = st.selectbox("Nombre de Produits", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Possède une Carte de Crédit", ["Oui", "Non"])
    is_active = st.selectbox("Membre Actif", ["Oui", "Non"])
    estimated_salary = st.number_input("Salaire Estimé", min_value=0.0, value=50000.0, step=1000.0)

with col3:
    geography = st.selectbox("Géographie", ["France", "Germany", "Spain"])
    gender = st.selectbox("Genre", ["Male", "Female"])

# Conversion des valeurs
has_cr_card_val = 1 if has_cr_card == "Oui" else 0
is_active_val = 1 if is_active == "Oui" else 0

# Mapping géographie et genre (selon votre StringIndexer)
geography_map = {"France": 0.0, "Germany": 1.0, "Spain": 2.0}
gender_map = {"Female": 0.0, "Male": 1.0}

geography_index = geography_map[geography]
gender_index = gender_map[gender]

st.markdown("---")

# Bouton de prédiction
if st.button("🔮 Prédire le Churn", type="primary"):
    with st.spinner("Chargement du modèle et prédiction en cours..."):
        try:
            # Initialisation Spark et chargement du modèle et scaler
            spark = init_spark()
            model_and_scaler = load_model(spark)
            model = model_and_scaler[0]
            scaler_model = model_and_scaler[1]
            
            # Création du DataFrame
            input_data = pd.DataFrame({
                "CreditScore": [credit_score],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "NumOfProducts": [num_products],
                "HasCrCard": [has_cr_card_val],
                "IsActiveMember": [is_active_val],
                "EstimatedSalary": [estimated_salary],
                "Geography_index": [geography_index],
                "Gender_index": [gender_index]
            })
            
            # Conversion en Spark DataFrame
            spark_df = spark.createDataFrame(input_data)
            
            # Assemblage des features
            feature_cols = ["CreditScore", "Age", "Tenure", "Balance",
                           "NumOfProducts", "HasCrCard", "IsActiveMember",
                           "EstimatedSalary", "Geography_index", "Gender_index"]
            
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            spark_df = assembler.transform(spark_df)
            
            # Normalisation avec le scaler sauvegardé
            spark_df = scaler_model.transform(spark_df)
            
            # Prédiction
            prediction = model.transform(spark_df)
            result = prediction.select("prediction", "probability").collect()[0]
            
            # Affichage des résultats
            st.success("✅ Prédiction terminée !")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if result["prediction"] == 1.0:
                    st.error("⚠️ **RISQUE ÉLEVÉ DE CHURN**")
                    st.markdown("Ce client est susceptible de quitter la banque.")
                else:
                    st.success("✅ **FAIBLE RISQUE DE CHURN**")
                    st.markdown("Ce client devrait rester fidèle à la banque.")
            
            with col2:
                prob = result["probability"]
                churn_prob = prob[1] * 100
                stay_prob = prob[0] * 100
                
                st.metric("Probabilité de Churn", f"{churn_prob:.2f}%")
                st.metric("Probabilité de Rester", f"{stay_prob:.2f}%")
            
            # Barre de progression
            st.markdown("### Niveau de Risque")
            st.progress(churn_prob / 100)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
            st.info("Assurez-vous que le modèle est bien sauvegardé au chemin spécifié.")
