# services/epf-flower-data-science/src/api/routes/data.py

import os
import joblib
import logging
import traceback
from fastapi import APIRouter, HTTPException
from starlette.responses import JSONResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
import requests

# Initialiser le logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Création de l'instance du sous-router
router = APIRouter()

# Définir le chemin relatif pour le dataset et le modèle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Obtient le répertoire de 'data.py'
DATASET_PATH = os.path.join(BASE_DIR, "..", "..", "data", "Iris.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "random_forest_model.joblib")

logger.debug(f"Dataset path: {DATASET_PATH}")  # Ajout d'un log pour vérifier le chemin du dataset
logger.debug(f"Model path: {MODEL_PATH}")      # Ajout d'un log pour vérifier le chemin du modèle

# Route pour télécharger le dataset
@router.get("/download-dataset")
async def download_dataset():
    """
    Route pour télécharger le dataset Iris depuis Kaggle ou une autre source.
    """
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
        logger.debug("Downloading dataset from URL.")
        
        response = requests.get(url)
        response.raise_for_status()  # Vérifie si la requête a échoué
        
        # Sauvegarder le fichier dans le dossier data
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)  # Créer le dossier data s'il n'existe pas
        with open(DATASET_PATH, "wb") as f:
            f.write(response.content)
        
        logger.debug(f"Dataset downloaded and saved at {DATASET_PATH}")
        
        return {"message": f"Dataset downloaded successfully and saved at {DATASET_PATH}"}
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error downloading dataset: {str(e)}")


# Route pour charger le dataset
@router.get("/load-dataset")
async def load_dataset():
    """
    Route pour charger le dataset Iris.
    """
    try:
        # Vérifier si le dataset existe
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=404, detail=f"Dataset not found at {DATASET_PATH}")
        
        # Charger le dataset dans un DataFrame pandas
        df = pd.read_csv(DATASET_PATH)
        logger.debug(f"Dataset loaded successfully from {DATASET_PATH}")
        
        return {"message": "Dataset loaded successfully."}
    
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=404, detail=f"File not found: {fnf_error}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")


# Route pour traiter le dataset (nettoyage, normalisation, etc.)
@router.get("/process-dataset")
async def process_dataset():
    """
    Route pour traiter le dataset Iris (nettoyage, normalisation, etc.).
    """
    try:
        # Vérifier si le dataset existe
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=404, detail=f"Dataset not found at {DATASET_PATH}")
        
        # Charger le dataset dans un DataFrame pandas
        df = pd.read_csv(DATASET_PATH)
        
        # Vérifier les valeurs manquantes et les supprimer (si nécessaire)
        if df.isnull().sum().any():
            df = df.dropna()  # Suppression des lignes avec des valeurs manquantes
        
        # Mise à l'échelle des caractéristiques avec StandardScaler
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df.drop(columns=["Species"]))
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns[:-1])  # Remplacer les valeurs transformées dans le DataFrame
        
        logger.debug("Dataset processed successfully (missing values removed and features scaled).")
        
        return {"message": "Dataset processed successfully."}
    
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")


# Route pour diviser le dataset en ensembles d'entraînement et de test
@router.get("/split-dataset")
async def split_dataset():
    """
    Route pour diviser le dataset Iris en ensembles d'entraînement et de test.
    """
    try:
        # Vérifier si le dataset existe
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=404, detail=f"Dataset not found at {DATASET_PATH}")
        
        # Charger le dataset dans un DataFrame pandas
        df = pd.read_csv(DATASET_PATH)
        
        # Séparer les caractéristiques et la cible
        X = df.drop(columns=["Species"])
        y = df["Species"]
        
        # Diviser les données en ensembles d'entraînement et de test (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.debug(f"Dataset split into train and test sets.")
        
        return {"message": "Dataset split into train and test sets."}
    
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error splitting dataset: {str(e)}")


# Route pour entraîner un modèle de classification et sauvegarder le modèle
@router.get("/train-model")
async def train_model():
    """
    Route pour entraîner un modèle de classification (Random Forest) sur le dataset Iris et le sauvegarder.
    """
    try:
        logger.debug("Starting model training...")

        # Vérifier si le dataset existe
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=404, detail=f"Dataset not found at {DATASET_PATH}")
        
        logger.debug(f"Dataset found at {DATASET_PATH}")
        
        # Charger le dataset dans un DataFrame pandas
        df = pd.read_csv(DATASET_PATH)
        logger.debug("Dataset loaded successfully.")

        # Vérifier les valeurs manquantes et les supprimer (si nécessaire)
        if df.isnull().sum().any():
            raise ValueError("Dataset contains missing values. Please clean the dataset.")
        
        logger.debug("No missing values found in the dataset.")

        # Séparer les caractéristiques et la cible
        X = df.drop(columns=["Species"])
        y = df["Species"]

        # Encoder la colonne cible (species) avec LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        logger.debug("Target variable encoded.")

        # Mise à l'échelle des caractéristiques avec StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.debug("Feature scaling completed.")

        # Diviser les données en ensembles d'entraînement et de test (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        logger.debug("Dataset split into train and test sets.")

        # Initialiser le classificateur RandomForest avec des paramètres par défaut
        rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

        # Entraîner le modèle
        rf.fit(X_train, y_train)
        logger.debug("Model training completed.")

        # Tester le modèle et calculer l'exactitude
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.debug(f"Model accuracy: {accuracy}")

        # Sauvegarder le modèle entraîné avec joblib
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Créer le dossier src/models s'il n'existe pas
        joblib.dump(rf, MODEL_PATH)
        logger.debug(f"Model saved at {MODEL_PATH}")

        # Retourner un message de succès et l'exactitude du modèle
        return JSONResponse(
            status_code=200,
            content={
                "message": "Model trained and saved successfully.",
                "accuracy": accuracy,
                "model_path": MODEL_PATH
            }
        )

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=404, detail=f"File not found: {fnf_error}")
    except ValueError as ve_error:
        logger.error(f"Value error: {ve_error}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Value error: {ve_error}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
