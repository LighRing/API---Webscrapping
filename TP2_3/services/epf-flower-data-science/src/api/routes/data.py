# services/epf-flower-data-science/src/api/routes/data.py

import os
from dotenv import load_dotenv
import kaggle
import joblib
from fastapi import APIRouter, HTTPException
from starlette.responses import JSONResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API Kaggle depuis les variables d'environnement
KAGGLE_API_KEY = os.getenv("KAGGLE_API_KEY")

# Vérifier si la clé API a bien été récupérée
if not KAGGLE_API_KEY:
    raise ValueError("Kaggle API key is missing in the environment variables")

# Création de l'instance du sous-router
router = APIRouter()

# Path to the Iris dataset
DATASET_PATH = os.path.join(os.getcwd(), "src", "data", "iris.csv")

# Path to save the trained model
MODEL_PATH = os.path.join(os.getcwd(), "src", "models", "random_forest_model.joblib")

# Exemple de route pour télécharger le dataset
@router.get("/download-dataset")
async def download_dataset():
    """
    Route pour télécharger le dataset Iris depuis Kaggle et le sauvegarder dans le dossier src/data.
    """
    try:
        # Authentification avec la clé API Kaggle
        kaggle.api.authenticate()

        # Télécharger le dataset Iris depuis Kaggle
        kaggle.api.dataset_download_files('uciml/iris', path='src/data', unzip=True)

        return JSONResponse(
            status_code=200,
            content={"message": "Dataset downloaded and saved successfully."}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading dataset: {str(e)}")

# Exemple de route pour charger le dataset
@router.get("/load-dataset")
async def load_dataset():
    """
    Route pour charger le dataset Iris.
    """
    try:
        # Vérifier si le dataset existe
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=404, detail="Dataset not found.")
        
        # Charger le dataset dans un DataFrame pandas
        df = pd.read_csv(DATASET_PATH)

        return JSONResponse(
            status_code=200,
            content={"message": "Dataset loaded successfully.", "data": df.head().to_dict()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

# Exemple de route pour traiter le dataset
@router.get("/process-dataset")
async def process_dataset():
    """
    Route pour traiter le dataset Iris (nettoyage, normalisation, etc.).
    """
    try:
        # Vérifier si le dataset existe
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=404, detail="Dataset not found.")
        
        # Charger le dataset dans un DataFrame pandas
        df = pd.read_csv(DATASET_PATH)

        # Vérifier les valeurs manquantes et les supprimer (si nécessaire)
        if df.isnull().sum().any():
            df = df.dropna()

        # Séparer les caractéristiques et la cible
        X = df.drop(columns=["species"])
        y = df["species"]

        # Mise à l'échelle des caractéristiques avec StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return JSONResponse(
            status_code=200,
            content={"message": "Dataset processed successfully."}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

# Exemple de route pour diviser le dataset
@router.get("/split-dataset")
async def split_dataset():
    """
    Route pour diviser le dataset Iris en ensembles d'entraînement et de test.
    """
    try:
        # Vérifier si le dataset existe
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=404, detail="Dataset not found.")
        
        # Charger le dataset dans un DataFrame pandas
        df = pd.read_csv(DATASET_PATH)

        # Séparer les caractéristiques et la cible
        X = df.drop(columns=["species"])
        y = df["species"]

        # Encoder la colonne cible (species) avec LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Mise à l'échelle des caractéristiques avec StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Diviser les données en ensembles d'entraînement et de test (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        return JSONResponse(
            status_code=200,
            content={"message": "Dataset split into train and test sets."}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error splitting dataset: {str(e)}")

# Route pour entraîner un modèle de classification et sauvegarder le modèle
@router.get("/train-model")
async def train_model():
    """
    Route pour entraîner un modèle de classification (Random Forest) sur le dataset Iris et le sauvegarder.
    """
    try:
        # Vérifier si le dataset existe
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=404, detail="Dataset not found.")
        
        # Charger le dataset dans un DataFrame pandas
        df = pd.read_csv(DATASET_PATH)

        # Vérifier les valeurs manquantes et les supprimer (si nécessaire)
        if df.isnull().sum().any():
            df = df.dropna()

        # Séparer les caractéristiques et la cible
        X = df.drop(columns=["species"])
        y = df["species"]

        # Encoder la colonne cible (species) avec LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Mise à l'échelle des caractéristiques avec StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Diviser les données en ensembles d'entraînement et de test (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        # Initialiser le classificateur RandomForest avec des paramètres par défaut
        rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

        # Entraîner le modèle
        rf.fit(X_train, y_train)

        # Tester le modèle et calculer l'exactitude
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Sauvegarder le modèle entraîné avec joblib
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Créer le dossier src/models s'il n'existe pas
        joblib.dump(rf, MODEL_PATH)

        # Retourner un message de succès et l'exactitude du modèle
        return JSONResponse(
            status_code=200,
            content={
                "message": "Model trained and saved successfully.",
                "accuracy": accuracy,
                "model_path": MODEL_PATH
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
