# services/epf-flower-data-science/src/api/router.py

from fastapi import APIRouter

# Importation des sous-routers
from src.api.routes.hello import router as hello_router
from src.api.routes.data import router as data_router

# Création de l'instance principale de APIRouter
router = APIRouter()

# Inclusion des sous-routers
router.include_router(hello_router, tags=["Hello"])

# Ici, nous incluons les routes de data sans préfixe, donc elles seront directement accessibles sous /download-dataset, /load-dataset, etc.
router.include_router(data_router, tags=["Data"])
