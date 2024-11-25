# services/epf-flower-data-science/src/app.py

from fastapi import FastAPI
from starlette.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from src.api.router import router

def get_application() -> FastAPI:
    """
    Function to create and configure the FastAPI application.
    Returns:
        FastAPI application instance.
    """
    application = FastAPI(
        title="EPF Flower Data Science API",
        description="API to serve flower species predictions based on the Iris dataset",
        version="1.0.0",
        redoc_url=None,  # Swagger UI will be used by default
    )

    # Add CORS middleware for handling cross-origin requests
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],
    )

    # Redirect the root endpoint to the Swagger UI documentation
    @application.get("/")
    async def root():
        return RedirectResponse(url="/docs")

    # Include the router for all API endpoints
    application.include_router(router)

    return application
