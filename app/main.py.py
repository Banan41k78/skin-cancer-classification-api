from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from app.api.endpoints import router as api_router
from app.models.model_manager import SkinCancerModel
from app.utils.image_processor import ImageProcessor
from config.settings import get_settings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Глобальные экземпляры
model_manager = SkinCancerModel()
image_processor = ImageProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Starting Skin Cancer Classification API...")
    
    try:
        success = model_manager.load_model(settings.MODEL_PATH)
        if success:
            logger.info("Model loaded successfully")
        else:
            logger.error("Failed to load model")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Skin Cancer Classification API...")

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Skin Cancer Classification API",
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded,
        "timestamp": "2024-01-15T10:00:00Z"  # В реальном приложении используйте datetime
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )