from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Основные настройки приложения
    APP_NAME: str = "Skin Cancer Classification API"
    APP_DESCRIPTION: str = "API для классификации кожных заболеваний по изображениям и метаданным"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Настройки модели
    MODEL_PATH: str = "models/trained_models/best_model.h5"
    
    # Настройки сервера
    HOST: str = "0.0.0.0"  # 0.0.0.0 - доступ с любых адресов
    PORT: int = 8000
    
    # CORS (Cross-Origin Resource Sharing)
    ALLOWED_ORIGINS: List[str] = ["*"]  # Разрешить все домены
    
    # Логирование
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # Получение абсолютного пути к модели
    @property
    def absolute_model_path(self) -> str:
        """Возвращает абсолютный путь к файлу модели"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, self.MODEL_PATH)
    
    class Config:
        # Загружать переменные из .env файла
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """Функция для получения настроек"""
    return Settings()