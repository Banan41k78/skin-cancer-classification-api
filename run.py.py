#!/usr/bin/env python3
"""
Skin Cancer Classification API - Запуск сервера
"""

import uvicorn
import os
import sys
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/api_server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Добавляем корневую директорию в путь для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def check_environment():
    """Проверка окружения и зависимостей"""
    print("Проверка окружения...")
    
    # Проверяем необходимые папки
    required_dirs = ['models/trained_models', 'logs']
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Папка: {dir_path}")
    
    # Проверяем наличие файла модели
    model_path = os.path.join(current_dir, 'models/trained_models/best_model.h5')
    if not os.path.exists(model_path):
        print(f"Файл модели не найден: {model_path}")
        print("Поместите модель в models/trained_models/best_model.h5")
    else:
        file_size = os.path.getsize(model_path) / (1024*1024)
        print(f"Модель найдена: {file_size:.2f} MB")
    
    return True

def main():
    """Основная функция запуска"""
    try:
        from config.settings import get_settings
        settings = get_settings()
    except ImportError as e:
        logger.error(f"Ошибка импорта настроек: {e}")
        print("Не удалось загрузить настройки")
        print("Убедитесь что файл config/settings.py существует")
        return
    except Exception as e:
        logger.error(f"Ошибка загрузки настроек: {e}")
        print("Ошибка загрузки настроек")
        return
    
    print("Запуск Skin Cancer Classification API")
    print("=" * 60)
    print(f"Название: {settings.APP_NAME}")
    print(f"Версия: {settings.APP_VERSION}")
    print(f"Хост: {settings.HOST}")
    print(f"Порт: {settings.PORT}")
    print(f"Режим отладки: {settings.DEBUG}")
    print(f"Путь к модели: {settings.absolute_model_path}")
    print("=" * 60)
    
    # Проверка окружения
    if not check_environment():
        print("Проверка окружения не пройдена")
        return
    
    # Проверяем существование файла модели
    if not os.path.exists(settings.absolute_model_path):
        print(f"ВНИМАНИЕ: Файл модели не найден!")
        print(f"Путь: {settings.absolute_model_path}")
        print("\nРешение:")
        print("   1. Поместите файл модели в models/trained_models/")
        print("   2. Назовите его best_model.h5")
        print("   3. Или измените MODEL_PATH в .env файле")
        print("\nAPI запустится, но предсказания работать не будут!")
    
    print("\nДоступные эндпоинты:")
    print(f"Документация: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"Альтернативная docs: http://{settings.HOST}:{settings.PORT}/redoc")
    print(f"Health check: http://{settings.HOST}:{settings.PORT}/health")
    print(f"Предсказание: http://{settings.HOST}:{settings.PORT}/api/v1/predict")
    print("\nДля остановки сервера нажмите Ctrl+C")
    print("=" * 60)
    
    try:
        # Запуск сервера
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True,
            workers=1
        )
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
        print(f"❌ Ошибка запуска сервера: {e}")

if __name__ == "__main__":
    main()