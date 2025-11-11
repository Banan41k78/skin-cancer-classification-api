#!/usr/bin/env python3
"""
Скрипт для скачивания модели из облачного хранилища
"""

import requests
import os
import sys
from tqdm import tqdm

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings

def download_model():
    """Скачивание модели из облачного хранилища"""
    settings = get_settings()
    
    # URL для скачивания модели (замените на реальный)
    model_urls = [
        "https://example.com/models/skin_cancer_model.h5",
        "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID",
        "https://dropbox.com/s/.../best_model.h5?dl=1"
    ]
    
    print("📥 Скачивание модели...")
    print("⚠️  ВАЖНО: Замените URL в коде на реальную ссылку на вашу модель")
    
    # Создаем директорию если нужно
    os.makedirs(os.path.dirname(settings.absolute_model_path), exist_ok=True)
    
    # Пробуем скачать с первого доступного URL
    for i, model_url in enumerate(model_urls, 1):
        print(f"\nПопытка {i}: {model_url}")
        
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Получаем общий размер файла
            total_size = int(response.headers.get('content-length', 0))
            
            # Скачиваем с прогресс-баром
            with open(settings.absolute_model_path, 'wb') as f, tqdm(
                desc="Скачивание",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            file_size = os.path.getsize(settings.absolute_model_path) / (1024*1024)
            print(f"✅ Модель сохранена: {settings.absolute_model_path} ({file_size:.2f} MB)")
            
            # Проверяем валидность файла
            if file_size < 1:  # Меньше 1MB - вероятно ошибка
                print("❌ Файл слишком маленький, вероятно ошибка скачивания")
                os.remove(settings.absolute_model_path)
                continue
                
            return True
            
        except Exception as e:
            print(f"❌ Ошибка скачивания: {e}")
            continue
    
    print("\n❌ Не удалось скачать модель ни с одного URL")
    print("📋 Альтернативные варианты:")
    print("   1. Поместите модель вручную в models/trained_models/best_model.h5")
    print("   2. Обучите модель заново в ноутбуке")
    print("   3. Используйте Google Drive/Dropbox для обмена моделью")
    
    return False

if __name__ == "__main__":
    download_model()