#!/usr/bin/env python3
"""
Скрипт для проверки загрузки модели
"""

import os
import sys
import logging

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.model_manager import SkinCancerModel
from config.settings import get_settings

def check_model():
    """Проверка загрузки и работы модели"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    settings = get_settings()
    
    print("🔍 Проверка модели...")
    print(f"📁 Путь к модели: {settings.absolute_model_path}")
    print(f"📁 Абсолютный путь: {os.path.abspath(settings.absolute_model_path)}")
    
    # Проверяем существование файла
    if not os.path.exists(settings.absolute_model_path):
        print("❌ Файл модели не найден!")
        print("📋 Что делать:")
        print("   1. Поместите файл модели в models/trained_models/")
        print("   2. Назовите его best_model.h5")
        print("   3. Или укажите правильный путь в .env файле")
        
        # Показываем содержимое папки models/
        models_dir = os.path.dirname(settings.absolute_model_path)
        if os.path.exists(models_dir):
            print(f"\n📁 Содержимое {models_dir}:")
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path) / (1024*1024)
                    print(f"   📄 {item} ({size:.2f} MB)")
                else:
                    print(f"   📁 {item}/")
        else:
            print(f"❌ Папка {models_dir} не существует!")
        
        return False
    
    file_size = os.path.getsize(settings.absolute_model_path) / (1024*1024)
    print(f"✅ Файл найден! Размер: {file_size:.2f} MB")
    
    # Пробуем загрузить модель
    print("🔄 Загрузка модели...")
    model = SkinCancerModel()
    
    if model.load_model(settings.absolute_model_path):
        print("🎉 Модель успешно загружена!")
        
        # Показываем информацию о модели
        info = model.get_model_info()
        print(f"📊 Информация о модели:")
        print(f"   - Классы диагнозов: {info['total_diagnosis_classes']}")
        print(f"   - Уровни риска: {info['total_risk_classes']}")
        print(f"   - Размер изображения: {info['image_shape']}")
        print(f"   - Мета-признаки: {info['meta_dim']}")
        
        # Тестовое предсказание
        print("\n🧪 Тестовое предсказание...")
        from PIL import Image
        test_image = Image.new('RGB', (300, 200), color='red')
        test_metadata = [45, 1, 5, 1]  # age, sex, localization, dx_type
        
        result = model.predict(test_image, test_metadata)
        if result["success"]:
            print("✅ Тестовое предсказание успешно!")
            print(f"   Диагноз: {result['diagnosis']['name']}")
            print(f"   Риск: {result['risk']['name']}")
        else:
            print(f"❌ Ошибка предсказания: {result.get('error')}")
        
        return True
    else:
        print("❌ Не удалось загрузить модель")
        return False

if __name__ == "__main__":
    check_model()