#!/usr/bin/env python3
"""
Скрипт для тестирования API
"""

import requests
import json
import sys
import os
from PIL import Image
import io

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Тестирование Skin Cancer Classification API\n")
    
    # 1. Проверка здоровья
    print("1. Проверка здоровья API...")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"   ✅ Статус: {health_data['status']}")
        print(f"   ✅ Модель загружена: {health_data['model_loaded']}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        print("   💡 Убедитесь, что сервер запущен: python run.py")
        return
    
    # 2. Информация о модели
    print("\n2. Информация о модели...")
    try:
        response = requests.get(f"{base_url}/api/v1/model-info")
        model_info = response.json()
        print(f"   ✅ Классов диагнозов: {model_info.get('total_diagnosis_classes', 'N/A')}")
        print(f"   ✅ Уровней риска: {model_info.get('total_risk_classes', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # 3. Создание тестового изображения
    print("\n3. Создание тестового изображения...")
    try:
        test_image = Image.new('RGB', (300, 200), color='blue')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        print("   ✅ Тестовое изображение создано")
    except Exception as e:
        print(f"   ❌ Ошибка создания изображения: {e}")
        return
    
    # 4. Тестирование предсказания
    print("\n4. Тестирование предсказания...")
    try:
        files = {
            'image': ('test_image.jpg', img_bytes, 'image/jpeg')
        }
        data = {
            'age': 45,
            'sex': 1,      # female
            'localization': 5,  # lower extremity
            'dx_type': 1   # consensus
        }
        
        response = requests.post(f"{base_url}/api/v1/predict", files=files, data=data)
        prediction = response.json()
        
        if prediction.get('success'):
            diagnosis = prediction['diagnosis']
            risk = prediction['risk']
            print(f"   ✅ Диагноз: {diagnosis['name']} ({diagnosis['full_name']})")
            print(f"   ✅ Уверенность: {diagnosis['confidence']:.3f}")
            print(f"   ✅ Уровень риска: {risk['level']} - {risk['name']}")
            print(f"   ✅ Цвет риска: {risk['color']}")
        else:
            print(f"   ❌ Ошибка предсказания: {prediction.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ❌ Ошибка запроса: {e}")
    
    print("\n🎯 Тестирование завершено!")

def test_batch_predictions():
    """Тестирование с разными сценариями"""
    base_url = "http://localhost:8000"
    
    print("\n🧪 Расширенное тестирование с разными сценариями...")
    
    test_cases = [
        {
            "name": "Молодой мужчина с образованием на спине",
            "age": 30, "sex": 0, "localization": 10, "dx_type": 1,
            "color": "red"
        },
        {
            "name": "Женщина среднего возраста с образованием на лице", 
            "age": 55, "sex": 1, "localization": 13, "dx_type": 2,
            "color": "green"
        },
        {
            "name": "Пожилой человек с образованием на волосистой части головы",
            "age": 70, "sex": 0, "localization": 14, "dx_type": 3,
            "color": "blue"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Тест {i}: {test_case['name']}")
        
        try:
            # Создаем тестовое изображение
            test_image = Image.new('RGB', (300, 200), color=test_case['color'])
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            files = {'image': (f'test_{i}.jpg', img_bytes, 'image/jpeg')}
            data = {
                'age': test_case['age'],
                'sex': test_case['sex'],
                'localization': test_case['localization'], 
                'dx_type': test_case['dx_type']
            }
            
            response = requests.post(f"{base_url}/api/v1/predict", files=files, data=data)
            prediction = response.json()
            
            if prediction.get('success'):
                diagnosis = prediction['diagnosis']
                risk = prediction['risk']
                print(f"    ✅ {diagnosis['name']} -> Риск {risk['level']} ({risk['name']})")
            else:
                print(f"    ❌ Ошибка: {prediction.get('error')}")
                
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")

if __name__ == "__main__":
    test_api()
    test_batch_predictions()