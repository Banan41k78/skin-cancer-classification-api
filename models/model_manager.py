import tensorflow as tf
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
import os

logger = logging.getLogger(__name__)

class SkinCancerModel:
    """
    Main model manager for skin cancer classification
    Based on the notebook: https://colab.research.google.com/drive/1b6MJRtXQFL4hKrmkohUpZIahfz6DTgtR
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.image_shape = (300, 200, 3)
        self.meta_dim = 4  # age, sex, localization, dx_type
        self.is_loaded = False
        
        # Маппинги из вашего ноутбука
        self.dx_type_mapping = {
            'histo': 3,      # Наиболее надежный
            'confocal': 2,   # Высокая надежность  
            'consensus': 1,   # Высокая надежность
            'follow_up': 0    # Менее надежный
        }
        
        self.dx_type_reverse = {v: k for k, v in self.dx_type_mapping.items()}
        
        # Маппинг диагнозов (из dx_danger_mapping)
        self.diagnosis_mapping = {
            0: {"name": "df", "full_name": "Dermatofibroma", "description": "Дерматофиброма - наименее опасное"},
            1: {"name": "vasc", "full_name": "Vascular Lesions", "description": "Сосудистые поражения"},
            2: {"name": "bkl", "full_name": "Benign Keratosis", "description": "Доброкачественный кератоз"},
            3: {"name": "nv", "full_name": "Melanocytic Nevi", "description": "Родинки (требуют наблюдения)"},
            4: {"name": "akiec", "full_name": "Actinic Keratoses", "description": "Актинический кератоз (предраковое)"},
            5: {"name": "bcc", "full_name": "Basal Cell Carcinoma", "description": "Базальноклеточный рак"},
            6: {"name": "mel", "full_name": "Melanoma", "description": "Меланома - наиболее опасное"}
        }
        
        # Маппинг уровней опасности (из danger_mapping_multi)
        self.danger_mapping = {
            'bkl': 0,  # безопасно (доброкачественное)
            'nv': 0,   # безопасно (доброкачественное)
            'df': 0,   # безопасно (доброкачественное) 
            'vasc': 0, # безопасно (доброкачественное)
            'akiec': 1, # умеренная опасность (предраковое)
            'bcc': 2,   # опасно (рак низкой злокачественности)
            'mel': 3    # очень опасно (рак высокой злокачественности)
        }
        
        # Маппинг пола (из sex_mapping)
        self.sex_mapping = {
            'male': 0,
            'female': 1, 
            'unknown': 2
        }
        
        self.sex_reverse = {v: k for k, v in self.sex_mapping.items()}
        
        # Маппинг локализации (из localization_danger_mapping)
        self.localization_mapping = {
            'unknown': 0,           # Неизвестная локализация
            'genital': 1,           # Гениталии (редко, но важно)
            'acral': 2,             # Актральные зоны (ладони, стопы) - риск меланомы
            'foot': 3,              # Стопы
            'hand': 4,              # Кисти
            'lower extremity': 5,   # Нижние конечности
            'upper extremity': 6,   # Верхние конечности  
            'abdomen': 7,           # Живот
            'chest': 8,             # Грудь
            'trunk': 9,             # Туловище
            'back': 10,             # Спина
            'neck': 11,             # Шея
            'ear': 12,              # Уши
            'face': 13,             # Лицо
            'scalp': 14             # Волосистая часть головы (высокий риск)
        }
        
        self.localization_reverse = {v: k for k, v in self.localization_mapping.items()}
        
        # Система оценки риска (0-3)
        self.risk_classes = {
            0: {
                "name": "Всё хорошо",
                "description": "Доброкачественное образование, нет признаков злокачественности",
                "recommendation": "Плановое наблюдение не требуется. Рекомендуется ежегодный осмотр у дерматолога.",
                "color": "green",
                "urgency": "very_low"
            },
            1: {
                "name": "Требует наблюдения", 
                "description": "Предраковое состояние или образование, требующее наблюдения",
                "recommendation": "Рекомендуется консультация дерматолога в плановом порядке (в течение 1-2 месяцев)",
                "color": "yellow",
                "urgency": "low"
            },
            2: {
                "name": "Опасно",
                "description": "Рак низкой злокачественности, требуется лечение",
                "recommendation": "Срочная консультация онколога в течение 2-4 недель. Требуется биопсия.",
                "color": "orange", 
                "urgency": "medium"
            },
            3: {
                "name": "Очень опасно",
                "description": "Рак высокой злокачественности (меланома), требуется немедленное лечение",
                "recommendation": "НЕМЕДЛЕННОЕ обращение к онкологу! Требуется срочная биопсия и лечение.",
                "color": "red",
                "urgency": "high"
            }
        }
    
    def load_model(self, model_path: str = 'model/best_multimodal_model.h5') -> bool:
        """Load model from file"""
        try:
            if model_path:
                self.model_path = model_path
            
            if not self.model_path or not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"Model loaded successfully from: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model inference"""
        # Resize to model input size
        image = image.resize((self.image_shape[1], self.image_shape[0]))
        image_array = np.array(image)
        
        # Convert to RGB if needed
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        # Normalize to [0, 1]
        image_array = image_array.astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def preprocess_metadata(self, age: float, sex: float, localization: float, dx_type: float) -> np.ndarray:
        """Preprocess metadata for model inference using exact mappings from notebook"""
        metadata = np.array([age, sex, localization, dx_type], dtype='float32')
        return np.expand_dims(metadata, axis=0)
    
    def _get_diagnosis_info(self, diagnosis_class: int) -> Dict:
        """Get diagnosis information based on predicted class"""
        diagnosis_info = self.diagnosis_mapping.get(diagnosis_class, self.diagnosis_mapping[0])
        diagnosis_name = diagnosis_info["name"]
        
        # Get risk level from danger mapping
        risk_level = self.danger_mapping.get(diagnosis_name, 0)
        risk_info = self.risk_classes[risk_level]
        
        return {
            "diagnosis": diagnosis_info,
            "risk": {
                "level": risk_level,
                **risk_info
            }
        }
    
    def predict(self, image: Image.Image, metadata: List[float]) -> Dict:
        """Make prediction for single image"""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess inputs
            processed_image = self.preprocess_image(image)
            processed_metadata = self.preprocess_metadata(*metadata)
            
            # Make prediction (7 classes as in notebook)
            predictions = self.model.predict(
                [processed_image, processed_metadata], 
                verbose=0
            )
            diagnosis_class = int(np.argmax(predictions[0]))
            diagnosis_confidence = float(np.max(predictions[0]))
            
            # Get diagnosis and risk info
            info = self._get_diagnosis_info(diagnosis_class)
            diagnosis_info = info["diagnosis"]
            risk_info = info["risk"]
            
            # Diagnosis probabilities for all classes
            diagnosis_probabilities = {
                str(i): {
                    "diagnosis": self.diagnosis_mapping[i]["name"],
                    "full_name": self.diagnosis_mapping[i]["full_name"],
                    "description": self.diagnosis_mapping[i]["description"],
                    "probability": float(predictions[0][i]),
                    "risk_level": self.danger_mapping[self.diagnosis_mapping[i]["name"]]
                } for i in range(len(self.diagnosis_mapping))
            }
            
            # Aggregated risk probabilities
            risk_probabilities = {
                str(i): {
                    "probability": 0.0,
                    "risk_info": self.risk_classes[i]
                } for i in range(len(self.risk_classes))
            }
            
            for i, prob in enumerate(predictions[0]):
                diagnosis_name = self.diagnosis_mapping[i]["name"]
                risk_level = self.danger_mapping.get(diagnosis_name, 0)
                risk_probabilities[str(risk_level)]["probability"] += float(prob)
            
            # Process metadata for response
            age, sex_code, localization_code, dx_type_code = metadata
            
            sex_name = self.sex_reverse.get(int(sex_code), "unknown")
            localization_name = self.localization_reverse.get(int(localization_code), "unknown")
            dx_type_name = self.dx_type_reverse.get(int(dx_type_code), "consensus")
            
            return {
                "success": True,
                "diagnosis": {
                    "class": diagnosis_class,
                    "name": diagnosis_info["name"],
                    "full_name": diagnosis_info["full_name"],
                    "description": diagnosis_info["description"],
                    "confidence": diagnosis_confidence
                },
                "risk": {
                    "level": risk_info["level"],
                    "name": risk_info["name"],
                    "description": risk_info["description"],
                    "recommendation": risk_info["recommendation"],
                    "color": risk_info["color"],
                    "urgency": risk_info["urgency"],
                    "confidence": risk_probabilities[str(risk_info["level"])]["probability"]
                },
                "metadata": {
                    "age": age,
                    "sex": {
                        "code": sex_code,
                        "name": sex_name
                    },
                    "localization": {
                        "code": localization_code,
                        "name": localization_name,
                        "description": self._get_localization_description(localization_name)
                    },
                    "dx_type": {
                        "code": dx_type_code,
                        "name": dx_type_name,
                        "description": self._get_dx_type_description(dx_type_name)
                    }
                },
                "probabilities": {
                    "diagnosis": diagnosis_probabilities,
                    "risk": risk_probabilities
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_dx_type_description(self, dx_type: str) -> str:
        """Get description for diagnosis type"""
        descriptions = {
            'histo': "Гистологическое подтверждение (наиболее надежный метод)",
            'confocal': "Confocal микроскопия (высокая надежность)",
            'consensus': "Консенсус экспертов (высокая надежность)", 
            'follow_up': "Наблюдение в динамике (менее надежный метод)"
        }
        return descriptions.get(dx_type, "Неизвестный метод диагностики")
    
    def _get_localization_description(self, localization: str) -> str:
        """Get description for localization"""
        descriptions = {
            'unknown': "Неизвестная локализация",
            'genital': "Гениталии (редко, но важно)",
            'acral': "Актральные зоны (ладони, стопы) - риск меланомы",
            'foot': "Стопы",
            'hand': "Кисти",
            'lower extremity': "Нижние конечности",
            'upper extremity': "Верхние конечности",  
            'abdomen': "Живот",
            'chest': "Грудь",
            'trunk': "Туловище",
            'back': "Спина",
            'neck': "Шея",
            'ear': "Уши",
            'face': "Лицо",
            'scalp': "Волосистая часть головы (высокий риск)"
        }
        return descriptions.get(localization, "Неизвестная локализация")
    
    def validate_metadata(self, age: float, sex: float, localization: float, dx_type: float) -> Tuple[bool, str]:
        """Validate metadata inputs using exact ranges from notebook"""
        if age < 0 or age > 120:
            return False, "Возраст должен быть от 0 до 120 лет"
        
        # Sex validation (0, 1, 2 as in sex_mapping)
        if sex not in [0, 1, 2]:
            return False, "Пол должен быть 0 (male), 1 (female) или 2 (unknown)"
        
        # Localization validation (0-14 as in localization_danger_mapping)
        if localization < 0 or localization > 14:
            return False, "Локализация должна быть в диапазоне от 0 до 14"
        
        # Diagnosis type validation (0-3 as in dx_type_final_mapping)
        if dx_type not in [0, 1, 2, 3]:
            return False, "Тип диагностики должен быть 0 (follow_up), 1 (consensus), 2 (confocal) или 3 (histo)"
        
        return True, "OK"
    
    def get_sex_options(self) -> Dict:
        """Get available sex options"""
        return {
            name: {
                "value": value,
                "description": "Мужской" if name == "male" else "Женский" if name == "female" else "Неизвестно"
            }
            for name, value in self.sex_mapping.items()
        }
    
    def get_localization_options(self) -> Dict:
        """Get available localization options"""
        return {
            name: {
                "value": value,
                "description": self._get_localization_description(name)
            }
            for name, value in self.localization_mapping.items()
        }
    
    def get_dx_type_options(self) -> Dict:
        """Get available diagnosis type options"""
        return {
            name: {
                "value": value,
                "description": self._get_dx_type_description(name)
            }
            for name, value in self.dx_type_mapping.items()
        }
    
    def get_diagnosis_classes(self) -> Dict:
        """Get diagnosis classes information"""
        return self.diagnosis_mapping
    
    def get_risk_classes(self) -> Dict:
        """Get risk classes information"""
        return self.risk_classes
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "image_shape": self.image_shape,
            "meta_dim": self.meta_dim,
            "sex_options": self.get_sex_options(),
            "localization_options": self.get_localization_options(),
            "dx_type_options": self.get_dx_type_options(),
            "diagnosis_classes": self.get_diagnosis_classes(),
            "risk_classes": self.get_risk_classes(),
            "total_diagnosis_classes": len(self.diagnosis_mapping),
            "total_risk_classes": len(self.risk_classes)
        }