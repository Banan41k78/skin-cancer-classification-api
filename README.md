# skin-cancer-classification-api
A deep learning API for skin cancer classification and risk assessment using multimodal data (images + clinical metadata). The model predicts 4 types of skin lesions and provides risk levels for clinical decision support.
 ## train
 https://colab.research.google.com/drive/1b6MJRtXQFL4hKrmkohUpZIahfz6DTgtR?usp=sharing
 ## Features
- **Multimodal Analysis**: Combines image data with clinical metadata
- - **4-Level Risk Assessment**: Provides clinical risk stratification
- **RESTful API**: Easy integration with healthcare systems
- **Docker Support**: Containerized deployment
- **Comprehensive Documentation**: OpenAPI/Swagger documentation
## dx_type
- **0 - follow_up**: Наблюдение в динамике (менее надежный метод)
- **1 - consensus**: Консенсус экспертов (высокая надежность)  
- **2 - confocal**: Confocal микроскопия (высокая надежность)
- **3 - histo**: Гистологическое подтверждение (наиболее надежный метод)

## Метаданные для предсказания
- `age`: Возраст пациента (0-120)
- `sex`: Пол (0-женский, 1-мужской, 2 - неизвестный)
- `localization`: Локализация образования (кодированная)
- `dx_type`: Тип диагностики (0-3)

## dx
'df': 0,      # Дерматофиброма - наименее опасное
'vasc': 1,    # Сосудистые поражения
'bkl': 2,     # Доброкачественный кератоз
'nv': 3,      # Родинки (требуют наблюдения)
'akiec': 4,   # Актинический кератоз (предраковое)
'bcc': 5,     # Базальноклеточный рак
'mel': 6      # Меланома - наиболее опасное

## target 
'bkl': 0,  # безопасно (доброкачественное)
'nv': 0,   # безопасно (доброкачественное)
'df': 0,   # безопасно (доброкачественное) 
'vasc': 0, # безопасно (доброкачественное)
'akiec': 1, # умеренная опасность (предраковое)
'bcc': 2,   # опасно (рак низкой злокачественности)
'mel': 3    # очень опасно (рак высокой злокачественности)
