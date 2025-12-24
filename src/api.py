from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

# Определяем модель данных для API
class FlightData(BaseModel):
    carrier: str
    dep_hour: int
    distance: float
    weather_delay: float

# Инициализация приложения
app = FastAPI(
    title="Flight Delay Prediction API",
    description="API для предсказания задержки рейса на основе 4 признаков",
    version="2.0"
)

# Глобальные переменные
model = None
feature_names = None
carrier_encoder = None

@app.on_event("startup")
async def load_model():
    """Загрузка модели при старте приложения"""
    global model, feature_names, carrier_encoder
    
    try:
        # Загружаем модель и признаки
        model_data = joblib.load("models/logistic_regression_model.pkl")
        
        if isinstance(model_data, dict) and 'model' in model_data:
            # Новая версия: модель с метаданными
            model = model_data['model']
            feature_names = model_data.get('feature_names', [])
        else:
            # Старая версия: только модель
            model = model_data
            feature_names = ['carrier_encoded', 'dep_hour', 'distance', 'weather_delay']
        
        # Загружаем энкодер для carrier
        try:
            encoders = joblib.load("models/encoders.pkl")
            carrier_encoder = encoders.get('carrier')
        except:
            print("⚠️ Не удалось загрузить энкодер carrier, используем маппинг по умолчанию")
        
        print(f"✅ Модель загружена успешно")
        print(f"   Тип модели: {type(model).__name__}")
        print(f"   Признаки: {feature_names}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        model = None

def encode_carrier(carrier_code: str) -> int:
    """Кодирование кода авиакомпании"""
    # Маппинг популярных авиакомпаний
    carrier_mapping = {
        'AA': 0,  # American Airlines
        'DL': 1,  # Delta
        'UA': 2,  # United
        'WN': 3,  # Southwest
        'B6': 4,  # JetBlue
        'AS': 5,  # Alaska
        'NK': 6,  # Spirit
        'F9': 7,  # Frontier
    }
    
    if carrier_encoder:
        try:
            # Пробуем использовать обученный энкодер
            return int(carrier_encoder.transform([carrier_code])[0])
        except:
            pass
    
    # Используем маппинг по умолчанию
    return carrier_mapping.get(carrier_code.upper(), 0)

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Flight Delay Prediction API",
        "version": "2.0",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": {
                "method": "POST",
                "path": "/predict",
                "description": "Предсказание вероятности задержки рейса",
                "required_fields": ["carrier", "dep_hour", "distance", "weather_delay"]
            },
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "Проверка здоровья API"
            },
            "example": {
                "method": "GET",
                "path": "/example",
                "description": "Пример запроса"
            }
        }
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "features_expected": feature_names,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/example")
async def get_example():
    """Возвращает пример запроса"""
    example_data = {
        "carrier": "AA",
        "dep_hour": 9,
        "distance": 550.0,
        "weather_delay": 0.0
    }
    
    # Показываем предсказание для примера
    if model is not None:
        try:
            df = prepare_features(example_data)
            probability = model.predict_proba(df)[0, 1]
            example_data["predicted_delay_probability"] = round(float(probability), 4)
        except Exception as e:
            example_data["prediction_error"] = str(e)
    
    return {
        "example_request": example_data,
        "curl_example": 'curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d \'{"carrier":"AA","dep_hour":9,"distance":550,"weather_delay":0}\''
    }

def prepare_features(data: Dict[str, Any]) -> pd.DataFrame:
    """Подготовка признаков для модели"""
    
    # Кодируем carrier
    carrier_encoded = encode_carrier(data['carrier'])
    
    # Создаем DataFrame с правильными признаками
    features = {
        'carrier_encoded': [carrier_encoded],
        'dep_hour': [data['dep_hour']],
        'distance': [float(data['distance'])],
        'weather_delay': [float(data['weather_delay'])]
    }
    
    df = pd.DataFrame(features)
    
    # Убеждаемся, что признаки в правильном порядке
    if feature_names:
        df = df[feature_names]
    
    return df

@app.post("/predict")
async def predict(data: FlightData):
    """
    Предсказание вероятности задержки рейса
    
    Пример запроса:
    ```json
    {
        "carrier": "AA",
        "dep_hour": 9,
        "distance": 550,
        "weather_delay": 0.0
    }
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        # Подготавливаем признаки
        df = prepare_features(data.dict())
        
        # Предсказание
        probability = model.predict_proba(df)[0, 1]
        
        return {
            "carrier": data.carrier,
            "dep_hour": data.dep_hour,
            "distance": data.distance,
            "weather_delay": data.weather_delay,
            "delay_probability": round(float(probability), 4),
            "status": "success",
            "message": f"Вероятность задержки рейса {data.carrier} в {data.dep_hour}:00: {probability:.1%}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Дополнительный эндпоинт для батч-предсказаний
@app.post("/predict/batch")
async def predict_batch(data: list[FlightData]):
    """Предсказание для нескольких рейсов"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for item in data:
        try:
            df = prepare_features(item.dict())
            probability = model.predict_proba(df)[0, 1]
            
            results.append({
                **item.dict(),
                "delay_probability": round(float(probability), 4)
            })
        except Exception as e:
            results.append({
                **item.dict(),
                "error": str(e)
            })
    
    return {"predictions": results}