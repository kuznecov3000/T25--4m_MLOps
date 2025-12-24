import pytest
from fastapi.testclient import TestClient
import sys
import os

# Добавляем путь к src в sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import app

# Мок для модели, чтобы тесты проходили без реальной модели
class MockModel:
    def predict_proba(self, X):
        # Возвращаем предсказания для тестов
        import numpy as np
        n_samples = X.shape[0]
        return np.array([[0.2, 0.8]] * n_samples)

@pytest.fixture(autouse=True)
def mock_model():
    """Автоматически мокаем модель для всех тестов"""
    import api
    
    # Сохраняем оригинальную модель
    original_model = api.model
    
    # Устанавливаем мок-модель
    api.model = MockModel()
    
    # Также устанавливаем feature_names
    api.feature_names = ['carrier_encoded', 'dep_hour', 'distance', 'weather_delay']
    
    yield
    
    # Восстанавливаем оригинальную модель
    api.model = original_model

@pytest.fixture
def client():
    """Фикстура для тестирования FastAPI"""
    return TestClient(app)

@pytest.fixture
def sample_flight_data():
    """Пример данных о рейсе"""
    return {
        "carrier": "AA",
        "dep_hour": 9,
        "distance": 550,
        "weather_delay": 0.0
    }