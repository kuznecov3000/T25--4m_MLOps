import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import yaml
import argparse
import os

def load_config(config_path='params.yaml'):
    """Загрузка конфигурации"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def preprocess_flight_data(raw_path, processed_path, config):
    """Основная функция предобработки данных"""
    print(f"Загружаем данные из {raw_path}")
    df = pd.read_csv(raw_path)
    
    print(f"Исходный размер данных: {df.shape}")
    print(f"Столбцы: {df.columns.tolist()}")
    
    # 1. Очистка данных
    df = clean_data(df, config['preprocess'])
    
    # 2. Feature engineering (используем ТОЛЬКО нужные признаки для API)
    df = create_features(df, config['preprocess'])
    
    # 3. СОЗДАЕМ ЦЕЛЕВУЮ ПЕРЕМЕННУЮ - КРИТИЧЕСКИ ВАЖНО!
    df = create_target_variable(df, config['preprocess'])
    
    # 4. Кодирование категориальных признаков
    df, encoders = encode_categorical(df, config['preprocess'])
    
    print(f"Финальный размер данных: {df.shape}")
    print(f"Целевая переменная распределение:\n{df['is_delayed'].value_counts()}")
    
    # Сохранение обработанных данных
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    
    # Сохранение препроцессоров
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/encoders.pkl')
    
    return df

def clean_data(df, config):
    """Очистка данных"""
    # Удаляем дубликаты
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    print(f"Удалено дубликатов: {initial_shape - df.shape[0]}")
    
    # Оставляем только нужные колонки для API
    required_cols = ['carrier', 'scheduled_dep_time', 'distance', 'weather_delay', 
                     'arr_delay', 'dep_delay']
    
    # Проверяем, какие колонки есть в данных
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols].copy()
    
    # Обработка пропусков
    if config['handle_missing']:
        df = df.dropna()
        print(f"Размер после удаления пропусков: {df.shape}")
    
    return df

def create_features(df, config):
    """Создание новых признаков для API"""
    # 1. Преобразуем время вылета
    if 'scheduled_dep_time' in df.columns:
        df['dep_hour'] = df['scheduled_dep_time'] // 100
        df['dep_minute'] = df['scheduled_dep_time'] % 100
    else:
        # Если нет времени, создаем случайные
        df['dep_hour'] = np.random.randint(0, 24, len(df))
        df['dep_minute'] = np.random.randint(0, 60, len(df))
    
    # 2. Кодируем авиакомпанию (carrier)
    if 'carrier' not in df.columns:
        df['carrier'] = 'AA'  # Значение по умолчанию
    
    # 3. Проверяем наличие weather_delay
    if 'weather_delay' not in df.columns:
        df['weather_delay'] = 0.0
    
    # 4. Проверяем distance
    if 'distance' not in df.columns:
        df['distance'] = 500.0  # Среднее значение
    
    print(f"Созданные признаки: {df.columns.tolist()}")
    return df

def create_target_variable(df, config):
    """Создание целевой переменной is_delayed"""
    # Используем arrival delay или departure delay для определения задержки
    if 'arr_delay' in df.columns:
        # Рейс считается задержанным, если задержка прибытия > 15 минут
        df['is_delayed'] = (df['arr_delay'] > 15).astype(int)
    elif 'dep_delay' in df.columns:
        # Или используем задержку вылета
        df['is_delayed'] = (df['dep_delay'] > 15).astype(int)
    else:
        # Если нет данных о задержках, создаем случайную целевую переменную
        print("⚠️ ВНИМАНИЕ: Нет данных о задержках, создаю случайную целевую переменную")
        df['is_delayed'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
    
    print(f"Целевая переменная создана:")
    print(f"  0 (не задержан): {sum(df['is_delayed'] == 0)}")
    print(f"  1 (задержан): {sum(df['is_delayed'] == 1)}")
    
    return df

def encode_categorical(df, config):
    """Кодирование категориальных признаков"""
    encoders = {}
    
    # Кодируем только carrier (остальные признаки уже числовые)
    if 'carrier' in df.columns:
        encoder = LabelEncoder()
        df['carrier_encoded'] = encoder.fit_transform(df['carrier'])
        encoders['carrier'] = encoder
    
    return df, encoders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Config file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    preprocess_flight_data(
        raw_path=config['data']['raw'],
        processed_path=config['data']['processed'],
        config=config
    )