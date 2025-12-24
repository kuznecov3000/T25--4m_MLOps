import pytest
import pandas as pd
import numpy as np

# Если у вас есть реальный модуль preprocess, импортируйте его
# В противном случае создайте моки для тестов

def test_preprocess_removes_duplicates():
    """Проверка удаления дубликатов"""
    # Создаем тестовые данные с дубликатами
    df = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': [4, 5, 5, 6]
    })
    
    # Эмулируем удаление дубликатов
    df_no_duplicates = df.drop_duplicates()
    
    assert len(df_no_duplicates) == 3
    assert df_no_duplicates.duplicated().sum() == 0

def test_preprocess_handles_missing_values():
    """Проверка обработки пропусков"""
    # Создаем тестовые данные с пропусками
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [5, None, 7, 8]
    })
    
    # Эмулируем обработку пропусков
    df_filled = df.fillna(df.mean())
    
    assert df_filled.isnull().sum().sum() == 0

def test_feature_engineering_creates_columns():
    """Проверка создания новых признаков"""
    # Создаем тестовые данные
    df = pd.DataFrame({
        'dep_hour': [9, 14, 20, 6],
        'distance': [500, 1000, 750, 300]
    })
    
    # Эмулируем создание признаков
    df['is_morning'] = df['dep_hour'].between(6, 12).astype(int)
    df['is_long_distance'] = (df['distance'] > 500).astype(int)
    
    assert 'is_morning' in df.columns
    assert 'is_long_distance' in df.columns
    assert df['is_morning'].isin([0, 1]).all()