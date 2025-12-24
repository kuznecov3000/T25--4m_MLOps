import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
import yaml
import argparse
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path='params.yaml'):
    """Загрузка конфигурации"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(config):
    """Загрузка подготовленных данных"""
    split_dir = config['data']['split_dir']
    
    X_train = pd.read_csv(f'{split_dir}/X_train.csv')
    X_test = pd.read_csv(f'{split_dir}/X_test.csv')
    y_train = pd.read_csv(f'{split_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{split_dir}/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def prepare_features_for_api(X_train, X_test):
    """Подготавливаем признаки для API: carrier, dep_hour, distance, weather_delay"""
    
    print("Подготовка признаков для API...")
    print(f"Доступные признаки: {X_train.columns.tolist()}")
    
    # Список признаков для API (должны совпадать с api.py)
    api_features = ['carrier_encoded', 'dep_hour', 'distance', 'weather_delay']
    
    # Проверяем, какие признаки есть
    available_features = [f for f in api_features if f in X_train.columns]
    
    if len(available_features) < len(api_features):
        print(f"⚠️ Предупреждение: Не все признаки доступны")
        print(f"   Доступно: {available_features}")
        print(f"   Ожидалось: {api_features}")
        
        # Создаем недостающие признаки
        for feature in api_features:
            if feature not in X_train.columns:
                if feature == 'carrier_encoded':
                    X_train[feature] = 0  # AA по умолчанию
                    X_test[feature] = 0
                elif feature == 'dep_hour':
                    X_train[feature] = 12  # полдень
                    X_test[feature] = 12
                elif feature == 'weather_delay':
                    X_train[feature] = 0.0
                    X_test[feature] = 0.0
    
    # Оставляем только признаки для API
    X_train_api = X_train[api_features].copy()
    X_test_api = X_test[api_features].copy()
    
    print(f"Признаки для обучения: {X_train_api.columns.tolist()}")
    print(f"Размер X_train_api: {X_train_api.shape}")
    
    return X_train_api, X_test_api

def train_logistic_regression(X_train, y_train, X_test, y_test, config):
    """Обучение логистической регрессии"""
    
    model_params = config['models']['logistic_regression']
    
    with mlflow.start_run(run_name=f"logistic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Логируем параметры
        mlflow.log_params(model_params)
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("purpose", "flight_delay_api")
        
        # Создаем и обучаем модель
        model = LogisticRegression(**model_params)
        print("Обучение модели...")
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Метрики
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1])
        }
        
        # Логируем метрики
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = "reports/confusion_matrix.png"
        os.makedirs('reports', exist_ok=True)
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # Classification Report
        report = classification_report(y_test, y_pred)
        report_path = "reports/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        
        # Сохраняем модель
        os.makedirs('models', exist_ok=True)
        model_path = "models/logistic_regression_model.pkl"
        joblib.dump(model, model_path)
        
        # Сохраняем имена признаков с моделью
        model_metadata = {
            'model': model,
            'feature_names': X_train.columns.tolist()
        }
        joblib.dump(model_metadata, model_path)
        
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)
        
        # Сохраняем информацию о признаках
        features_info = {
            'feature_names': X_train.columns.tolist(),
            'feature_count': X_train.shape[1],
            'training_samples': X_train.shape[0]
        }
        
        features_path = "models/model_features.json"
        import json
        with open(features_path, 'w') as f:
            json.dump(features_info, f)
        
        print(f"\nРезультаты модели:")
        print(f"Точность: {metrics['accuracy']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Признаки модели: {X_train.columns.tolist()}")
        
        return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Настраиваем MLflow
    mlflow.set_experiment(config['experiments']['experiment_name'])
    
    # Загружаем данные
    print("Загрузка данных...")
    X_train, X_test, y_train, y_test = load_data(config)
    
    # Подготавливаем признаки для API
    X_train_api, X_test_api = prepare_features_for_api(X_train, X_test)
    
    # Преобразуем целевые переменные в int
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # Проверяем баланс классов
    print(f"\nРаспределение классов:")
    print(f"Train: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
    print(f"Test:  0={sum(y_test==0)}, 1={sum(y_test==1)}")
    
    # Обучаем модель
    model = train_logistic_regression(
        X_train_api, y_train, 
        X_test_api, y_test, 
        config
    )
    
    # Тестируем модель на примере из API
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ НА ПРИМЕРЕ ИЗ API:")
    print("="*50)
    
    # Пример данных из API
    test_example = pd.DataFrame([{
        'carrier_encoded': 0,  # AA
        'dep_hour': 9,
        'distance': 550.0,
        'weather_delay': 0.0
    }])
    
    # Убедимся в правильном порядке признаков
    test_example = test_example[X_train_api.columns]
    
    prediction = model.predict_proba(test_example)[0, 1]
    print(f"Пример запроса: carrier=AA, dep_hour=9, distance=550, weather_delay=0")
    print(f"Предсказанная вероятность задержки: {prediction:.4f}")
    
    print("\n✅ Модель готова к использованию в API!")
    print(f"   Сохранена в: models/logistic_regression_model.pkl")
    print(f"   Признаки: {X_train_api.columns.tolist()}")

if __name__ == "__main__":
    main()