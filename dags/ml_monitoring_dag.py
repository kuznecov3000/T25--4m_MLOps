from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import json

default_args = {
    'owner': 'ml-team',
    'start_date': datetime(2025, 1, 1),
}

dag = DAG(
    dag_id='ml_monitoring_pipeline',
    default_args=default_args,
    description='DAG для мониторинга качества модели',
    schedule_interval='@daily',
    catchup=False,
    tags=['monitoring'],
)

def check_model_drift(**context):
    """Проверка data drift"""
    print("Проверяем data drift...")
    
    import pandas as pd
    from scipy.stats import ks_2samp
    import numpy as np
    
    # Загружаем training data (reference)
    train_data = pd.read_csv('data/raw/flights.csv')
    
    # Загружаем текущие данные
    current_data = pd.read_csv('data/processed/flights_processed.csv')
    
    # Проверяем drift по ключевым признакам
    features_to_check = ['dep_hour', 'distance']
    
    drifts = {}
    for feature in features_to_check:
        if feature in train_data.columns and feature in current_data.columns:
            stat, p_value = ks_2samp(train_data[feature].dropna(), 
                                     current_data[feature].dropna())
            
            drifted = p_value < 0.05
            drifts[feature] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'drifted': bool(drifted)
            }
            
            print(f"  {feature}: {'DRIFT!' if drifted else 'OK'} (p-value: {p_value:.4f})")
    
    # Сохраняем результаты
    with open('reports/drift_report.json', 'w') as f:
        json.dump(drifts, f, indent=2)
    
    context['task_instance'].xcom_push(key='drifts', value=drifts)

def alert_if_drift(**context):
    """Отправляем алерт если есть drift"""
    ti = context['task_instance']
    drifts = ti.xcom_pull(task_ids='check_drift', key='drifts')
    
    has_drift = any(d['drifted'] for d in drifts.values())
    
    if has_drift:
        print("⚠️  DATA DRIFT DETECTED!")
        print("Дрифт найден в признаках:")
        for feature, data in drifts.items():
            if data['drifted']:
                print(f"  - {feature} (p-value: {data['p_value']:.4f})")
    else:
        print("✓ No data drift detected")

# Таски
task_check_drift = PythonOperator(
    task_id='check_drift',
    python_callable=check_model_drift,
    dag=dag,
)

task_alert = PythonOperator(
    task_id='alert_if_drift',
    python_callable=alert_if_drift,
    dag=dag,
)

task_check_drift >> task_alert