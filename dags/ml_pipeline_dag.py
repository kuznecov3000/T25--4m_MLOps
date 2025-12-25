from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

import sys
sys.path.insert(0, '/home/user/mlops-flight-delay')  # Путь к проекту

from src.preprocess import load_and_clean_data
from src.train import train_model
from src.evaluate import evaluate_model

# ============= КОНФИГУРАЦИЯ DAG'а =============

default_args = {
    'owner': 'ml-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
}

dag = DAG(
    dag_id='ml_flight_delay_pipeline',
    default_args=default_args,
    description='ML pipeline для предсказания задержки рейсов',
    schedule_interval='@daily',  # Запускается каждый день в 00:00
    catchup=False,
    tags=['ml', 'flight-delay', 'production'],
)

# ============= PYTHON ФУНКЦИИ ДЛЯ ТАСК'ОВ =============

def preprocess_data(**context):
    """Задача 1: Предобработка данных"""
    print("=" * 50)
    print("Начинаем предобработку данных...")
    print("=" * 50)
    
    # Загружаем данные
    df = load_and_clean_data('data/raw/flights.csv')
    
    # Сохраняем обработанные данные
    df.to_csv('data/processed/flights_processed.csv', index=False)
    
    print(f"✓ Данные обработаны. Размер: {df.shape}")
    print(f"✓ Сохранено в: data/processed/flights_processed.csv")
    
    # Пушим метрики в XCom (для следующих таск'ов)
    context['task_instance'].xcom_push(
        key='rows_processed',
        value=len(df)
    )
    context['task_instance'].xcom_push(
        key='columns_count',
        value=len(df.columns)
    )

def train_model_task(**context):
    """Задача 2: Обучение модели"""
    print("=" * 50)
    print("Начинаем обучение модели...")
    print("=" * 50)
    
    # Получаем информацию из предыдущей таски
    ti = context['task_instance']
    rows = ti.xcom_pull(task_ids='preprocess_data', key='rows_processed')
    
    print(f"Используем {rows} примеров для обучения")
    
    # Загружаем обработанные данные
    import pandas as pd
    df = pd.read_csv('data/processed/flights_processed.csv')
    
    # Разделяем на train/test
    from sklearn.model_selection import train_test_split
    X = df.drop('dep_delay', axis=1)
    y = df['dep_delay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Тренируем модель
    model = train_model(X_train, y_train)
    
    # Сохраняем модель
    import joblib
    joblib.dump(model, 'models/flight_delay_model.pkl')
    
    print(f"✓ Модель обучена и сохранена")
    
    # Пушим информацию о модели
    ti.xcom_push(key='model_version', value='v1')
    ti.xcom_push(key='train_samples', value=len(X_train))

def evaluate_model_task(**context):
    """Задача 3: Оценка модели"""
    print("=" * 50)
    print("Начинаем оценку модели...")
    print("=" * 50)
    
    ti = context['task_instance']
    model_version = ti.xcom_pull(task_ids='train_model', key='model_version')
    train_samples = ti.xcom_pull(task_ids='train_model', key='train_samples')
    
    print(f"Оцениваем модель {model_version} (обучена на {train_samples} примерах)")
    
    # Загружаем данные
    import pandas as pd
    df = pd.read_csv('data/processed/flights_processed.csv')
    
    # Разделяем
    from sklearn.model_selection import train_test_split
    X = df.drop('dep_delay', axis=1)
    y = df['dep_delay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Загружаем обученную модель
    import joblib
    model = joblib.load('models/flight_delay_model.pkl')
    
    # Оцениваем
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"Метрики:")
    print(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
    print(f"  - F1 Score: {metrics.get('f1', 'N/A'):.3f}")
    print(f"  - MAE: {metrics.get('mae', 'N/A'):.3f}")
    
    # Сохраняем метрики
    import json
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Пушим метрики
    ti.xcom_push(key='accuracy', value=metrics.get('accuracy', 0))
    ti.xcom_push(key='f1', value=metrics.get('f1', 0))

def validate_model(**context):
    """Задача 4: Валидация модели"""
    print("=" * 50)
    print("Валидируем модель...")
    print("=" * 50)
    
    ti = context['task_instance']
    accuracy = ti.xcom_pull(task_ids='evaluate_model', key='accuracy')
    
    print(f"Accuracy модели: {accuracy:.3f}")
    
    # Проверяем что accuracy выше baseline
    MIN_ACCURACY = 0.75
    if accuracy < MIN_ACCURACY:
        raise ValueError(
            f"Model accuracy {accuracy:.3f} is below minimum {MIN_ACCURACY}. "
            "Model will not be deployed."
        )
    
    print(f"✓ Модель валидна! Accuracy > {MIN_ACCURACY}")

# ============= ОПРЕДЕЛЕНИЕ ТАСК'ОВ =============

# Таска 1: Предобработка
task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

# Таска 2: Обучение
task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

# Таска 3: Оценка
task_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag,
)

# Таска 4: Валидация
task_validate = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

# Таска 5: Логирование успеха
task_success = BashOperator(
    task_id='log_success',
    bash_command='echo "✓ ML Pipeline completed successfully!" && date',
    dag=dag,
)

# ============= ОПРЕДЕЛЕНИЕ ЗАВИСИМОСТЕЙ (DAG) =============

# Порядок выполнения
task_preprocess >> task_train >> task_evaluate >> task_validate >> task_success

"""
Граф зависимостей:
preprocess_data → train_model → evaluate_model → validate_model → log_success
"""