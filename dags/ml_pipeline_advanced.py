from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.task_group import TaskGroup
from airflow.exceptions import AirflowException

# ============= КОНФИГУРАЦИЯ =============

default_args = {
    'owner': 'ml-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': 'ml-team@company.com',
}

dag = DAG(
    dag_id='ml_flight_delay_pipeline_v2',
    default_args=default_args,
    description='Advanced ML pipeline с обработкой ошибок',
    schedule_interval='@daily',
    max_active_runs=1,  # Одновременно запускается только одна версия
    catchup=False,
    tags=['ml', 'flight-delay', 'advanced'],
)

# ============= ФУНКЦИИ =============

def check_data_quality(**context):
    """Проверка качества данных перед обработкой"""
    print("Проверяем качество данных...")
    
    import pandas as pd
    df = pd.read_csv('data/raw/flights.csv')
    
    # Проверяем пропуски
    missing_pct = df.isnull().sum().max() / len(df) * 100
    if missing_pct > 50:
        raise AirflowException(f"Too many missing values: {missing_pct:.1f}%")
    
    # Проверяем размер
    if len(df) < 1000:
        raise AirflowException(f"Dataset too small: {len(df)} rows")
    
    print(f"✓ Data quality OK. Rows: {len(df)}, Missing: {missing_pct:.1f}%")

def preprocess_data_v2(**context):
    """Предобработка с более строгой обработкой ошибок"""
    print("Предобработка данных (v2)...")
    
    try:
        import pandas as pd
        df = pd.read_csv('data/raw/flights.csv')
        
        # Очистка
        df = df.dropna(thresh=len(df) * 0.8, axis=1)
        df = df.dropna()
        
        # Валидация
        if 'dep_delay' not in df.columns:
            raise ValueError("Missing target column 'dep_delay'")
        
        # Сохранение
        df.to_csv('data/processed/flights_processed.csv', index=False)
        
        context['task_instance'].xcom_push(
            key='data_shape',
            value=f"{df.shape[0]}x{df.shape[1]}"
        )
        
        print(f"✓ Data preprocessed: {df.shape}")
        
    except Exception as e:
        raise AirflowException(f"Preprocessing failed: {str(e)}")

def train_with_cv(**context):
    """Обучение с cross-validation"""
    print("Обучение модели с CV...")
    
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    df = pd.read_csv('data/processed/flights_processed.csv')
    
    X = df.drop('dep_delay', axis=1)
    y = df['dep_delay']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    
    print(f"CV Scores: {scores}")
    print(f"Mean CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
    
    # Полное обучение
    model.fit(X, y)
    joblib.dump(model, 'models/flight_delay_model.pkl')
    
    context['task_instance'].xcom_push(key='cv_mean', value=float(scores.mean()))
    context['task_instance'].xcom_push(key='cv_std', value=float(scores.std()))

def generate_report(**context):
    """Генерирование отчета"""
    print("Генерируем отчет...")
    
    ti = context['task_instance']
    data_shape = ti.xcom_pull(task_ids='preprocess_data', key='data_shape')
    cv_mean = ti.xcom_pull(task_ids='train_model', key='cv_mean')
    cv_std = ti.xcom_pull(task_ids='train_model', key='cv_std')
    
    report = f"""
    ============ ML PIPELINE REPORT ============
    
    Execution Date: {context['execution_date']}
    DAG: {context['dag'].dag_id}
    
    DATA STATISTICS:
    - Shape: {data_shape}
    
    MODEL PERFORMANCE:
    - CV Mean Score: {cv_mean:.3f}
    - CV Std: {cv_std:.3f}
    
    STATUS: ✓ SUCCESS
    
    ==========================================
    """
    
    print(report)
    
    # Сохраняем отчет
    with open('reports/pipeline_report.txt', 'w') as f:
        f.write(report)

# ============= TASK GROUPS =============

with TaskGroup("data_pipeline", dag=dag) as data_group:
    check_quality = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
    )
    
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_v2,
    )
    
    check_quality >> preprocess

with TaskGroup("model_pipeline", dag=dag) as model_group:
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_with_cv,
    )
    
    report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
    )
    
    train >> report

# ============= DAG FLOW =============

data_group >> model_group