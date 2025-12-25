from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

# Получаем logger Airflow
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'ml-team',
    'start_date': datetime(2025, 1, 1),
}

dag = DAG(
    dag_id='ml_flight_delay_with_logging',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
)

def task_with_logging(**context):
    """Таска с детальным логированием"""
    
    logger.info("=" * 50)
    logger.info("Начинаем обработку...")
    logger.info("=" * 50)
    
    execution_date = context['execution_date']
    logger.info(f"Execution date: {execution_date}")
    
    try:
        # Ваш код
        logger.info("Загружаем данные...")
        import pandas as pd
        df = pd.read_csv('data/raw/flights.csv')
        
        logger.info(f"✓ Данные загружены: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Статистика
        logger.debug(f"Descriptive stats:\n{df.describe()}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {str(e)}", exc_info=True)
        raise
    
    logger.info("✓ Таска завершена успешно!")

task = PythonOperator(
    task_id='process_with_logging',
    python_callable=task_with_logging,
    dag=dag,
)