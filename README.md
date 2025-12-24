# mlops-flight-delay
демо проект курса MLOps (7с)

## 📌 Описание
Мини-проект для курса по MLOps: предсказание задержки рейса.

## 🚀 Быстрый старт
1. Установите окружение:
   ```bash
   pip install -r requirements.txt

## Структура проекта

```bash
mlops-flight-delay/
│
├── data/                
│   └── raw/             # исходные данные (через DVC)
│
├── models/              # сохранённые модели
├── reports/             # отчёты (метрики, графики)
│
├── src/
│   ├── preprocess.py    # очистка и подготовка данных
│   ├── train.py         # обучение модели
│   ├── evaluate.py      # оценка модели
│   ├── predict.py       # скрипт для инференса
│   ├── api.py           # REST API (FastAPI)
│   └── utils.py         # вспомогательные функции
│
├── tests/               # unit-тесты
│   └── test_dummy.py
│
├── .gitignore
├── requirements.txt
├── dvc.yaml
├── Dockerfile
└── README.md

```
### Предобработка данных:

```bash
python src/preprocess.py
```

### Обучение модели:

```bash
python src/train.py
```

### Запуск API:

```bash
uvicorn src.api:app --reload
```
# Flight Delay API (Lab 6)

## Запуск локально
uvicorn src.api:app --host 0.0.0.0 --port 8080


## Сборка и запуск контейнера
docker build -t flight-api .
docker run -d -p 8080:8080 --name flight-api flight-api


## Пример запроса
curl -X POST "http://localhost:8080/predict"
-H "Content-Type: application/json"
-d '{"carrier":"AA","dep_hour":9,"distance":550}'


## Ответ
{"delay_probability": 0.3035}


# Flight Delay Prediction

![CI Status](LR7.jpg)

...
