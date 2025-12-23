FROM python:3.10-slim

# Назначим рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Установим пакеты
RUN pip install --no-cache-dir -r requirements.txt

# Скопируем исходники
COPY . .

# Запускаем FastAPI через uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]