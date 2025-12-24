def test_predict_endpoint_returns_json(client, sample_flight_data):
    """Проверка, что /predict возвращает JSON"""
    response = client.post("/predict", json=sample_flight_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

def test_predict_output_format(client, sample_flight_data):
    """Проверка формата ответа /predict"""
    response = client.post("/predict", json=sample_flight_data)
    assert response.status_code == 200
    json_data = response.json()
    assert "delay_probability" in json_data
    assert "carrier" in json_data
    assert "status" in json_data

def test_predict_probability_range(client, sample_flight_data):
    """Проверка, что вероятность в диапазоне [0, 1]"""
    response = client.post("/predict", json=sample_flight_data)
    assert response.status_code == 200
    json_data = response.json()
    prob = json_data["delay_probability"]
    assert 0 <= prob <= 1

def test_root_endpoint(client):
    """Проверка корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "message" in json_data
    assert "version" in json_data

def test_health_check(client):
    """Проверка эндпоинта health"""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert "status" in json_data
    assert "model_loaded" in json_data