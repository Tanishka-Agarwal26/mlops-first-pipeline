
from fastapi.testclient import TestClient
from main import create_app
import pytest


class FakeIrisModel:
    def predict(self, features):
        return [0]


def fake_model_loader():
    return FakeIrisModel()


def fake_version_loader():
    return "test-version"


@pytest.fixture
def client():
    app = create_app(
        model_loader=fake_model_loader,
        version_loader=fake_version_loader
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_name"] == "iris-classifier"
    assert data["model_version"] == "test-version"


def test_valid_iris_prediction(client):

    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "class_name" in result

    assert result["class_name"] in [
        "setosa", "versicolor", "virginica"
    ]

    assert result["prediction"] in [0, 1, 2]
