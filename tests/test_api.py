import pytest
from fastapi.testclient import TestClient
from src import main

# Crear una instancia de TestClient para la aplicación FastAPI
client = TestClient(main)

# Ruta de las imágenes de prueba
IMAGE_FILES = ["test/gun1.jpg", "test/gun2.jpg", "test/gun6.jpg"]

@pytest.mark.parametrize("image_file", IMAGE_FILES)
def test_get_model_info(image_file):
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert "gun_detector_model" in response.json()
    assert "semantic_segmentation_model" in response.json()
    assert response.json()["input_type"] == "image"


@pytest.mark.parametrize("image_file", IMAGE_FILES)
def test_detect_guns(image_file):
    with open(image_file, "rb") as f:
        response = client.post("/detect_guns", files={"file": (image_file, f, "image/jpeg")})
    
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "n_detections" in response.json()
    assert "boxes" in response.json()


@pytest.mark.parametrize("image_file", IMAGE_FILES)
def test_annotate_guns(image_file):
    with open(image_file, "rb") as f:
        response = client.post("/annotate_guns", files={"file": (image_file, f, "image/jpeg")})
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("image_file", IMAGE_FILES)
def test_detect_people(image_file):
    with open(image_file, "rb") as f:
        response = client.post("/detect_people", files={"file": (image_file, f, "image/jpeg")})

    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "n_detections" in response.json()
    assert "boxes" in response.json()
    assert "polygons" in response.json()


@pytest.mark.parametrize("image_file", IMAGE_FILES)
def test_annotate_people(image_file):
    with open(image_file, "rb") as f:
        response = client.post("/annotate_people", files={"file": (image_file, f, "image/jpeg")})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("image_file", IMAGE_FILES)
def test_detect(image_file):
    with open(image_file, "rb") as f:
        response = client.post("/detect", files={"file": (image_file, f, "image/jpeg")})

    assert response.status_code == 200
    assert "detection" in response.json()
    assert "segmentation" in response.json()


@pytest.mark.parametrize("image_file", IMAGE_FILES)
def test_annotate(image_file):
    with open(image_file, "rb") as f:
        response = client.post("/annotate", files={"file": (image_file, f, "image/jpeg")})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("image_file", IMAGE_FILES)
def test_guns(image_file):
    with open(image_file, "rb") as f:
        response = client.post("/guns", files={"file": (image_file, f, "image/jpeg")})

    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.parametrize("image_file", IMAGE_FILES)
def test_people(image_file):
    with open(image_file, "rb") as f:
        response = client.post("/people", files={"file": (image_file, f, "image/jpeg")})

    assert response.status_code == 200
    assert isinstance(response.json(), list)

