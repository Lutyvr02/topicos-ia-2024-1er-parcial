import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from config import get_settings
from models import Gun, Person, PixelLocation  # Asegúrate que estas clases están definidas en src.models

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array


@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...), 
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)
    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


# Nuevo Endpoint: /detect_people
@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    img_array = np.array(img_obj)
    segmentation = detector.segment_people(img_array, threshold)
    return segmentation


# Nuevo Endpoint: /annotate_people
@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    img_array = np.array(img_obj)
    segmentation = detector.segment_people(img_array, threshold)
    annotated_img = annotate_segmentation(img_array, segmentation, draw_boxes)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


# Nuevo Endpoint: /detect
@app.post("/detect")
def detect(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> dict:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    img_array = np.array(img_obj)
    detection = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(img_array, threshold)
    
    return {"detection": detection, "segmentation": segmentation}


# Nuevo Endpoint: /annotate
@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    img_array = np.array(img_obj)
    detection = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(img_array, threshold)

    annotated_img = annotate_detection(img_array, detection)
    final_annotated_img = annotate_segmentation(annotated_img, segmentation, draw_boxes)

    img_pil = Image.fromarray(final_annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


# Nuevo Endpoint: /guns
@app.post("/guns")
def guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    img_array = np.array(img_obj)
    detection = detector.detect_guns(img_array, threshold)
    
    gun_list = []
    for label, box in zip(detection.labels, detection.boxes):
        x_center = (box[0] + box[2]) // 2
        y_center = (box[1] + box[3]) // 2
        gun = Gun(gun_type=label, location=PixelLocation(x=x_center, y=y_center))
        gun_list.append(gun)
    
    return gun_list


# Nuevo Endpoint: /people
@app.post("/people")
def people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list:
    img_stream = io.BytesIO(file.file.read())
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    img_array = np.array(img_obj)
    segmentation = detector.segment_people(img_array, threshold)
    
    people_list = []
    for label, box, polygon in zip(segmentation.labels, segmentation.boxes, segmentation.polygons):
        x_center = (box[0] + box[2]) // 2
        y_center = (box[1] + box[3]) // 2
        area = int(cv2.contourArea(np.array(polygon)))  # Convertir área a entero
        person = Person(person_type=label, location=PixelLocation(x=x_center, y=y_center), area=area)
        people_list.append(person)
    
    return people_list


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8080, host="127.0.0.1")
