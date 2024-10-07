from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from models import Detection, PredictionType, Segmentation
from config import get_settings

SETTINGS = get_settings() 

def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    matched_box = None
    min_distance = float('inf')
    
    person_box = box(segment[0], segment[1], segment[2], segment[3])

    for bbox in bboxes:
        gun_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
        distance = person_box.distance(gun_box)
        
        if distance < max_distance and distance < min_distance:
            min_distance = distance
            matched_box = bbox

    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img

def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()

    for label, box in zip(segmentation.labels, segmentation.boxes):
        color = (0, 255, 0) if label == "safe" else (0, 0, 255)
        mask_polygon = np.array(segmentation.polygons, dtype=np.int32)
        
        cv2.fillPoly(annotated_img, [mask_polygon], color)

        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )
    
    return annotated_img


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]].lower() for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )

    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10) -> Segmentation:
        results = self.seg_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        person_indexes = [i for i in range(len(labels)) if labels[i] == 0]
        person_boxes = [
            [int(v) for v in box] for i, box in enumerate(results.boxes.xyxy.tolist()) if i in person_indexes
        ]
        
        # Convertir los polígonos a enteros
        polygons = [
            [[int(coord) for coord in point] for point in results.masks.xy[i].tolist()] for i in person_indexes
        ]

        detection = self.detect_guns(image_array, threshold)

        person_labels = []
        areas = []
        for person_box in person_boxes:
            gun_box = match_gun_bbox(person_box, detection.boxes, max_distance)
            if gun_box is not None:
                person_labels.append("danger")
            else:
                person_labels.append("safe")
            
            # Calcular el área y asegurarse de que sea un entero
            x1, y1, x2, y2 = person_box
            area = (x2 - x1) * (y2 - y1)
            areas.append(int(area))  # Convertir el área a entero

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(person_boxes),
            polygons=polygons,  # Asegurarse de que los valores son enteros
            boxes=person_boxes,
            labels=person_labels,
            areas=areas  # Devolver el área como entero
        )