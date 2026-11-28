from __future__ import annotations

import io
import json
import logging
import os
import hashlib
import threading
import subprocess
import imageio_ffmpeg
from database import (
    DatabaseConfigurationError,
    DatabaseUnavailableError,
    get_db_connection,
    initialize_database as initialize_mysql_database,
    verify_required_tables,
)
import textwrap
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path

import cv2
from auth import auth_bp
import joblib
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, g, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from xclip_classifier import XCLIPEventClassifier

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

BASE_DIR = Path(__file__).resolve().parent
LOCAL_FRONTEND_ENV = BASE_DIR.parent / "frontend" / ".env.local"
if LOCAL_FRONTEND_ENV.is_file():
    load_dotenv(LOCAL_FRONTEND_ENV)

DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR))
UPLOAD_DIR = DATA_DIR / "uploads"
GENERATED_DIR = DATA_DIR / "generated"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)

analysis_jobs = {}
analysis_jobs_lock = threading.Lock()
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


class EventPipeline:
    EVENT_LABELS = [
        "project presentation",
        "speech",
        "panel discussion",
        "dance performance",
        "singing performance",
        "prize distribution",
        "cultural programme",
        "unknown",
    ]

    def __init__(self):
        self.model_metadata = {}
        self.model = self._build_model()
        self.model_quality = "trained" if self.model is not None else "unavailable"
        self.yolo_model = self._load_yolo_model()

    def _load_yolo_model(self):
        if YOLO is None:
            logging.warning("Ultralytics is unavailable; using heuristic classification only.")
            return None
        try:
            return YOLO(str(BASE_DIR / "yolov8n.pt"))
        except Exception as exc:  # pragma: no cover
            logging.warning("YOLO model could not be loaded: %s", exc)
            return None

    def _build_model(self):
     configured_model = os.environ.get("EVENT_MLP_MODEL_PATH")
     model_path = Path(configured_model) if configured_model else BASE_DIR / "event_classifier.joblib"

     if model_path.is_file():
        loaded = joblib.load(model_path)
        if not isinstance(loaded, dict) or "model" not in loaded:
            logging.warning(
                "Event classifier must be a model package containing model and metadata."
            )
            return None
        model = loaded["model"]
        metadata = loaded.get("metadata")
        if not isinstance(metadata, dict):
            logging.warning("Event classifier package has no valid metadata.")
            return None
        feature_dimension = metadata.get("feature_dimension")
        model_dimension = getattr(model, "n_features_in_", None)
        if feature_dimension != 512 or model_dimension != 512:
            logging.warning(
                "Event classifier feature dimensions are incompatible (metadata=%s, model=%s).",
                feature_dimension,
                model_dimension,
            )
            return None
        model_classes = {
            str(label).strip().lower()
            for label in getattr(model, "classes_", [])
        }
        metadata_classes = {
            str(label).strip().lower()
            for label in metadata.get("class_labels", [])
        }
        supported_classes = set(self.EVENT_LABELS)
        if not model_classes or not model_classes.issubset(supported_classes):
            logging.warning("Event classifier contains unsupported event labels.")
            return None
        if metadata_classes and metadata_classes != model_classes:
            logging.warning("Event classifier metadata does not match model classes.")
            return None
        self.model_metadata = metadata
        return model

     if configured_model:
        raise FileNotFoundError(
            f"Configured MLP model does not exist: {model_path}"
        )

     logging.debug(
        "Optional legacy 512-D event_classifier.joblib not present; using EventLensMLPClassifier."
    )
     return None

    def classify_video_features(self, features, detection_result):
        if self.model is None or features is None:
            return None

        vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != 512:
            logging.warning(
                "Skipping event classification: expected 512 R3D features, got %s.",
                vector.shape[1],
            )
            return None

        try:
            event_type = str(self.model.predict(vector)[0]).strip().lower()
            if event_type not in self.EVENT_LABELS:
                logging.warning(
                    "Event classifier returned an unsupported category: %s",
                    event_type,
                )
                return None
            probabilities = self.model.predict_proba(vector)[0]
            confidence = float(np.max(probabilities))
        except (AttributeError, ValueError, TypeError):
            logging.exception("Event-level classifier inference failed")
            return None

        counts = Counter({
            item["class_name"]: item["count"]
            for item in detection_result.get("top_objects", [])
        })
        return {
            "event_type": event_type,
            "confidence": round(confidence, 3),
            "summary": self._build_summary(event_type, counts),
            "model_quality": "trained",
        }

    def _feature_vector(self, detections, counts=None):
        counts = counts or Counter()
        confidence_values = []

        for detection in detections:
            label = str(detection.get("class_name", "unknown")).lower()
            confidence_values.append(float(detection.get("confidence", 0.0)))

        return [
            counts.get("person", 0),
            counts.get("laptop", 0),
            counts.get("tvmonitor", 0) + counts.get("monitor", 0),
            counts.get("microphone", 0),
            counts.get("chair", 0),
            counts.get("cell phone", 0) + counts.get("cellphone", 0),
            float(np.mean(confidence_values)) if confidence_values else 0.0,
        ]

    def _fallback_event_label(self, counts):
        people = counts.get("person", 0)
        laptop = counts.get("laptop", 0)
        microphone = counts.get("microphone", 0)
        chairs = counts.get("chair", 0)
        phone = counts.get("cell phone", 0) + counts.get("cellphone", 0)

        if people >= 1 and (laptop > 0 or counts.get("tvmonitor", 0) > 0):
            return "presentation"
        if microphone > 0 and people >= 1:
            return "speech"
        if people >= 3 and (chairs > 0 or microphone > 0):
            return "panel discussion"
        if people >= 2 and microphone > 0:
            return "singing performance"
        if people >= 2 and phone > 0:
            return "prize distribution"
        if people >= 2:
            return "cultural programme"
        return "unknown"

    def process_detections(self, detections, frame_detections=None):
        if not detections:
            return {
                "event_type": "unknown",
                "confidence": None,
                "summary": "No clear event objects were detected in the uploaded video.",
                "top_objects": [],
                "peak_person_count": 0,
                "model_quality": self.model_quality,
            }

        frame_counts = []
        if frame_detections:
            for frame in frame_detections:
                frame_counts.append(Counter(
                    str(item.get("class_name", "unknown")).lower() for item in frame
                ))
        else:
            frame_counts.append(Counter(
                str(item.get("class_name", "unknown")).lower() for item in detections
            ))

        labels = set().union(*(frame_count.keys() for frame_count in frame_counts))
        counts = Counter({
            label: max(frame_count.get(label, 0) for frame_count in frame_counts)
            for label in labels
        })

        feature_vector = self._feature_vector(detections, counts)
        if self.model is not None:
            expected_features = getattr(self.model, "n_features_in_", len(feature_vector))
            if int(expected_features) != len(feature_vector):
                event_type = "unknown"
                confidence = None
                supported_labels = self.EVENT_LABELS
            else:
                try:
                    event_type = str(self.model.predict([feature_vector])[0])
                    probabilities = self.model.predict_proba([feature_vector])[0]
                    confidence = float(np.max(probabilities))
                    supported_labels = getattr(self.model, "classes_", self.EVENT_LABELS)
                except (AttributeError, ValueError):
                    event_type = "unknown"
                    confidence = None
                    supported_labels = self.EVENT_LABELS
        else:
             event_type = "unknown"
             confidence = None
             supported_labels = self.EVENT_LABELS

        if event_type not in {str(label) for label in supported_labels}:
            event_type = "unknown"

        event_summary = self._build_summary(event_type, counts)
        return {
            "event_type": event_type,
            "confidence": (
                round(float(confidence), 3)
                if confidence is not None else None
            ),
            "summary": event_summary,
            "top_objects": [{"class_name": label, "count": count} for label, count in counts.most_common(4)],
            "peak_person_count": counts.get("person", 0),
            "model_quality": self.model_quality,
        }

    def _build_summary(self, event_type, counts):
        people = counts.get("person", 0)
        laptops = counts.get("laptop", 0)
        monitors = counts.get("tvmonitor", 0) + counts.get("monitor", 0)
        microphones = counts.get("microphone", 0)
        chairs = counts.get("chair", 0)

        if event_type == "unknown":
            return (
                f"The scene could not be matched confidently to a trained event category. "
                f"The peak number of visible person detections in a sampled frame was {people}."
            )

        observed_objects = ", ".join(
            f"{count} {name}(s)" for name, count in counts.most_common(4) if name != "person"
        )
        evidence = f" Other visible objects included {observed_objects}." if observed_objects else ""
        return (
            f'The video most closely matches the trained event category "{event_type}". '
            f"The peak number of visible person detections in a sampled frame was {people}."
            f"{evidence} The person count is not a unique attendee count."
        )

    def run_yolo(self, frame):
        if self.yolo_model is None:
            return []

        height, width = frame.shape[:2]
        inference_frame = frame
        scale_x = scale_y = 1.0
        max_inference_width = int(os.environ.get("YOLO_MAX_INFERENCE_WIDTH", "1280"))
        if width > max_inference_width:
            scale = max_inference_width / float(width)
            inference_width = max_inference_width
            inference_height = max(1, int(round(height * scale)))
            inference_frame = cv2.resize(
                frame,
                (inference_width, inference_height),
                interpolation=cv2.INTER_AREA,
            )
            scale_x = width / float(inference_width)
            scale_y = height / float(inference_height)

        result = self.yolo_model(inference_frame, conf=0.25, verbose=False)
        detections = []

        for item in result:
            boxes = item.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cls_index = int(box.cls[0].cpu().item())
                class_name = item.names.get(cls_index, "object")
                confidence = float(box.conf[0].cpu().item())
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [
                            round((((x1 * scale_x) / width) * 100), 2),
                            round((((y1 * scale_y) / height) * 100), 2),
                            round((((x2 * scale_x) / width) * 100), 2),
                            round((((y2 * scale_y) / height) * 100), 2),
                        ],
                    }
                )

        return detections

    def track_people(self, frames):
        if self.yolo_model is None or not frames:
            return {
                "unique_tracked_people": None,
                "peak_tracked_people": 0,
                "tracking_status": "unavailable",
            }

        tracked_ids = set()
        peak_count = 0

        try:
            for frame in frames:
                height, width = frame.shape[:2]
                inference_frame = frame
                max_width = int(os.environ.get("YOLO_MAX_INFERENCE_WIDTH", "1280"))
                if width > max_width:
                    scale = max_width / float(width)
                    inference_frame = cv2.resize(
                        frame,
                        (max_width, max(1, int(round(height * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )

                results = self.yolo_model.track(
                    inference_frame,
                    conf=0.25,
                    persist=True,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )
                frame_people = 0
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    ids = boxes.id
                    classes = boxes.cls
                    for index, cls_index in enumerate(classes):
                        if int(cls_index.cpu().item()) != int(
                            next(
                                (
                                    key
                                    for key, name in result.names.items()
                                    if name == "person"
                                ),
                                -1,
                            )
                        ):
                            continue
                        frame_people += 1
                        if ids is not None:
                            track_id = int(ids[index].cpu().item())
                            tracked_ids.add(track_id)
                peak_count = max(peak_count, frame_people)
        except Exception:
            logging.exception("Local person tracking failed")
            return {
                "unique_tracked_people": None,
                "peak_tracked_people": peak_count,
                "tracking_status": "failed",
            }

        if not tracked_ids:
            return {
                "unique_tracked_people": None,
                "peak_tracked_people": peak_count,
                "tracking_status": "tracking_ids_unavailable",
            }

        return {
            "unique_tracked_people": len(tracked_ids),
            "peak_tracked_people": peak_count,
            "tracking_status": "ready",
        }


class VideoActionRecognizer:
    RELEVANT_ACTIONS = {
        "presenting",
        "public speaking",
        "giving a speech",
        "teaching",
        "dancing",
        "singing",
        "clapping",
        "playing musical instrument",
    }

    def __init__(self):
        self.enabled = os.environ.get(
            "EVENT_ACTION_MODEL_ENABLED", "true"
        ).lower() not in {"0", "false", "no", "off"}

        self.model = None
        self.weights = None
        self.torch = None
        self.device = None
        self.load_attempted = False
        self.status = "not_loaded" if self.enabled else "disabled"

        self.clip_length = 16
        self.num_clips = 5

    def _load_model(self):
        import torch
        from torchvision.models.video import R3D_18_Weights, r3d_18

        self.torch = torch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.weights = R3D_18_Weights.KINETICS400_V1
        extractor = r3d_18(weights=self.weights)
        extractor.fc = torch.nn.Identity()
        self.model = extractor.to(self.device).eval()

        action_classifier_path = (
            BASE_DIR / "models" / "eventlens_action_classifier_v2.pkl"
        )

        self.action_classifier = None
        self.action_classes = []

        if action_classifier_path.is_file():
            package = joblib.load(action_classifier_path)

            self.action_classifier = package["classifier"]
            self.action_classes = package["classes"]

            if len(self.action_classes) != 12:
                raise ValueError("Invalid EventLens classifier classes.")

            if package["feature_dimensions"] != 512:
                raise ValueError("Invalid EventLens feature dimensions.")

            print(
                f"EventLens V2 classifier loaded: "
                f"{len(self.action_classes)} classes"
            )
        else:
            raise FileNotFoundError(
                f"Trained classifier not found: {action_classifier_path}"
            )

        self.status = "ready"
    def _extract_clip(self, capture, start_frame):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

        frames = []

        for _ in range(self.clip_length):
            success, frame = capture.read()

            if not success:
                break

            resized = cv2.resize(
                frame,
                (171, 128),
                interpolation=cv2.INTER_AREA,
            )

            rgb = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(rgb)

        if not frames:
            return None

        while len(frames) < self.clip_length:
            frames.append(frames[-1].copy())

        return frames

    def _predict_clip(self, frames):
        if self.model is None:
            self.load_attempted = True
            self._load_model()

        rgb_array = np.stack(frames)

        clip = self.torch.from_numpy(rgb_array).permute(0, 3, 1, 2)
        clip = self.weights.transforms()(clip)
        clip = clip.unsqueeze(0).to(self.device)

        with self.torch.inference_mode():
            features = self.model(clip)

        features = features.cpu().numpy().reshape(1, -1)

        if self.action_classifier is not None:
            decision_scores = self.action_classifier.decision_function(
                features
            )

            if decision_scores.ndim == 1:
                decision_scores = decision_scores.reshape(1, -1)

            scores = decision_scores[0]

            class_ids = list(self.action_classifier.classes_)

            classes = [
                self.action_classes[int(class_id)]
                for class_id in class_ids
            ]

            return scores, classes, features[0]

        return np.array([]), [], features[0]

    def _format_predictions(self, scores, top_k=3, categories=None):
        categories = categories or []
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "action_name": categories[int(index)],
                "confidence": None,
                "model_score": round(float(scores[index]), 4),
                "score_type": "uncalibrated LinearSVC decision score",
            }
            for index in indices
        ]

    def predict_video(self, video_path):
        if not self.enabled:
            self.status = "disabled"
            return {
                "action_predictions": [],
                "action_clip_details": [],
                "action_timeline": [],
                "action_summary": "Action recognition is disabled.",
            }

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            self.status = "unavailable"
            return {
                "action_predictions": [],
                "action_clip_details": [],
                "action_timeline": [],
                "action_summary": "Unable to open the uploaded video.",
            }

        try:
            total_frames = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = capture.get(cv2.CAP_PROP_FPS)

            if total_frames <= 0:
                raise ValueError("Video contains no readable frames.")

            if not fps or fps <= 0:
                fps = 25.0

            duration = total_frames / fps

            max_start = max(
                0,
                total_frames - self.clip_length
            )

            if max_start == 0:
                clip_starts = [0]
            else:
                clip_starts = np.linspace(
                    0,
                    max_start,
                    num=min(self.num_clips, max_start + 1),
                ).round().astype(int).tolist()

            clip_details = []
            action_timeline = []
            score_vectors = []
            feature_vectors = []

            for clip_number, start_frame in enumerate(
                clip_starts, start=1
            ):
                frames = self._extract_clip(
                    capture,
                    start_frame,
                )

                if not frames:
                    continue

                scores, categories, video_features = self._predict_clip(frames)
                if not categories:
                    continue

                score_vectors.append(scores)
                feature_vectors.append(video_features)

                start_seconds = start_frame / fps
                end_seconds = min(
                    duration,
                    (start_frame + self.clip_length) / fps,
                )

                predictions = self._format_predictions(
                    scores,
                    top_k=3,
                    categories=categories,
                )

                clip_details.append({
                    "clip_number": clip_number,
                    "start_seconds": round(start_seconds, 2),
                    "end_seconds": round(end_seconds, 2),
                    "predictions": predictions,
                })

                if predictions and end_seconds > start_seconds:
                    best = predictions[0]
                    action_timeline.append({
                        "activity_name": best["action_name"],
                        "start_seconds": round(start_seconds, 2),
                        "end_seconds": round(end_seconds, 2),
                        "confidence": None,
                        "model_score": best["model_score"],
                        "score_type": best["score_type"],
                        "evidence": "R3D-18 clip features scored by the trained UCF101 LinearSVC.",
                        "source": "r3d-18-ucf101-v2",
                        "uncertainty": "Supplementary action prediction; not a confirmed college event.",
                    })

            if not score_vectors:
                raise ValueError(
                    "No usable video clips could be extracted."
                )

            average_scores = np.mean(
                score_vectors,
                axis=0,
            )
            categories_for_average = categories

            action_predictions = self._format_predictions(
                average_scores,
                top_k=3,
                categories=categories_for_average,
            )

            action_counts = Counter(
                item["activity_name"]
                for item in action_timeline
            )

            action_summary = ", ".join(
                f"{name} ({count} clip(s))"
                for name, count in action_counts.most_common(5)
            )

            self.status = "ready"

            merged_timeline = []
            for item in sorted(action_timeline, key=lambda row: row["start_seconds"]):
                if item["end_seconds"] <= item["start_seconds"]:
                    continue
                if (
                    merged_timeline
                    and merged_timeline[-1]["activity_name"] == item["activity_name"]
                    and item["start_seconds"] <= merged_timeline[-1]["end_seconds"] + 1.0
                ):
                    merged_timeline[-1]["end_seconds"] = max(
                        merged_timeline[-1]["end_seconds"],
                        item["end_seconds"],
                    )
                    merged_timeline[-1]["model_score"] = max(
                        merged_timeline[-1]["model_score"],
                        item["model_score"],
                    )
                    continue
                merged_timeline.append(dict(item))

            return {
                "action_predictions": action_predictions,
                "action_clip_details": clip_details,
                "action_timeline": merged_timeline,
                "action_summary": action_summary,
                "_video_feature": np.mean(
                    np.stack(feature_vectors),
                    axis=0,
                ),
            }

        except Exception:
            self.status = "unavailable"

            logging.exception(
                "Multi-clip action recognition failed"
            )

            return {
                "action_predictions": [],
                "action_clip_details": [],
                "action_timeline": [],
                "action_summary": (
                    "Action recognition could not be completed."
                ),
            }

        finally:
            capture.release()

    def predict(self, frames):
        if not self.enabled or not frames:
            return []

        try:
            if self.model is None:
                if self.load_attempted:
                    return []

            clip_frames = []

            indices = np.linspace(
                0,
                len(frames) - 1,
                num=self.clip_length,
            ).round().astype(int)

            for index in indices:
                resized = cv2.resize(
                    frames[int(index)],
                    (171, 128),
                    interpolation=cv2.INTER_AREA,
                )

                clip_frames.append(
                    cv2.cvtColor(
                        resized,
                        cv2.COLOR_BGR2RGB,
                    )
                )

            probabilities, categories = self._predict_clip(
                clip_frames
            )

            self.status = "ready"

            return self._format_predictions(
                probabilities,
                top_k=3,
                categories=categories,
            )

        except Exception:
            self.status = "unavailable"

            logging.exception(
                "Video action recognition could not be loaded or run"
            )

            return []


def extract_sample_frames(video_path: str, max_frames: int = 16, return_timestamps: bool = False):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to read video file.")

    frames = []
    timestamps = []

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    video_duration = total_frames / fps if fps and fps > 0 else 0.0
    step = max(1, (total_frames + max_frames - 1) // max_frames)

    index = 0

    while len(frames) < max_frames:
        ok, frame = capture.read()

        if not ok:
            break

        if index % step == 0:
            frames.append(frame)
            timestamp = index / fps if fps and fps > 0 else 0.0
            timestamps.append(round(timestamp, 2))

        index += 1

    capture.release()

    if return_timestamps:
        return frames, timestamps, round(video_duration, 2)

    return frames

def annotate_frame(frame, detections, label):
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        pt1 = (int((x1 / 100) * width), int((y1 / 100) * height))
        pt2 = (int((x2 / 100) * width), int((y2 / 100) * height))
        cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{detection['class_name']} {detection['confidence']:.2f}",
            (pt1[0], max(10, pt1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        label.upper(),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def create_summary_video(frames, label, output_path: Path, detections_by_frame=None):
    if not frames:
        return None

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (width, height),
    )

    if not writer.isOpened():
        return None

    for index, frame in enumerate(frames):
        frame_detections = detections_by_frame[index] if detections_by_frame else []
        writer.write(annotate_frame(frame, frame_detections, label))
    writer.release()
    return output_path

def create_full_detection_video(
    video_path: Path,
    output_path: Path,
    label: str,
    detection_interval: int = 10
):
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        logging.error("Could not open uploaded video.")
        return None

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    if width <= 0 or height <= 0:
        capture.release()
        return None

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    process = None
    frame_count = 0
    inference_count = 0
    latest_detections = []

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_count % detection_interval == 0:
                latest_detections = pipeline.run_yolo(frame)
                inference_count += 1

            annotated = annotate_frame(
                frame,
                latest_detections,
                label
            )

            process.stdin.write(annotated.tobytes())
            frame_count += 1

        process.stdin.close()
        return_code = process.wait()

        if return_code != 0:
            logging.error("FFmpeg video encoding failed.")
            output_path.unlink(missing_ok=True)
            return None

        logging.info(
            "H264 detection video completed: %s frames, %s YOLO inferences",
            frame_count,
            inference_count
        )

    except Exception:
        logging.exception("Full-video detection generation failed.")

        if process and process.poll() is None:
            process.kill()
            process.wait()

        output_path.unlink(missing_ok=True)
        raise

    finally:
        capture.release()

    if (
        frame_count == 0
        or not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        output_path.unlink(missing_ok=True)
        return None

    check = cv2.VideoCapture(str(output_path))
    valid = check.isOpened()

    if valid:
        valid, _ = check.read()

    check.release()

    if not valid:
        output_path.unlink(missing_ok=True)
        logging.error("Generated H264 video is invalid.")
        return None

    return output_path

def build_pdf_report(report: dict) -> bytes:
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title="EventLens Analysis Report",
        author="EventLens",
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=29,
        textColor=colors.HexColor("#176B9A"),
        alignment=TA_CENTER,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#176B9A"),
        spaceBefore=14,
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    ))

    def safe(value):
        return escape(str(value if value is not None else "N/A"))

    def heading(text):
        return Paragraph(safe(text), styles["SectionHeading"])

    def cell(text):
        return Paragraph(str(text), styles["ReportBody"])

    def make_table(rows, widths=None):
        table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEFFA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#174D70")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7E2EA")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    title = report.get("report_title") or report.get("title") or "Event Analysis Report"
    event_type = report.get("event_type") or "Unknown"
    summary = report.get("summary") or "No summary available."
    narrative = report.get("narrative") or summary

    model_quality = report.get("model_quality", "unavailable")
    confidence = report.get("confidence")
    event_classification = report.get("event_classification") or {}
    event_classification_status = event_classification.get("classification_status")
    model_score = event_classification.get("score", event_classification.get("model_score"))

    if event_classification.get("source") == "X-CLIP pretrained zero-shot":
        if event_classification_status == "supported":
            classification_status = (
                f"X-CLIP zero-shot score {float(model_score):.3f} (uncalibrated)"
                if model_score is not None else "X-CLIP classification (uncalibrated score unavailable)"
            )
        else:
            classification_status = "Unknown — no supported X-CLIP category assigned"
    elif event_classification_status == "uncertain":
        candidate = event_classification.get("predicted_category")
        if (
            event_classification.get("confidence_calibrated") is True
            and model_score is not None
        ):
            classification_status = (
                "Uncertain — no event category assigned; "
                f"top candidate {candidate or 'N/A'} had calibrated score "
                f"{float(model_score):.3f} below validated threshold"
            )
        else:
            classification_status = "Uncertain — no event category assigned"
            if model_score is not None:
                candidate_prefix = (
                    f"top candidate {candidate} had " if candidate else ""
                )
                classification_status += (
                    f"; {candidate_prefix}raw softmax score "
                    f"{float(model_score):.3f} (uncalibrated)"
                )
    elif event_classification_status == "learned_unknown":
        if (
            event_classification.get("confidence_calibrated") is True
            and confidence is not None
        ):
            classification_status = (
                f"Unknown (out of scope) — calibrated confidence ({float(confidence):.1%})"
            )
        else:
            classification_status = (
                "Unknown — unverified out-of-scope classification"
            )
    elif (
        event_classification.get("confidence_calibrated") is True
        and confidence is not None
    ):
        classification_status = (
            f"Calibrated confidence ({float(confidence):.1%})"
        )
    elif model_quality == "trained" and confidence is not None:
        classification_status = (
            f"Model score {float(confidence):.3f} (not calibrated)"
        )
    else:
        classification_status = "Unclassified — local evidence is unavailable or insufficient"

    story = [
        Spacer(1, 8 * mm),
        Paragraph(safe(title), styles["ReportTitle"]),
        Paragraph(
            "AUTOMATED VIDEO EVENT ANALYSIS",
            ParagraphStyle(
                "Subtitle",
                parent=styles["SmallNote"],
                alignment=TA_CENTER,
                fontSize=9,
            ),
        ),
        Spacer(1, 8 * mm),
        heading("1. Event Overview"),
    ]

    overview = [
        [cell("Field"), cell("Analysis Result")],
        [cell("Video"), cell(report.get("video_name", "N/A"))],
        [cell("Detected event category"), cell(event_type.title())],
        [cell("Classification status"), cell(classification_status)],
        [cell("Peak people detected in one frame"), cell(report.get("peak_person_count", "N/A"))],
        [cell("Unique tracked people"), cell(
            report.get("unique_tracked_people")
            if report.get("unique_tracked_people") is not None
            else "Not reliably available"
        )],
        [cell("Sampled frames"), cell(report.get("sampled_frame_count", "N/A"))],
        [cell("Detections"), cell(report.get("sampled_detection_count", "N/A"))],
        [cell("Video duration (seconds)"), cell(report.get("video_duration", "N/A"))],
    ]

    story.append(make_table(overview, [65 * mm, 105 * mm]))
    story.extend([
        heading("2. Analysis Summary"),
        Paragraph(safe(summary), styles["ReportBody"]),
        heading("3. Detailed Observations"),
        Paragraph(safe(narrative), styles["ReportBody"]),
    ])

    participants = report.get("participants") or []
    story.append(heading("4. Detected Participants"))
    if participants:
        story.append(Paragraph(
            safe(", ".join(participants)),
            styles["ReportBody"],
        ))
    else:
        story.append(Paragraph(
            "No participants could be identified from supported visual evidence.",
            styles["ReportBody"],
        ))

    top_objects = report.get("top_objects") or []

    story.append(heading("5. Detected Objects"))

    if top_objects:
        object_rows = [[cell("Object"), cell("Count")]]
        for item in top_objects:
            object_rows.append([
                cell(item.get("class_name", "Unknown")),
                cell(item.get("count", "N/A")),
            ])
        story.append(make_table(object_rows, [85 * mm, 85 * mm]))
    else:
        story.append(Paragraph(
            "No supported objects were detected in the sampled frames.",
            styles["ReportBody"],
        ))

    action_predictions = report.get("action_predictions") or []

    recognized_activities = report.get("recognized_activities") or []
    story.append(heading("6. Recognized Activities"))
    if recognized_activities:
        story.append(Paragraph(
            safe(", ".join(recognized_activities)),
            styles["ReportBody"],
        ))
    else:
        story.append(Paragraph(
            "No activity could be confirmed from the available evidence.",
            styles["ReportBody"],
        ))

    story.append(heading("7. Supplementary Activity Suggestions"))

    if action_predictions:
        action_rows = [[cell("Suggested action"), cell("Confidence")]]
        for item in action_predictions:
            action_confidence = item.get("confidence")
            confidence_value = (
                f"{float(action_confidence):.1%}"
                if action_confidence is not None else "N/A"
            )
            action_rows.append([
                cell(item.get("action_name", "Unknown")),
                cell(confidence_value),
            ])
        story.append(make_table(action_rows, [110 * mm, 60 * mm]))
        story.append(Paragraph(
            "Action predictions are model suggestions and require human verification.",
            styles["SmallNote"],
        ))
    else:
        story.append(Paragraph(
            "No reliable action predictions were available for this video.",
            styles["ReportBody"],
        ))

    events = report.get("events") or []

    story.append(heading("8. Activity Timeline"))

    if events:
        timeline_rows = [[
            cell("Activity"),
            cell("Start (s)"),
            cell("End (s)"),
            cell("Confidence"),
        ]]

        for event in events:
            event_confidence = event.get("confidence")
            confidence_value = (
                f"{float(event_confidence):.1%}"
                if event_confidence is not None else "N/A"
            )

            timeline_rows.append([
                cell(event.get("activity_name", "Unknown")),
                cell(event.get("start_seconds", "N/A")),
                cell(event.get("end_seconds", "N/A")),
                cell(confidence_value),
            ])

        story.append(make_table(
            timeline_rows,
            [65 * mm, 30 * mm, 30 * mm, 45 * mm],
        ))
        story.append(Paragraph(
            "Timeline entries represent X-CLIP sliding-window model suggestions; "
            "they are not independently verified event boundaries.",
            styles["SmallNote"],
        ))
    else:
        story.append(Paragraph(
            "No timeline segments were generated.",
            styles["ReportBody"],
        ))

    story.extend([
        heading("7. Limitations and Disclaimer"),
        Paragraph(
            "This report is generated automatically from sampled video frames "
            "and available computer-vision models. Object detections do not by "
            "themselves confirm an event category. The peak people count is a "
            "per-frame estimate, not a count of unique attendees. Action "
            "recognition results are suggestions and may require manual review. "
            "Please review the original video before relying on these findings.",
            styles["ReportBody"],
        ),
    ])

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setSubject(narrative)
        canvas.setStrokeColor(colors.HexColor("#D7E2EA"))
        canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(18 * mm, 10 * mm, "EventLens | Automated Video Analysis")
        canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    return buffer.getvalue()


def generate_report_narrative(
    event_result,
    detections,
    frame_count,
    action_predictions=None,
    video_duration=None,
    local_evidence=None,
    action_reporting_message=None,
):
    observed_objects = event_result.get("top_objects", [])
    if observed_objects:
        object_summary = ", ".join(
            f'{item["count"]} {item["class_name"]}(s)'
            for item in observed_objects
        )
        observation = f"The highest per-frame object counts were {object_summary}."
    else:
        observation = "No supported objects were detected in the sampled frames."

    event_classification = event_result.get("event_classification") or {}
    status = event_classification.get("classification_status")
    score = event_classification.get("score")
    category = (
        event_classification.get("event_category")
        or event_classification.get("activity_label")
        or event_result.get("event_type", "unknown")
    )

    if status == "supported":
        score_text = (
            f" Its raw zero-shot score was {float(score):.3f} (uncalibrated)."
            if score is not None else ""
        )
        confidence_text = (
            f'The pretrained X-CLIP model classified the video as "{category}".'
            f"{score_text}"
        )
    else:
        confidence_text = (
            "The pretrained X-CLIP model could not assign this video to a supported "
            "event category. The event is reported as Unknown."
        )

    duration_text = (
        f"The uploaded video duration was approximately {video_duration:.1f} seconds. "
        if video_duration is not None else ""
    )
    action_text = action_reporting_message or (
        "Activity timeline segments were generated from X-CLIP sliding-window predictions."
    )

    return (
        f'{event_result.get("summary", "Video analysis completed.")} '
        f"{duration_text}The analysis sampled {frame_count} frame(s). "
        f"{observation} {confidence_text} {action_text} "
        "This is an automated estimate based on sampled video frames and model predictions. "
        "The visible-person figure is a peak per-frame count, not a count of unique attendees. "
        "X-CLIP zero-shot scores are not calibrated probabilities. Review the original video "
        "before relying on these results."
    )


pipeline = EventPipeline()
xclip_classifier = XCLIPEventClassifier()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_BYTES", 250 * 1024 * 1024))
DEFAULT_CORS_ORIGINS = (
    "https://eventlensoffialorg-lgtm.github.io,"
    "https://manaswi-21.github.io,"
    "http://localhost:5173,http://127.0.0.1:5173"
)
CORS_ORIGINS = list(dict.fromkeys(
    origin.strip()
    for origin in (
        DEFAULT_CORS_ORIGINS + "," + os.environ.get("CORS_ORIGINS", "")
    ).split(",")
    if origin.strip()
))
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})
app.register_blueprint(auth_bp, url_prefix="/api/auth")


def save_report(report, user_id):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reports (id, user_id, created_at, report_json)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    created_at = IF(user_id = VALUES(user_id), VALUES(created_at), created_at),
                    report_json = IF(user_id = VALUES(user_id), VALUES(report_json), report_json)
                """,
                (
                    report["id"],
                    user_id,
                    report["created_at"],
                    json.dumps(report),
                ),
            )


def get_user_reports(user_id):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT report_json
                FROM reports
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            rows = cursor.fetchall()

    return [json.loads(row["report_json"]) for row in rows]


def get_user_report(user_id, report_id):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT report_json
                FROM reports
                WHERE user_id = %s AND id = %s
                """,
                (user_id, report_id),
            )
            row = cursor.fetchone()

    return json.loads(row["report_json"]) if row else None


def require_auth(handler):
    @wraps(handler)
    def wrapped(*args, **kwargs):
        authorization = request.headers.get("Authorization", "")

        if not authorization.startswith("Bearer "):
            return jsonify({"error": "Authentication is required."}), 401

        token = authorization.split(" ", 1)[1].strip()

        if not token:
            return jsonify({"error": "Invalid authentication token."}), 401

        token_hash = hashlib.sha256(token.encode()).hexdigest()

        try:
            with get_db_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT users.id, users.full_name, users.email
                        FROM sessions
                        JOIN users ON sessions.user_id = users.id
                        WHERE sessions.token_hash = %s
                        AND sessions.expires_at > NOW()
                        AND users.email_verified = TRUE
                        """,
                        (token_hash,),
                    )

                    user = cursor.fetchone()

        except DatabaseUnavailableError:
            return jsonify({"error": "Authentication service is unavailable."}), 503

        if not user:
            return jsonify({"error": "Your session is invalid or expired."}), 401

        g.user_id = user["id"]
        g.user = user

        return handler(*args, **kwargs)

    return wrapped

database_schema_initialized = False
database_initialization_lock = threading.Lock()


def ensure_database_initialized():
    global database_schema_initialized
    if database_schema_initialized:
        return True

    with database_initialization_lock:
        if database_schema_initialized:
            return True
        initialize_mysql_database()
        database_schema_initialized = True
    return True


try:
    ensure_database_initialized()
except (DatabaseConfigurationError, DatabaseUnavailableError):
    logging.warning(
        "Database initialization was deferred; check database connectivity and configuration."
    )


@app.before_request
def initialize_database_when_available():
    if (
        request.method == "OPTIONS"
        or request.path in {"/api/health", "/api/health/database"}
    ):
        return None

    try:
        ensure_database_initialized()
    except (DatabaseConfigurationError, DatabaseUnavailableError):
        return jsonify({
            "error": "Database service is temporarily unavailable."
        }), 503
    return None


@app.errorhandler(DatabaseUnavailableError)
def database_unavailable(_error):
    return jsonify({"error": "Database service is temporarily unavailable."}), 503


@app.errorhandler(DatabaseConfigurationError)
def database_configuration_error(_error):
    return jsonify({"error": "Database service is not configured."}), 503


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "The video exceeds the configured upload size limit."}), 413


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/health/database", methods=["GET"])
def database_health_check():
    try:
        verify_required_tables()
    except (DatabaseConfigurationError, DatabaseUnavailableError):
        return jsonify({
            "status": "error",
            "database": "unavailable",
        }), 503

    return jsonify({
        "status": "ok",
        "database": "connected",
        "schema": "ready",
    })


@app.route("/api/media/<path:filename>", methods=["GET"])
@require_auth
def serve_media(filename):
    report_media = any(
        f"/api/media/{filename}" in (report.get("annotated_frame_url") or "")
        or f"/api/media/{filename}" in (report.get("generated_video_url") or "")
        for report in get_user_reports(g.user_id)
    )
    if not report_media:
        return jsonify({"error": "Media not found."}), 404
    return send_from_directory(GENERATED_DIR, filename)

@app.route("/api/original/<path:filename>", methods=["GET"])
@require_auth
def serve_original_video(filename):
    user_reports = get_user_reports(g.user_id)

    owns_video = any(
        report.get("original_video_file") == filename
        for report in user_reports
    )

    if not owns_video:
        return jsonify({"error": "Video not found."}), 404

    return send_from_directory(UPLOAD_DIR, filename)


def update_analysis_job(job_id, status, stage, progress=None, result=None, error=None):
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
        if job is None:
            return
        job["status"] = status
        job["stage"] = stage
        if progress is not None:
            job["progress"] = progress
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error


def process_analysis_job(job_id, user_id, video_path, safe_name):
    try:
        action_predictions = []
        action_clip_details = []
        action_timeline = []

        update_analysis_job(job_id, "processing", "Extracting frames", 10)
        frames, frame_timestamps, video_duration = extract_sample_frames(
            str(video_path),
            max_frames=16,
            return_timestamps=True,
        )

        update_analysis_job(job_id, "processing", "Running object detection", 30)
        frame_detections = [pipeline.run_yolo(frame) for frame in frames]
        detections = [
            item
            for frame_items in frame_detections
            for item in frame_items
        ]
        event_result = pipeline.process_detections(detections, frame_detections)

        update_analysis_job(
            job_id,
            "processing",
            "Classifying event with pretrained X-CLIP",
            55,
        )
        xclip_prediction = xclip_classifier.predict_video(str(video_path))
        event_result["event_classification"] = xclip_prediction

        classification_status = xclip_prediction.get("classification_status")
        allowed_categories = {"cultural", "dance", "presentation"}
        predicted_category = (
            xclip_prediction.get("event_category")
            or xclip_prediction.get("activity_label")
            or "unknown"
        )
        normalized_category = str(predicted_category).strip().lower()

        # Preserve the strict four-label EventLens output vocabulary.
        if classification_status == "supported" and normalized_category in allowed_categories:
            event_result["event_type"] = normalized_category
            raw_score = xclip_prediction.get("score")
            event_result["confidence"] = (
                round(float(raw_score), 4) if raw_score is not None else None
            )
            event_result["model_quality"] = "pretrained_zero_shot_xclip"
            event_result["summary"] = (
                f'The pretrained X-CLIP model classified this video as "{predicted_category}". '
                f'The peak number of visible person detections in a sampled frame was '
                f'{event_result.get("peak_person_count", 0)}. Object detections are '
                "reported separately and do not independently establish the event category."
            )
            action_reporting_status = "supported"
            action_reporting_message = (
                "Timeline segments are generated from X-CLIP sliding-window predictions. "
                "Displayed scores are uncalibrated zero-shot scores, not probabilities."
            )
        else:
            event_result["event_type"] = "unknown"
            event_result["confidence"] = None
            event_result["model_quality"] = "pretrained_zero_shot_xclip"
            event_result["summary"] = (
                "The video could not be confidently assigned to Cultural, Dance, or "
                "Presentation by the pretrained X-CLIP model. It is classified as Unknown."
            )
            action_reporting_status = "unavailable"
            action_reporting_message = (
                "X-CLIP did not assign a supported event category. Any displayed timeline "
                "segments are model suggestions and should be reviewed against the video."
            )

        # Convert X-CLIP's actual sliding-window timestamps to the exact schema
        # consumed by EventTimeline.jsx and the PDF timeline table.
        window_predictions = xclip_prediction.get("window_predictions") or []
        action_clip_details = window_predictions
        for window in window_predictions:
            try:
                start_seconds = float(window.get("start", 0))
                end_seconds = float(window.get("end", 0))
            except (TypeError, ValueError):
                continue
            if end_seconds <= start_seconds:
                continue

            activity = (
                window.get("category")
                or window.get("activity_label")
                or "Unknown"
            )
            score = window.get("score")
            try:
                score = float(score) if score is not None else None
            except (TypeError, ValueError):
                score = None

            action_timeline.append({
                "activity_name": str(activity),
                "label": str(activity),
                "start_seconds": round(start_seconds, 2),
                "end_seconds": round(end_seconds, 2),
                "confidence": score,
                "model_score": score,
                "score_type": "uncalibrated zero-shot softmax",
                "source": "xclip-pretrained-zero-shot",
                "evidence": "Pretrained X-CLIP sliding-window video prediction.",
            })

        logging.info("X-CLIP classification status: %s", classification_status)
        logging.info("X-CLIP event category: %s", event_result["event_type"])
        logging.info("X-CLIP timeline segment count: %s", len(action_timeline))

        update_analysis_job(
            job_id,
            "processing",
            "Tracking visible people",
            62,
        )
        tracking_result = pipeline.track_people(frames)

        update_analysis_job(job_id, "processing", "Generating event report", 70)
        narrative = generate_report_narrative(
            event_result,
            detections,
            len(frames),
            action_predictions,
            video_duration,
            tracking_result,
            action_reporting_message=action_reporting_message,
        )

        annotated_name = None
        if frames:
            annotated_name = f"{uuid.uuid4()}.jpg"
            annotated_path = GENERATED_DIR / annotated_name
            annotated = annotate_frame(
                frames[0],
                frame_detections[0] if frame_detections else [],
                event_result["event_type"],
            )
            if not cv2.imwrite(str(annotated_path), annotated):
                annotated_name = None

        update_analysis_job(job_id, "processing", "Generating annotated video", 82)
        output_name = f"{uuid.uuid4()}.mp4"
        output_path = GENERATED_DIR / output_name
        generated_video = create_full_detection_video(
            video_path,
            output_path,
            event_result["event_type"],
        )
        generated_url = (
            f"/api/media/{generated_video.name}"
            if generated_video is not None else None
        )

        events = action_timeline
        report = {
            "id": job_id,
            "report_title": f'{event_result["event_type"].title()} event report',
            "video_name": safe_name,
            "event_type": event_result["event_type"],
            "summary": event_result["summary"],
            "narrative": narrative,
            "confidence": event_result["confidence"],
            "model_quality": event_result["model_quality"],
            "event_classification": event_result.get("event_classification"),
            "peak_person_count": event_result["peak_person_count"],
            "unique_tracked_people": tracking_result["unique_tracked_people"],
            "peak_tracked_people": tracking_result["peak_tracked_people"],
            "tracking_status": tracking_result["tracking_status"],
            "sampled_frame_count": len(frames),
            "video_duration": video_duration,
            "sampled_detection_count": len(detections),
            "detections": frame_detections[0] if frame_detections else [],
            "top_objects": event_result["top_objects"],
            "action_predictions": action_predictions,
            "action_clip_details": action_clip_details,
            "action_timeline": action_timeline,
            "action_summary": action_reporting_message,
            "action_reporting_status": action_reporting_status,
            "action_model_status": classification_status or "unavailable",
            "original_video_file": video_path.name,
            "original_video_url": f"/api/original/{video_path.name}",
            "events": events,
            "annotated_frame_url": (
                f"/api/media/{annotated_name}" if annotated_name else None
            ),
            "generated_video_url": generated_url,
            "created_at": datetime.utcnow().isoformat(),
            "status": "completed",
        }
        save_report(report, user_id)

        response_result = {
            "annotated_frame_url": report["annotated_frame_url"],
            "detections": report["detections"],
            "events": report["events"],
            "summary": report["summary"],
            "narrative": narrative,
            "event_type": report["event_type"],
            "confidence": report["confidence"],
            "model_quality": report["model_quality"],
            "event_classification": report.get("event_classification"),
            "peak_person_count": report["peak_person_count"],
            "unique_tracked_people": report["unique_tracked_people"],
            "peak_tracked_people": report["peak_tracked_people"],
            "tracking_status": report["tracking_status"],
            "sampled_frame_count": report["sampled_frame_count"],
            "sampled_detection_count": report["sampled_detection_count"],
            "top_objects": report["top_objects"],
            "action_predictions": report["action_predictions"],
            "action_clip_details": report["action_clip_details"],
            "action_timeline": report["action_timeline"],
            "action_summary": report["action_summary"],
            "action_reporting_status": report["action_reporting_status"],
            "action_model_status": report["action_model_status"],
            "generated_video_url": generated_url,
            "original_video_url": report["original_video_url"],
        }
        update_analysis_job(
            job_id,
            "completed",
            "Completed",
            100,
            {"result": response_result, "report_id": job_id, "report": report},
        )
    except Exception as exc:
        logging.exception("Video analysis job failed")
        update_analysis_job(
            job_id,
            "failed",
            "Analysis failed",
            0,
            error=str(exc),
        )


@app.route("/api/analyze", methods=["POST"])
@require_auth

def analyze_video_route():
    if "video" not in request.files:
        return jsonify({"error": "No video file was included in the request."}), 400

    uploaded = request.files["video"]
    if uploaded.filename == "":
        return jsonify({"error": "No file was selected for analysis."}), 400

    safe_name = secure_filename(uploaded.filename)
    if not safe_name:
        return jsonify({"error": "The uploaded file name is invalid."}), 400
    if Path(safe_name).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": "Unsupported video format."}), 400

    saved_name = f"{uuid.uuid4()}_{safe_name}"
    video_path = UPLOAD_DIR / saved_name
    uploaded.save(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        video_path.unlink(missing_ok=True)
        return jsonify({"error": "The uploaded file could not be read as a video."}), 400
    capture.release()

    job_id = str(uuid.uuid4())
    with analysis_jobs_lock:
        analysis_jobs[job_id] = {
            "id": job_id,
            "user_id": g.user_id,
            "status": "processing",
            "stage": "Extracting frames",
            "progress": 5,
        }

    worker = threading.Thread(
        target=process_analysis_job,
        args=(job_id, g.user_id, video_path, safe_name),
        daemon=True,
        name=f"eventlens-analysis-{job_id}",
    )
    worker.start()

    return jsonify({
        "status": "processing",
        "stage": "Extracting frames",
        "progress": 5,
        "report_id": job_id,
        "job_id": job_id,
    }), 202
    

@app.route("/api/reports", methods=["GET"])
@require_auth
def list_reports():
    return jsonify(get_user_reports(g.user_id))


@app.route("/api/reports/<report_id>", methods=["GET"])
@require_auth
def get_report(report_id):
    report = get_user_report(g.user_id, report_id)
    if report:
        return jsonify(report)
    return jsonify({"error": "Report not found"}), 404


@app.route("/api/reports/<report_id>/pdf", methods=["GET"])
@require_auth
def get_report_pdf(report_id):
    report = get_user_report(g.user_id, report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    return send_file(
        io.BytesIO(build_pdf_report(report)),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{report_id}.pdf",
    )


@app.route("/api/analyses", methods=["POST"])
@require_auth
def create_analysis_entry():
    payload = request.get_json(silent=True) or {}
    analysis_data = {
        "id": payload.get("id", str(uuid.uuid4())),
        "video_name": payload.get("video_name", "video.mp4"),
        "created_at": payload.get("created_at", datetime.utcnow().isoformat()),
        "status": payload.get("status", "completed"),
        "event_type": payload.get("event_type", "unknown"),
        "summary": payload.get("summary", "Analysis completed."),
    }
    save_report(analysis_data, g.user_id)
    return jsonify(analysis_data)


@app.route("/api/analyses/<analysis_id>", methods=["GET"])
@require_auth
def get_analysis_entry(analysis_id):
    with analysis_jobs_lock:
        job = analysis_jobs.get(analysis_id)
    if job and job["user_id"] == g.user_id:
        response = {
            "id": analysis_id,
            "status": job["status"],
            "stage": job["stage"],
            "progress": job["progress"],
        }
        if job["status"] == "completed":
            response.update(job.get("result") or {})
        if job["status"] == "failed":
            response["error"] = job.get("error") or "Analysis failed."
        return jsonify(response)

    record = get_user_report(g.user_id, analysis_id)
    if record:
        return jsonify(record)
    return jsonify({"error": "Analysis not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
