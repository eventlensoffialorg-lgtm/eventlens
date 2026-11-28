import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np


MOVINET_URL = (
    "https://tfhub.dev/tensorflow/movinet/"
    "a0/base/kinetics-600/classification/3"
)

LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/"
    "official/projects/movinet/files/kinetics_600_labels.txt"
)

FRAME_COUNT = 50
IMAGE_SIZE = 172

# Only map activities when the predicted label provides clear evidence.
DANCE_PATTERNS = (
    "dancing",
    "dance",
    "ballet",
    "breakdancing",
    "tap dancing",
)

PRESENTATION_PATTERNS = (
    "giving a speech",
    "lecturing",
    "presenting",
    "giving a lecture",
)


class MoViNetClassifier:

    def __init__(self):
        self.model = None
        self.inference = None
        self.labels = None
        self.enabled = True
        self.load_error = None

    def _load(self):
        if self.inference is not None:
            return True

        if self.load_error:
            return False

        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            logging.info("Loading pretrained MoViNet A0...")

            self.model = hub.load(MOVINET_URL)
            self.inference = self.model.signatures["serving_default"]

            label_path = tf.keras.utils.get_file(
                fname="kinetics_600_labels.txt",
                origin=LABELS_URL,
                cache_subdir="eventlens",
            )

            with open(label_path, "r", encoding="utf-8") as file:
                self.labels = [
                    line.strip().strip('"')
                    for line in file
                    if line.strip()
                ]

            if len(self.labels) != 600:
                raise ValueError(
                    f"Expected 600 labels, received {len(self.labels)}"
                )

            logging.info("MoViNet A0 loaded successfully.")
            return True

        except Exception as exc:
            self.load_error = str(exc)
            logging.exception("MoViNet initialization failed.")
            return False

    def _extract_video(self, video_path):

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError("Unable to open uploaded video.")

        try:
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if total_frames <= 0:
                raise ValueError("Video contains no readable frames.")

            indices = np.linspace(
                0,
                total_frames - 1,
                min(FRAME_COUNT, total_frames),
                dtype=int,
            )

            frames = []

            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, frame = cap.read()

                if not success or frame is None:
                    continue

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frame = cv2.resize(
                    frame,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_AREA,
                )

                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            if not frames:
                raise ValueError("Could not extract video frames.")

            # Repeat the last frame if fewer than 50 were readable.
            while len(frames) < FRAME_COUNT:
                frames.append(frames[-1].copy())

            return np.asarray(
                frames[:FRAME_COUNT],
                dtype=np.float32,
            )

        finally:
            cap.release()

    @staticmethod
    def _map_category(activity_label):

        label = re.sub(
            r"[^a-z0-9 ]",
            " ",
            activity_label.lower(),
        )

        label = " ".join(label.split())

        if any(pattern in label for pattern in DANCE_PATTERNS):
            return "Dance"

        if any(pattern in label for pattern in PRESENTATION_PATTERNS):
            return "Presentation"

        # A generic activity label is not enough to establish
        # that the video depicts a Cultural event.
        return "Unknown"

    def predict_video(self, video_path):

        if not self._load():
            return {
                "classification_status": "unavailable",
                "message": (
                    "MoViNet could not be loaded: "
                    f"{self.load_error}"
                ),
            }

        try:
            import tensorflow as tf

            video = self._extract_video(video_path)

            input_tensor = tf.convert_to_tensor(
                video[np.newaxis, ...],
                dtype=tf.float32,
            )

            outputs = self.inference(image=input_tensor)

            logits = outputs["classifier_head"]
            probabilities = tf.nn.softmax(
                logits,
                axis=-1,
            ).numpy()[0]

            top_indices = np.argsort(probabilities)[-5:][::-1]

            top_activities = []

            for index in top_indices:
                top_activities.append({
                    "label": self.labels[int(index)],
                    "score": float(probabilities[index]),
                })

            best = top_activities[0]
            activity_label = best["label"]
            score = best["score"]

            category = self._map_category(activity_label)

            if category == "Unknown":
                status = "unsupported"
            else:
                status = "supported"

            return {
                "classification_status": status,
                "status": status,
                "source": "MoViNet A0 - Kinetics-600",
                "model_name": "MoViNet A0",
                "predicted_category": category,
                "assigned_category": category,
                "activity_label": activity_label,
                "top_activities": top_activities,
                "model_score": score,
                "confidence": score,
                "confidence_calibrated": False,
                "score_type": "uncalibrated Kinetics-600 softmax score",
                "message": (
                    "General activity prediction from pretrained "
                    "MoViNet. EventLens category mapping is "
                    "conservative and is not custom-trained."
                ),
            }

        except Exception as exc:
            logging.exception("MoViNet video prediction failed.")

            return {
                "classification_status": "unavailable",
                "message": str(exc),
            }


def apply_movinet_result(event_result, prediction):

    status = prediction.get("classification_status")

    event_result["event_classification"] = prediction

    event_result["model_quality"] = "pretrained_general_action"

    if status == "supported":
        category = prediction["assigned_category"]

        event_result["event_type"] = category
        event_result["confidence"] = prediction["model_score"]

    else:
        event_result["event_type"] = "unknown"
        event_result["confidence"] = prediction.get(
            "model_score",
            0.0,
        )

    return event_result