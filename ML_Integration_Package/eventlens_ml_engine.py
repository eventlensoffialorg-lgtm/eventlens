# =========================================================================
# EVENTLENS - PRODUCTION VIDEO EVENT CLASSIFICATION ENGINE
# Phased implementation: Preprocessing, Feature Extraction & MLP Inference
# =========================================================================

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model

class EventLensMLEngine:
    """
    Core ML integration engine for EventLens production backend.
    """
    def __init__(
        self,
        model_assets_dir,
        confidence_threshold=0.75,
        model_filename="eventlens_mlp.keras",
    ):
        self.assets_dir = model_assets_dir
        self.confidence_threshold = confidence_threshold

        # Paths
        self.model_path = os.path.join(model_assets_dir, model_filename)
        self.mapping_path = os.path.join(model_assets_dir, "class_mapping.json")
        self.model_config_path = os.path.join(model_assets_dir, "model_config.json")
        self.feature_config_path = os.path.join(model_assets_dir, "feature_extraction_config.json")

        # Load configurations
        with open(self.mapping_path, "r") as f:
            self.mapping_data = json.load(f)
            # We map active model indices to category strings
            self.active_classes = {v: k for k, v in self.mapping_data["active_model_classes"].items()}
            self.all_classes = self.mapping_data["all_classes"]

        with open(self.feature_config_path, "r") as f:
            self.feature_config = json.load(f)

        # Verify expected MLP input shape is 5120
        self.expected_dim = self.feature_config["expected_feature_dimension"]
        if self.expected_dim != 5120:
            raise ValueError(f"Feature dimensions misaligned! Expected 5120, got {self.expected_dim}")

        # Initialize Models (load once to prevent memory leaks)
        print("[+] Loading MLP classifier model...")
        self.mlp_model = load_model(self.model_path)

        print("[+] Loading MobileNetV2 spatial backbone...")
        self.backbone = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(160, 160, 3))
        self.backbone.trainable = False

        print("[+] EventLens ML Engine successfully initialized.")

    def extract_features(self, video_path):
        """
        Uniformly sample 8 frames, resize to 160x160, preprocess via MobileNetV2,
        and construct the 5120-D representation.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "unreadable_source"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 8:
            cap.release()
            return None, "insufficient_frames"

        # Uniform sampling (8 frames)
        frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        frames_batch = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()
            if not success or frame is None:
                continue

            # Convert BGR to RGB, resize to 160x160
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (160, 160))
            # Standard MobileNetV2 pre-processing [-1, 1]
            frame_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(frame_resized)
            frames_batch.append(frame_normalized)

        cap.release()

        if len(frames_batch) < 4:
            return None, "unreadable_frames_threshold"

        frames_array = np.array(frames_batch, dtype=np.float32)

        # Spatial pooling
        spatial_vectors = self.backbone.predict(frames_array, batch_size=len(frames_batch), verbose=0)

        # Compute spatial mean & standard deviation
        spatial_mean = np.mean(spatial_vectors, axis=0)
        spatial_std = np.std(spatial_vectors, axis=0)

        # Compute temporal consecutive difference statistics
        if len(spatial_vectors) > 1:
            temporal_diffs = np.diff(spatial_vectors, axis=0)
            temporal_mean = np.mean(temporal_diffs, axis=0)
            temporal_std = np.std(temporal_diffs, axis=0)
        else:
            temporal_mean = np.zeros_like(spatial_mean)
            temporal_std = np.zeros_like(spatial_mean)

        # Concatenate: Spatial Mean + Spatial Std + Temporal Mean + Temporal Std
        combined_vector = np.concatenate([spatial_mean, spatial_std, temporal_mean, temporal_std], axis=0)
        return combined_vector, "success"

    def predict_video(self, video_path):
        """
        Generates predictions and evaluates thresholds securely.
        """
        features, status = self.extract_features(video_path)
        if status != "success":
            return {
                "status": "error",
                "message": f"Feature extraction failed: {status}"
            }

        # Perform MLP Inference
        input_tensor = np.expand_dims(features, axis=0) # shape: (1, 5120)
        prediction_distribution = self.mlp_model.predict(input_tensor, verbose=0)[0]

        predicted_idx = int(np.argmax(prediction_distribution))
        confidence = float(prediction_distribution[predicted_idx])
        raw_class_string = self.active_classes[predicted_idx]

        # Conservative threshold and classification boundaries handling
        is_low_confidence = confidence < self.confidence_threshold

        # Determine final event classification output safely
        if is_low_confidence:
            final_event = "Unknown"
            assessment = "uncertain / low confidence"
        else:
            final_event = raw_class_string
            assessment = "supported"

        return {
            "status": "success",
            "predicted_event_category": final_event,
            "confidence_score": round(confidence, 4),
            "raw_probabilities": {self.active_classes[i]: float(p) for i, p in enumerate(prediction_distribution)},
            "assessment_level": assessment,
            "features_dimension": len(features)
        }
