# analyzer.py
# =========================================================================
# EVENTLENS - PRODUCTION ACTION & HIGH-LEVEL EVENT ANALYZER
# =========================================================================

import os
import cv2
import torch
import numpy as np
import pandas as pd
import joblib
from torchvision.models.video import r3d_18, R3D_18_Weights

class EventLensAnalyzer:
    """
    Core ML integration engine for EventLens.
    Responsibility: loading pretrained R3D-18 video-feature extractor,
    loading custom trained action classification SVM pipeline,
    segmenting long-form videos, and mapping detected evidence to events.
    """
    def __init__(self, model_path, device=None):
        # 1. Determine execution device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2. Load the trained Action Classifier pipeline (Scaler + SVM)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained action classifier not found at: {model_path}")
        self.classifier = joblib.load(model_path)
        self.classes_ = self.classifier.classes_ if hasattr(self.classifier, 'classes_') else []

        # 3. Load the pre-trained feature extractor
        self.weights = R3D_18_Weights.KINETICS400_V1
        self.extractor = r3d_18(weights=self.weights)
        self.extractor.fc = torch.nn.Identity()  # Strip final classification layer
        self.extractor = self.extractor.to(self.device).eval()
        self.transform = self.weights.transforms()

    def extract_segment_features(self, cap, start_idx, step_frames):
        """
        Uniformly sample 16 frames within a given segment and extract its 512-D vector representation.
        """
        frame_offsets = np.linspace(0, step_frames - 1, 16, dtype=int)
        frames = []

        for offset in frame_offsets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_idx + offset))
            ret, frame = cap.read()
            if not ret:
                break
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        if len(frames) < 16:
            return None

        # Convert frames to [T, H, W, C]
        video_array = np.stack(frames)

        # Permute to [T, C, H, W]
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2)

        # Apply kinetics standard transform
        video_tensor = self.transform(video_tensor)

        # Re-arrange to [C, T, H, W] if layout is [T, C, H, W]
        if video_tensor.shape[0] == 16 and video_tensor.shape[1] == 3:
            video_tensor = video_tensor.permute(1, 0, 2, 3)

        # Add batch dimension -> [1, C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features = self.extractor(video_tensor).cpu().numpy()

        return features

    def analyze_video(self, video_path, num_segments=10):
        """
        Analyze a video file of arbitrary duration (7 to 20 minutes).
        Segments the video and reports true temporal evidence mapped to events.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "status": "error",
                "message": f"Failed to open or decode video file: {os.path.basename(video_path)}"
            }

        # 1. Gather file specifications
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / fps if fps > 0 else 0

        # Calculate window boundary steps
        step_frames = max(1, total_frames // num_segments)
        segment_results = []

        # 2. Iterate through each window and extract segments
        for i in range(num_segments):
            start_idx = i * step_frames
            segment_start_sec = start_idx / fps if fps > 0 else 0
            segment_end_sec = (start_idx + step_frames) / fps if fps > 0 else 0

            try:
                features = self.extract_segment_features(cap, start_idx, step_frames)
                if features is None:
                    continue

                # Make classifier prediction
                predicted_action = self.classifier.predict(features)[0]

                # Obtain probability / confidence margins where available
                if hasattr(self.classifier, "predict_proba"):
                    probs = self.classifier.predict_proba(features)[0]
                    confidence = float(np.max(probs))
                else:
                    # LinearSVC uses decision_function, normalize scores as a Softmax proxy
                    decision_scores = self.classifier.decision_function(features)[0]
                    exp_scores = np.exp(decision_scores - np.max(decision_scores))
                    confidence = float(np.max(exp_scores / np.sum(exp_scores)))

                segment_results.append({
                    "segment_index": i,
                    "start_time_seconds": round(segment_start_sec, 2),
                    "end_time_seconds": round(segment_end_sec, 2),
                    "detected_action": predicted_action,
                    "confidence": round(confidence, 4)
                })
            except Exception:
                # Catch inner runtime exceptions safely to avoid halting complete evaluation
                continue

        cap.release()

        if not segment_results:
            return {
                "status": "error",
                "message": "Could not extract valid analytical features from any segments."
            }

        # 3. Safe High-level Event Determination Heuristics
        action_series = pd.Series([s["detected_action"] for s in segment_results])
        action_distribution = action_series.value_counts().to_dict()

        determined_event = "Unknown / Not Supported"

        # Logical multi-window mapping logic
        if action_distribution.get("WritingOnBoard", 0) >= 3:
            determined_event = "Presentation / Lecture"
        elif any(act in action_distribution for act in ["PlayingGuitar", "PlayingPiano", "PlayingViolin", "PlayingTabla"]) and sum(action_distribution.get(a, 0) for a in ["PlayingGuitar", "PlayingPiano", "PlayingViolin", "PlayingTabla"]) >= 2:
            determined_event = "Singing / Instrumental Performance"
        elif any(act in action_distribution for act in ["SalsaSpin", "IceDancing"]) and sum(action_distribution.get(a, 0) for a in ["SalsaSpin", "IceDancing"]) >= 2:
            determined_event = "Dance Performance"
        elif any(act in action_distribution for act in ["BandMarching", "MilitaryParade"]) and sum(action_distribution.get(a, 0) for a in ["BandMarching", "MilitaryParade"]) >= 2:
            determined_event = "Cultural Programme / Parade"

        return {
            "status": "success",
            "video_metadata": {
                "filename": os.path.basename(video_path),
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_seconds": round(duration_sec, 2)
            },
            "mapped_high_level_event": determined_event,
            "detected_actions_summary": action_distribution,
            "timeline_analysis": segment_results
        }
