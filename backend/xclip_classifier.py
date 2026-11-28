
import logging

import cv2
import numpy as np
import torch

from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor


MODEL_ID = "microsoft/xclip-base-patch16-zero-shot"

CANDIDATES = {
    "Cultural": (
        "A cultural celebration, traditional ceremony, folk performance, "
        "traditional music festival or cultural event."
    ),
    "Dance": (
        "People performing dance, traditional dance, group choreography, "
        "folk dance, ballet or coordinated dance movements."
    ),
    "Presentation": (
        "A person giving a presentation, delivering a speech, explaining "
        "slides, lecturing or addressing an audience on a stage."
    ),
    "Other": (
        "Ordinary daily activities, conferences, meetings, walking, sitting, "
        "working, sports or scenes unrelated to cultural events, dance "
        "performances or presentations."
    ),
}


class XCLIPEventClassifier:

    def __init__(self, window_seconds=4.0, stride_seconds=2.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)

        self.processor = None
        self.model = None

    def load_model(self):
        if self.model is not None:
            return

        logging.info("Loading X-CLIP model: %s", MODEL_ID)

        self.processor = XCLIPProcessor.from_pretrained(MODEL_ID)

        self.model = XCLIPModel.from_pretrained(MODEL_ID)

        self.model.to(self.device)
        self.model.eval()

        logging.info("X-CLIP loaded successfully on %s", self.device)

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError("Unable to open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        if fps <= 0 or total_frames <= 0:
            raise RuntimeError("Invalid video frame rate or duration.")

        duration = total_frames / fps

        return fps, total_frames, duration

    def extract_frames(self, video_path, start, end):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError("Unable to open video.")

        timestamps = np.linspace(
            start,
            end,
            32,
            endpoint=False
        )

        frames = []

        for timestamp in timestamps:

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                float(timestamp) * 1000
            )

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(Image.fromarray(frame))

        cap.release()

        if len(frames) < 8:
            return []

        # Restore 32 frames if some video frames could not be read.
        if len(frames) != 32:

            indices = np.linspace(
                0,
                len(frames) - 1,
                32
            ).astype(int)

            frames = [
                frames[index]
                for index in indices
            ]

        return frames

    def predict_window(self, frames):

        self.load_model()

        descriptions = list(CANDIDATES.values())

        # Process video frames.
        image_inputs = self.processor.image_processor(
            images=frames,
            return_tensors="pt"
        )

        # Process text descriptions.
        text_inputs = self.processor.tokenizer(
            descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Convert the image processor output into the exact
        # five-dimensional video tensor required by X-CLIP.
        pixel_values = image_inputs["pixel_values"]

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)

        elif pixel_values.ndim == 6:
            if pixel_values.shape[1] == 1:
                pixel_values = pixel_values.squeeze(1)
            elif pixel_values.shape[0] == 1:
                pixel_values = pixel_values.squeeze(0)

        if pixel_values.ndim != 5:
            raise RuntimeError(
                "Invalid X-CLIP video tensor shape: "
                f"{tuple(pixel_values.shape)}"
            )

        if pixel_values.shape[1] != 32:
            raise RuntimeError(
                "X-CLIP requires 32 video frames. "
                f"Received {pixel_values.shape[1]}."
            )

        inputs = {
            "pixel_values": pixel_values.to(self.device),
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
        }

        logging.info(
            "X-CLIP video tensor shape: %s",
            tuple(inputs["pixel_values"].shape)
        )

        # Run inference only once.
        with torch.inference_mode():
            outputs = self.model(**inputs)

        scores = outputs.logits_per_video[0]

        probabilities = torch.softmax(scores, dim=-1)

        values = probabilities.cpu().numpy()

        ranked = sorted(
            [
                {
                    "category": category,
                    "score": float(values[index])
                }
                for index, category in enumerate(CANDIDATES)
            ],
            key=lambda item: item["score"],
            reverse=True
        )

        best = ranked[0]
        second = ranked[1]

        # Conservative classification thresholds.
        # These are heuristic thresholds, not calibrated confidence.
        accepted = (
            best["category"] != "Other"
            and best["score"] >= 0.35
            and best["score"] - second["score"] >= 0.08
        )

        category = (
            best["category"]
            if accepted
            else "Unknown"
        )

        return {
            "category": category,
            "candidate": best["category"],
            "score": best["score"],
            "ranking": ranked,
        }

    def predict_video(self, video_path):

        self.load_model()

        _, _, duration = self.get_video_info(video_path)

        segments = []

        current = 0.0

        while current < duration:

            start = current

            end = min(
                start + self.window_seconds,
                duration
            )

            current += self.stride_seconds

            if end - start < 1.0:
                continue

            frames = self.extract_frames(
                video_path,
                start,
                end
            )

            if not frames:
                continue

            result = self.predict_window(frames)

            category = result["category"]

            activity_label = (
                result["candidate"]
                if category != "Unknown"
                else "Uncertain activity"
            )

            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "category": category,
                "activity_label": activity_label,
                "score": round(result["score"], 4),
                "score_type": "uncalibrated zero-shot softmax",
                "ranking": result["ranking"],
            })

        if not segments:
            raise RuntimeError(
                "X-CLIP could not extract enough video frames."
            )

        # Merge adjacent windows with the same category.
        merged = []

        for segment in segments:

            if (
                merged
                and merged[-1]["category"] == segment["category"]
                and segment["start"] <= merged[-1]["end"]
            ):

                merged[-1]["end"] = segment["end"]

                merged[-1]["score"] = max(
                    merged[-1]["score"],
                    segment["score"]
                )

            else:
                merged.append(segment.copy())

        # Determine the overall category from supported windows.
        supported = [
            segment
            for segment in segments
            if segment["category"] != "Unknown"
        ]

        if supported:

            counts = {}

            for segment in supported:
                category = segment["category"]

                counts[category] = (
                    counts.get(category, 0) + 1
                )

            final_category = max(
                counts,
                key=counts.get
            )

        else:
            final_category = "Unknown"

        best_segment = max(
            segments,
            key=lambda item: item["score"]
        )

        return {
            "classification_status": (
                "supported"
                if final_category != "Unknown"
                else "uncertain"
            ),

            "source": "X-CLIP pretrained zero-shot",

            "model_name": "Microsoft X-CLIP Base Patch16",

            "event_category": final_category,

            "activity_label": best_segment["activity_label"],

            "score": best_segment["score"],

            "score_calibrated": False,

            "duration": round(duration, 2),

            "timeline": merged,

            "window_predictions": segments,
        }