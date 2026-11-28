from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import threading
from pathlib import Path

import cv2
import numpy as np

EXPECTED_ACTIVE_CLASSES = {
    "Cultural": 0,
    "Dance": 1,
    "Presentation": 2,
    "Unknown": 3,
}
SUPPORTED_EVENT_CATEGORIES = {"Cultural", "Dance", "Presentation"}
CALIBRATION_FILENAME = "calibration_config.json"
MIN_CALIBRATION_VALIDATION_SAMPLES = 40
EXPECTED_FEATURE_CONFIG = {
    "extractor_backbone": "MobileNetV2_ImageNet",
    "input_resolution": [160, 160],
    "uniformly_sampled_frames": 8,
    "spatial_aggregation": ["mean", "std"],
    "temporal_aggregation": ["mean_difference", "std_difference"],
    "concatenation_order": [
        "spatial_mean",
        "spatial_std",
        "temporal_mean",
        "temporal_std",
    ],
    "expected_feature_dimension": 5120,
}
CONFIDENCE_THRESHOLD = 0.75
MODEL_VARIANTS = {
    "production": ("ML_Integration_Package/model_assets", "eventlens_mlp.keras"),
    "candidate": (
        "backend/eventlens_final_mlp_4class",
        "eventlens_mlp_4class.keras",
    ),
    "best": ("backend/eventlens_final_mlp_4class", "best_4class_model.keras"),
}


def _compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_temperature_scaling(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    """Apply post-hoc temperature scaling on softmax probabilities."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a positive finite number.")
    eps = 1e-12
    scaled_logits = {
        label: math.log(max(float(prob), eps)) / temperature
        for label, prob in probabilities.items()
    }
    max_logit = max(scaled_logits.values())
    exp_values = {
        label: math.exp(value - max_logit)
        for label, value in scaled_logits.items()
    }
    normalizer = sum(exp_values.values())
    return {
        label: value / normalizer
        for label, value in exp_values.items()
    }


def _load_validated_calibration(assets_dir: Path, model_filename: str):
    calibration_path = assets_dir / CALIBRATION_FILENAME
    if not calibration_path.is_file():
        return None

    try:
        config = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Ignoring unreadable EventLens calibration file: %s", exc)
        return None

    if not isinstance(config, dict) or config.get("verified") is not True:
        logging.warning("Ignoring unverified EventLens calibration configuration.")
        return None

    if config.get("calibration_method") != "temperature_scaling":
        logging.warning("Unsupported calibration method in calibration_config.json.")
        return None

    if config.get("source_grouped_split") is not True:
        logging.warning("Calibration artifact lacks source-grouped validation guarantee.")
        return None

    validation_samples = config.get("validation_samples")
    if (
        not isinstance(validation_samples, int)
        or validation_samples < MIN_CALIBRATION_VALIDATION_SAMPLES
    ):
        logging.warning("Calibration validation sample count is insufficient.")
        return None

    try:
        temperature = float(config["temperature"])
        confidence_threshold = float(config["confidence_threshold"])
        margin_threshold = float(config["margin_threshold"])
        unknown_threshold = float(config["unknown_threshold"])
        ece_after = float(config["validation_ece_after"])
    except (KeyError, TypeError, ValueError):
        logging.warning("Calibration parameters in calibration_config.json are incomplete.")
        return None

    if not (
        math.isfinite(temperature)
        and 0.05 <= temperature <= 20.0
        and math.isfinite(confidence_threshold)
        and 0.25 <= confidence_threshold <= 0.999
        and math.isfinite(margin_threshold)
        and 0.0 <= margin_threshold <= 0.999
        and math.isfinite(unknown_threshold)
        and 0.25 <= unknown_threshold <= 0.999
        and math.isfinite(ece_after)
        and 0.0 <= ece_after <= 1.0
    ):
        logging.warning("Calibration thresholds or ECE values are out of valid bounds.")
        return None

    expected_sha = config.get("model_sha256")
    model_path = assets_dir / model_filename
    if expected_sha and model_path.is_file():
        actual_sha = _compute_file_sha256(model_path)
        if actual_sha != str(expected_sha).strip().lower():
            logging.warning(
                "Calibration artifact model_sha256 does not match %s; ignoring.",
                model_filename,
            )
            return None

    return {
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "unknown_threshold": unknown_threshold,
        "validation_samples": validation_samples,
        "validation_ece_after": ece_after,
    }


class EventLensMLPClassifier:
    def __init__(
        self,
        model_assets_dir=None,
        engine_factory=None,
        model_variant=None,
        calibration_config=None,
    ):
        project_root = Path(__file__).resolve().parent.parent
        configured_assets = os.environ.get("EVENTLENS_ML_ASSETS_DIR")
        requested_variant = model_variant or os.environ.get(
            "EVENTLENS_ML_MODEL_VARIANT", "production"
        )
        self.requested_variant = requested_variant.strip().lower()
        self._custom_assets_dir = model_assets_dir or configured_assets
        if self._custom_assets_dir:
            self.model_assets_dir = Path(self._custom_assets_dir)
            self.model_filename = "eventlens_mlp.keras"
            self.requested_variant = "custom"
        else:
            variant_config = MODEL_VARIANTS.get(self.requested_variant)
            if variant_config:
                relative_dir, self.model_filename = variant_config
                self.model_assets_dir = project_root / relative_dir
            else:
                self.model_assets_dir = (
                    project_root / MODEL_VARIANTS["production"][0]
                )
                self.model_filename = MODEL_VARIANTS["production"][1]
        self.production_assets_dir = (
            project_root / MODEL_VARIANTS["production"][0]
        )
        self.engine_path = (
            project_root
            / "ML_Integration_Package"
            / "eventlens_ml_engine.py"
        )
        self._engine_factory = engine_factory
        self._engine = None
        self._load_attempted = False
        self._load_error = None
        self._load_warning = None
        self._active_variant = None
        self._calibration_config = calibration_config
        self._lock = threading.Lock()

    @property
    def status(self):
        if self._engine is not None:
            return "ready"
        return "unavailable" if self._load_attempted else "not_loaded"

    @property
    def active_variant(self):
        return self._active_variant

    def _validate_assets(self, assets_dir, model_filename):
        required_files = (
            self.engine_path,
            assets_dir / model_filename,
            assets_dir / "class_mapping.json",
            assets_dir / "feature_extraction_config.json",
            assets_dir / "model_config.json",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required EventLens MLP files are missing: " + ", ".join(missing)
            )

        with (assets_dir / "class_mapping.json").open(encoding="utf-8") as source:
            mapping = json.load(source)
        if mapping.get("active_model_classes") != EXPECTED_ACTIVE_CLASSES:
            raise ValueError("The EventLens MLP active class mapping is incompatible.")

        with (assets_dir / "feature_extraction_config.json").open(
            encoding="utf-8"
        ) as source:
            feature_config = json.load(source)
        required_feature_config = {
            key: value
            for key, value in EXPECTED_FEATURE_CONFIG.items()
            if key != "concatenation_order"
        }
        if any(
            feature_config.get(key) != value
            for key, value in required_feature_config.items()
        ):
            raise ValueError(
                "The EventLens MLP feature extraction configuration is incompatible."
            )
        if feature_config.get(
            "concatenation_order",
            EXPECTED_FEATURE_CONFIG["concatenation_order"],
        ) != EXPECTED_FEATURE_CONFIG["concatenation_order"]:
            raise ValueError("The EventLens MLP feature order is incompatible.")

        with (assets_dir / "model_config.json").open(encoding="utf-8") as source:
            model_config = json.load(source)
        layers = model_config.get("layers", [])
        hidden_layers = model_config.get("hidden_layers")
        output_classes = model_config.get("output_classes")
        if hidden_layers is not None:
            layers_compatible = hidden_layers == [1024, 512, 256]
        else:
            layers_compatible = layers[:-1] == [1024, 512, 256]
        if output_classes is None and layers:
            output_classes = layers[-1]
        activations_compatible = (
            model_config.get("activation") in (None, "relu")
            and model_config.get("final_activation") in (None, "softmax")
        )
        if (
            model_config.get("input_dim") != 5120
            or not layers_compatible
            or output_classes != len(EXPECTED_ACTIVE_CLASSES)
            or not activations_compatible
        ):
            raise ValueError("The EventLens MLP model configuration is incompatible.")

    def _create_engine(self, assets_dir, model_filename):
        if self._engine_factory is not None:
            return self._engine_factory(
                model_assets_dir=str(assets_dir),
                confidence_threshold=CONFIDENCE_THRESHOLD,
                model_filename=model_filename,
            )

        spec = importlib.util.spec_from_file_location(
            "eventlens_package_engine",
            self.engine_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load the packaged EventLens MLP engine.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.EventLensMLEngine(
            model_assets_dir=str(assets_dir),
            confidence_threshold=CONFIDENCE_THRESHOLD,
            model_filename=model_filename,
        )

    def _validate_engine(self, engine):
        if engine.expected_dim != 5120:
            raise ValueError("The EventLens MLP engine does not expect 5120 features.")
        if engine.active_classes != {
            index: label for label, index in EXPECTED_ACTIVE_CLASSES.items()
        }:
            raise ValueError("The EventLens MLP engine class mapping is incompatible.")
        if engine.confidence_threshold != CONFIDENCE_THRESHOLD:
            raise ValueError("The EventLens MLP confidence threshold is incompatible.")

        input_shape = tuple(engine.mlp_model.input_shape)
        output_shape = tuple(engine.mlp_model.output_shape)
        if input_shape[-1] != 5120 or output_shape[-1] != 4:
            raise ValueError(
                "The EventLens MLP model dimensions do not match its configuration."
            )
        if tuple(engine.backbone.output_shape)[-1] != 1280:
            raise ValueError("MobileNetV2 does not produce the expected 1280 features.")

    def _ensure_engine(self):
        if self._load_attempted:
            return self._engine

        self._load_attempted = True
        if self._custom_assets_dir:
            requested_assets = self.model_assets_dir
            requested_filename = self.model_filename
        else:
            variant_config = MODEL_VARIANTS.get(self.requested_variant)
            if variant_config is None:
                self._load_warning = (
                    "Unknown model variant requested; production fallback selected."
                )
                logging.warning(
                    "Unknown EventLens MLP variant %r; using production.",
                    self.requested_variant,
                )
                requested_assets = self.production_assets_dir
                requested_filename = MODEL_VARIANTS["production"][1]
            else:
                requested_assets = self.model_assets_dir
                requested_filename = self.model_filename

        try:
            self._validate_assets(requested_assets, requested_filename)
            engine = self._create_engine(requested_assets, requested_filename)
            self._validate_engine(engine)
            self._engine = engine
            self._active_variant = (
                "production" if self._load_warning else self.requested_variant
            )
            if self._calibration_config is None:
                self._calibration_config = _load_validated_calibration(
                    requested_assets,
                    requested_filename,
                )
        except Exception:
            if self.requested_variant != "production" and not self._custom_assets_dir:
                logging.exception(
                    "Requested EventLens MLP variant %s failed to initialize.",
                    self.requested_variant,
                )
                try:
                    production_filename = MODEL_VARIANTS["production"][1]
                    self._validate_assets(
                        self.production_assets_dir,
                        production_filename,
                    )
                    engine = self._create_engine(
                        self.production_assets_dir,
                        production_filename,
                    )
                    self._validate_engine(engine)
                    self._engine = engine
                    self._active_variant = "production"
                    if self._calibration_config is None:
                        self._calibration_config = _load_validated_calibration(
                            self.production_assets_dir,
                            production_filename,
                        )
                    self._load_warning = (
                        f"Requested model variant '{self.requested_variant}' failed; "
                        "the production model is active."
                    )
                except Exception:
                    self._load_error = (
                        "The requested and production EventLens MLP models could "
                        "not be initialized."
                    )
                    logging.exception("EventLens MLP production fallback failed.")
            else:
                self._load_error = (
                    "The EventLens MLP could not be initialized. "
                    "Check the packaged model assets and installed dependencies."
                )
                logging.exception("EventLens MLP engine initialization failed.")
        return self._engine

    @staticmethod
    def _validate_video_samples(video_path):
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return "The uploaded video could not be opened by the MLP."

            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 8:
                return "The MLP requires at least eight readable video frames."

            frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int)
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                success, frame = capture.read()
                if not success or frame is None:
                    return "The MLP could not read all eight sampled video frames."
        finally:
            capture.release()
        return None

    def _failure(self, status, message):
        return {
            "status": status,
            "classification_status": status,
            "event_type": None,
            "predicted_category": None,
            "assigned_category": None,
            "confidence": None,
            "model_score": None,
            "score_type": "unavailable",
            "confidence_calibrated": False,
            "model_quality": "unavailable",
            "rejection_status": "unavailable",
            "requested_model_variant": self.requested_variant,
            "model_variant": self._active_variant,
            "model_fallback_used": (
                self._active_variant is not None
                and self._active_variant != self.requested_variant
            ),
            "features_dimension": None,
            "message": message,
        }

    def _build_prediction(
        self,
        *,
        classification_status,
        event_type,
        predicted_category,
        assigned_category,
        confidence,
        model_score,
        raw_probability,
        score_type,
        confidence_calibrated,
        runner_up_category,
        score_margin,
        model_quality,
        rejection_status,
        rejection_reason,
        raw_probabilities,
        summary,
        calibrated_probabilities=None,
        calibration_temperature=None,
    ):
        payload = {
            "status": "success",
            "classification_status": classification_status,
            "event_type": event_type,
            "predicted_category": predicted_category,
            "assigned_category": assigned_category,
            "confidence": confidence,
            "model_score": model_score,
            "raw_probability": raw_probability,
            "score_type": score_type,
            "confidence_calibrated": confidence_calibrated,
            "runner_up_category": runner_up_category,
            "score_margin": score_margin,
            "model_quality": model_quality,
            "rejection_status": rejection_status,
            "rejection_reason": rejection_reason,
            "requested_model_variant": self.requested_variant,
            "model_variant": self._active_variant,
            "model_fallback_used": self._active_variant != self.requested_variant,
            "model_load_warning": self._load_warning,
            "features_dimension": 5120,
            "raw_probabilities": raw_probabilities,
            "summary": summary,
        }
        if calibrated_probabilities is not None:
            payload["calibrated_probabilities"] = calibrated_probabilities
        if calibration_temperature is not None:
            payload["calibration_temperature"] = calibration_temperature
        return payload

    def predict_video(self, video_path):
        with self._lock:
            sample_error = self._validate_video_samples(video_path)
            if sample_error:
                return self._failure("unavailable", sample_error)

            engine = self._ensure_engine()
            if engine is None:
                return self._failure(
                    "unavailable",
                    self._load_error or "The EventLens MLP model is unavailable.",
                )

            try:
                result = engine.predict_video(str(video_path))
            except Exception:
                logging.exception("EventLens MLP video inference failed.")
                return self._failure(
                    "unavailable",
                    "The EventLens MLP could not process this video.",
                )

            if not isinstance(result, dict) or result.get("status") != "success":
                message = (
                    result.get("message", "Feature extraction or inference failed.")
                    if isinstance(result, dict)
                    else "The EventLens MLP returned an invalid result."
                )
                logging.warning("EventLens MLP video analysis failed: %s", message)
                return self._failure("unavailable", message)

            if result.get("features_dimension") != 5120:
                return self._failure(
                    "unsupported",
                    "The EventLens MLP did not receive a 5120-dimensional feature vector.",
                )

            probabilities = result.get("raw_probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != set(
                EXPECTED_ACTIVE_CLASSES
            ):
                return self._failure(
                    "unsupported",
                    "The EventLens MLP returned unsupported class labels.",
                )

            try:
                scores = {
                    label: float(probabilities[label])
                    for label in EXPECTED_ACTIVE_CLASSES
                }
            except (TypeError, ValueError):
                return self._failure(
                    "unsupported",
                    "The EventLens MLP returned invalid class scores.",
                )

            if (
                any(not math.isfinite(score) or score < 0 or score > 1 for score in scores.values())
                or not math.isclose(sum(scores.values()), 1.0, abs_tol=0.02)
            ):
                return self._failure(
                    "unsupported",
                    "The EventLens MLP returned an invalid probability distribution.",
                )

            if self._calibration_config is not None:
                calibration = self._calibration_config
                calibrated_scores = apply_temperature_scaling(
                    scores,
                    calibration["temperature"],
                )
                ranked_scores = sorted(
                    calibrated_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                candidate, model_score = ranked_scores[0]
                runner_up, runner_up_score = ranked_scores[1]
                score_margin = model_score - runner_up_score

                if (
                    candidate == "Unknown"
                    and model_score >= calibration["unknown_threshold"]
                    and score_margin >= calibration["margin_threshold"]
                ):
                    return self._build_prediction(
                        classification_status="learned_unknown",
                        event_type="Unknown",
                        predicted_category="Unknown",
                        assigned_category="Unknown",
                        confidence=round(model_score, 4),
                        model_score=model_score,
                        raw_probability=scores[candidate],
                        score_type="temperature_scaled_probability",
                        confidence_calibrated=True,
                        runner_up_category=runner_up,
                        score_margin=score_margin,
                        model_quality="calibrated",
                        rejection_status="rejected_as_unknown",
                        rejection_reason="Video matches the validated Unknown / out-of-scope class.",
                        raw_probabilities=scores,
                        calibrated_probabilities=calibrated_scores,
                        calibration_temperature=calibration["temperature"],
                        summary=(
                            'The EventLens MLP classified this video as "Unknown" '
                            "(outside Cultural, Dance, and Presentation) with "
                            f"calibrated confidence {model_score:.3f}."
                        ),
                    )

                if (
                    candidate in SUPPORTED_EVENT_CATEGORIES
                    and model_score >= calibration["confidence_threshold"]
                    and score_margin >= calibration["margin_threshold"]
                ):
                    return self._build_prediction(
                        classification_status="supported",
                        event_type=candidate,
                        predicted_category=candidate,
                        assigned_category=candidate,
                        confidence=round(model_score, 4),
                        model_score=model_score,
                        raw_probability=scores[candidate],
                        score_type="temperature_scaled_probability",
                        confidence_calibrated=True,
                        runner_up_category=runner_up,
                        score_margin=score_margin,
                        model_quality="calibrated",
                        rejection_status="accepted",
                        rejection_reason=(
                            "Accepted by validated temperature-scaled confidence "
                            "and top-two margin thresholds."
                        ),
                        raw_probabilities=scores,
                        calibrated_probabilities=calibrated_scores,
                        calibration_temperature=calibration["temperature"],
                        summary=(
                            f'The EventLens MLP classified this video as "{candidate}" '
                            f"with calibrated confidence {model_score:.3f}."
                        ),
                    )

                return self._build_prediction(
                    classification_status="uncertain",
                    event_type="Uncertain",
                    predicted_category=candidate,
                    assigned_category=None,
                    confidence=None,
                    model_score=model_score,
                    raw_probability=scores[candidate],
                    score_type="temperature_scaled_probability",
                    confidence_calibrated=True,
                    runner_up_category=runner_up,
                    score_margin=score_margin,
                    model_quality="calibrated",
                    rejection_status="rejected_uncertain",
                    rejection_reason=(
                        "Calibrated confidence or top-two margin is below the "
                        "validated acceptance threshold."
                    ),
                    raw_probabilities=scores,
                    calibrated_probabilities=calibrated_scores,
                    calibration_temperature=calibration["temperature"],
                    summary=(
                        f'The MLP top candidate is "{candidate}" with calibrated '
                        f"score {model_score:.3f} (margin {score_margin:.3f}), "
                        "which is below the validated acceptance criteria; no "
                        "event category has been assigned."
                    ),
                )

            ranked_scores = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            candidate, model_score = ranked_scores[0]
            runner_up, runner_up_score = ranked_scores[1]
            score_margin = model_score - runner_up_score
            return self._build_prediction(
                classification_status="uncertain",
                event_type="Uncertain",
                predicted_category=candidate,
                assigned_category=None,
                confidence=None,
                model_score=model_score,
                raw_probability=model_score,
                score_type="uncalibrated_softmax_score",
                confidence_calibrated=False,
                runner_up_category=runner_up,
                score_margin=score_margin,
                model_quality="unverified",
                rejection_status="not_assessed",
                rejection_reason=(
                    "No independently validated out-of-distribution rejection "
                    "rule is configured."
                ),
                raw_probabilities=scores,
                summary=(
                    f'The MLP top candidate is "{candidate}" with raw softmax score '
                    f"{model_score:.3f}. This score is not calibrated; no event "
                    "category has been assigned. No independently validated "
                    "out-of-distribution rejection rule is configured."
                ),
            )


def apply_event_mlp_result(event_result, prediction):
    event_result["event_classification"] = prediction
    if prediction.get("classification_status") in {
        "supported",
        "learned_unknown",
        "uncertain",
    }:
        event_result.update({
            "event_type": prediction["event_type"],
            "confidence": prediction["confidence"],
            "summary": prediction["summary"],
            "model_quality": prediction["model_quality"],
        })
    return event_result
