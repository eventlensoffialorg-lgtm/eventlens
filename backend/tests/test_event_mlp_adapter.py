from pathlib import Path

from event_mlp import EventLensMLPClassifier


class FakeModel:
    input_shape = (None, 5120)
    output_shape = (None, 4)


class FakeBackbone:
    output_shape = (None, 1280)


class FakeEngine:
    expected_dim = 5120
    active_classes = {
        0: "Cultural",
        1: "Dance",
        2: "Presentation",
        3: "Unknown",
    }
    confidence_threshold = 0.75
    mlp_model = FakeModel()
    backbone = FakeBackbone()

    def predict_video(self, _video_path):
        return {
            "status": "success",
            "features_dimension": 5120,
            "raw_probabilities": {
                "Cultural": 0.1,
                "Dance": 0.7,
                "Presentation": 0.1,
                "Unknown": 0.1,
            },
        }


def test_model_variant_defaults_to_existing_production_model(monkeypatch):
    monkeypatch.delenv("EVENTLENS_ML_MODEL_VARIANT", raising=False)

    classifier = EventLensMLPClassifier()

    assert classifier.requested_variant == "production"
    assert classifier.model_filename == "eventlens_mlp.keras"


def test_candidate_package_configuration_is_accepted():
    classifier = EventLensMLPClassifier(model_variant="candidate")

    classifier._validate_assets(classifier.model_assets_dir, classifier.model_filename)

    assert classifier.model_filename == "eventlens_mlp_4class.keras"


def test_candidate_load_failure_uses_production_fallback(monkeypatch):
    engine = FakeEngine()
    factory_calls = []

    def engine_factory(**kwargs):
        factory_calls.append(kwargs)
        return engine

    classifier = EventLensMLPClassifier(
        engine_factory=engine_factory,
        model_variant="candidate",
    )
    monkeypatch.setattr(classifier, "_validate_video_samples", lambda _path: None)
    monkeypatch.setattr(classifier, "_validate_engine", lambda _engine: None)

    validated_assets = []

    def validate_assets(assets_dir, model_filename):
        assets_dir = Path(assets_dir)
        validated_assets.append((assets_dir, model_filename))
        if assets_dir == classifier.model_assets_dir:
            raise ValueError("candidate load failure")

    monkeypatch.setattr(classifier, "_validate_assets", validate_assets)

    prediction = classifier.predict_video("video.mp4")

    assert prediction["predicted_category"] == "Dance"
    assert prediction["rejection_status"] == "not_assessed"
    assert "No independently validated out-of-distribution rejection rule" in (
        prediction["rejection_reason"]
    )
    assert "no event category has been assigned" in prediction["summary"]
    assert prediction["requested_model_variant"] == "candidate"
    assert prediction["model_variant"] == "production"
    assert prediction["model_fallback_used"] is True
    assert "production model is active" in prediction["model_load_warning"]
    assert validated_assets[-1] == (
        classifier.production_assets_dir,
        "eventlens_mlp.keras",
    )
    assert factory_calls[-1]["model_filename"] == "eventlens_mlp.keras"


def test_production_keras_model_and_assets_load_with_expected_5120_4class_shapes():
    import numpy as np
    from tensorflow.keras.models import load_model

    classifier = EventLensMLPClassifier()
    classifier._validate_assets(classifier.model_assets_dir, classifier.model_filename)
    model = load_model(classifier.model_assets_dir / classifier.model_filename)

    assert tuple(model.input_shape) == (None, 5120)
    assert tuple(model.output_shape) == (None, 4)

    probs = model.predict(np.zeros((1, 5120), dtype=np.float32), verbose=0)[0]
    assert len(probs) == 4
    assert abs(float(np.sum(probs)) - 1.0) < 1e-4


def test_production_engine_extracts_5120_features_and_maps_four_classes(
    monkeypatch,
    tmp_path,
):
    import importlib.util
    import cv2
    import numpy as np

    classifier = EventLensMLPClassifier()
    spec = importlib.util.spec_from_file_location(
        "eventlens_package_engine_test",
        classifier.engine_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class StubBackbone:
        output_shape = (None, 1280)
        trainable = False

        def predict(self, frames_array, batch_size=None, verbose=0):
            assert frames_array.shape[1:] == (160, 160, 3)
            return np.full((len(frames_array), 1280), 0.1, dtype=np.float32)

    monkeypatch.setattr(module, "MobileNetV2", lambda **_kwargs: StubBackbone())

    video_path = tmp_path / "sample_event.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (160, 160),
    )
    for idx in range(10):
        frame = np.full((160, 160, 3), (idx * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    engine_classifier = EventLensMLPClassifier(
        engine_factory=lambda **kwargs: module.EventLensMLEngine(**kwargs)
    )
    result = engine_classifier.predict_video(video_path)

    assert result["status"] == "success"
    assert result["classification_status"] == "uncertain"
    assert result["event_type"] == "Uncertain"
    assert result["assigned_category"] is None
    assert result["confidence"] is None
    assert result["confidence_calibrated"] is False
    assert result["features_dimension"] == 5120
    assert set(result["raw_probabilities"]) == {
        "Cultural",
        "Dance",
        "Presentation",
        "Unknown",
    }
    assert result["predicted_category"] in result["raw_probabilities"]
    assert result["rejection_status"] == "not_assessed"


def test_production_load_failure_returns_unavailable_without_crashing(monkeypatch):
    def broken_factory(**_kwargs):
        raise RuntimeError("broken production model")

    classifier = EventLensMLPClassifier(engine_factory=broken_factory)
    monkeypatch.setattr(classifier, "_validate_video_samples", lambda _path: None)

    result = classifier.predict_video("video.mp4")

    assert result["classification_status"] == "unavailable"
    assert result["event_type"] is None
    assert result["confidence"] is None
    assert result["model_quality"] == "unavailable"
    assert "could not be initialized" in result["message"]
