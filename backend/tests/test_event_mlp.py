from event_mlp import EventLensMLPClassifier, apply_event_mlp_result
from app import build_pdf_report, generate_report_narrative


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
    mlp_model = FakeModel()
    backbone = FakeBackbone()

    def __init__(self, probabilities, features_dimension=5120, status="success"):
        self.confidence_threshold = 0.75
        self.probabilities = probabilities
        self.features_dimension = features_dimension
        self.result_status = status
        self.calls = 0

    def predict_video(self, _video_path):
        self.calls += 1
        if self.result_status != "success":
            return {
                "status": self.result_status,
                "message": "Test feature extraction failure.",
            }
        return {
            "status": "success",
            "features_dimension": self.features_dimension,
            "raw_probabilities": self.probabilities,
            "confidence_score": max(self.probabilities.values()),
        }


def build_classifier(engine):
    loads = []

    def factory(**kwargs):
        loads.append(kwargs)
        return engine

    classifier = EventLensMLPClassifier(engine_factory=factory)
    classifier._validate_video_samples = lambda _video_path: None
    return classifier, loads


def test_mlp_classifier_keeps_uncalibrated_prediction_unassigned_and_reuses_engine():
    engine = FakeEngine({
        "Cultural": 0.82,
        "Dance": 0.08,
        "Presentation": 0.06,
        "Unknown": 0.04,
    })
    classifier, loads = build_classifier(engine)

    first = classifier.predict_video("real-video.mp4")
    second = classifier.predict_video("real-video.mp4")

    assert first["classification_status"] == "uncertain"
    assert first["event_type"] == "Uncertain"
    assert first["predicted_category"] == "Cultural"
    assert first["assigned_category"] is None
    assert first["confidence"] is None
    assert first["model_score"] == 0.82
    assert first["score_type"] == "uncalibrated_softmax_score"
    assert first["confidence_calibrated"] is False
    assert first["runner_up_category"] == "Dance"
    assert first["score_margin"] == 0.74
    assert first["features_dimension"] == 5120
    assert second["assigned_category"] is None
    assert len(loads) == 1
    assert engine.calls == 2


def test_each_active_class_remains_unassigned_without_calibration():
    classes = ("Cultural", "Dance", "Presentation", "Unknown")
    for candidate in classes:
        probabilities = {
            label: (0.91 if label == candidate else 0.03)
            for label in classes
        }
        classifier, _ = build_classifier(FakeEngine(probabilities))

        result = classifier.predict_video("video.mp4")

        assert result["predicted_category"] == candidate
        assert result["assigned_category"] is None
        assert result["classification_status"] == "uncertain"
        assert result["confidence_calibrated"] is False


def test_mlp_classifier_does_not_claim_unverified_unknown_as_learned():
    unknown_engine = FakeEngine({
        "Cultural": 0.03,
        "Dance": 0.04,
        "Presentation": 0.08,
        "Unknown": 0.85,
    })
    unknown_classifier, _ = build_classifier(unknown_engine)
    unknown = unknown_classifier.predict_video("video.mp4")

    uncertain_engine = FakeEngine({
        "Cultural": 0.09,
        "Dance": 0.72,
        "Presentation": 0.11,
        "Unknown": 0.08,
    })
    uncertain_classifier, _ = build_classifier(uncertain_engine)
    uncertain = uncertain_classifier.predict_video("video.mp4")

    assert unknown["classification_status"] == "uncertain"
    assert unknown["event_type"] == "Uncertain"
    assert unknown["predicted_category"] == "Unknown"
    assert unknown["assigned_category"] is None
    assert uncertain["classification_status"] == "uncertain"
    assert uncertain["event_type"] == "Uncertain"
    assert uncertain["predicted_category"] == "Dance"
    assert uncertain["assigned_category"] is None


def test_mlp_classifier_rejects_bad_dimensions_and_unsupported_labels():
    bad_dimension = FakeEngine(
        {
            "Cultural": 0.82,
            "Dance": 0.08,
            "Presentation": 0.06,
            "Unknown": 0.04,
        },
        features_dimension=512,
    )
    classifier, _ = build_classifier(bad_dimension)
    assert classifier.predict_video("video.mp4")["classification_status"] == "unsupported"

    for unsupported in ("Panel", "Prize Distribution", "Speech"):
        unsupported_labels = FakeEngine({unsupported: 1.0})
        classifier, _ = build_classifier(unsupported_labels)
        assert classifier.predict_video("video.mp4")["classification_status"] == "unsupported"


def test_mlp_classifier_reports_extraction_failure_without_prediction():
    engine = FakeEngine({}, status="error")
    classifier, _ = build_classifier(engine)

    result = classifier.predict_video("invalid-video.mp4")

    assert result["classification_status"] == "unavailable"
    assert result["event_type"] is None
    assert result["confidence"] is None


def test_mlp_classifier_rejects_unreadable_video_without_loading_models(tmp_path):
    loads = []

    def factory(**kwargs):
        loads.append(kwargs)
        raise AssertionError("Invalid input must not initialize models.")

    classifier = EventLensMLPClassifier(engine_factory=factory)
    result = classifier.predict_video(tmp_path / "missing.mp4")

    assert result["classification_status"] == "unavailable"
    assert result["event_type"] is None
    assert not loads
    assert classifier.status == "not_loaded"


def test_applying_uncertain_result_does_not_assign_learned_unknown():
    event_result = {
        "event_type": "unknown",
        "confidence": None,
        "top_objects": [{"class_name": "person", "count": 2}],
    }
    prediction = {
        "classification_status": "uncertain",
        "event_type": "Uncertain",
        "predicted_category": "Dance",
        "confidence": None,
        "model_score": 0.72,
        "score_type": "uncalibrated_softmax_score",
        "confidence_calibrated": False,
        "model_quality": "unverified",
        "summary": "The raw score is not calibrated; no event category was assigned.",
    }

    applied = apply_event_mlp_result(event_result, prediction)

    assert applied["event_type"] == "Uncertain"
    assert applied["event_classification"]["classification_status"] == "uncertain"
    assert applied["top_objects"] == [{"class_name": "person", "count": 2}]


def test_uncertain_mlp_result_is_described_as_unassigned_in_report_and_pdf():
    event_result = {
        "event_type": "Uncertain",
        "confidence": None,
        "model_quality": "unverified",
        "summary": "The raw score is not calibrated; no event category was assigned.",
        "top_objects": [],
        "event_classification": {
            "classification_status": "uncertain",
            "predicted_category": "Dance",
            "confidence": None,
            "model_score": 0.72,
            "score_type": "uncalibrated_softmax_score",
            "confidence_calibrated": False,
        },
    }

    narrative = generate_report_narrative(event_result, [], 8)
    pdf = build_pdf_report({
        "event_type": event_result["event_type"],
        "summary": event_result["summary"],
        "narrative": narrative,
        "model_quality": event_result["model_quality"],
        "confidence": event_result["confidence"],
        "event_classification": event_result["event_classification"],
    })

    assert 'top candidate was "Dance"' in narrative
    assert "raw softmax score 0.720" in narrative
    assert "not a calibrated probability" in narrative
    assert "No event category was assigned." in narrative
    assert pdf.startswith(b"%PDF-")
    assert b"0.720" in pdf
    assert b"not a calibrated probability" in pdf


def test_calibrated_classifier_supports_event_classes_and_distinguishes_unknown():
    calibration = {
        "temperature": 1.0,
        "confidence_threshold": 0.70,
        "margin_threshold": 0.15,
        "unknown_threshold": 0.70,
        "validation_samples": 60,
        "validation_ece_after": 0.04,
    }
    for category in ("Cultural", "Dance", "Presentation"):
        probs = {
            label: (0.85 if label == category else 0.05)
            for label in ("Cultural", "Dance", "Presentation", "Unknown")
        }
        classifier = EventLensMLPClassifier(
            engine_factory=lambda p=probs, **_kw: FakeEngine(p),
            calibration_config=calibration,
        )
        classifier._validate_video_samples = lambda _path: None
        result = classifier.predict_video("event.mp4")
        assert result["classification_status"] == "supported"
        assert result["event_type"] == category
        assert result["assigned_category"] == category
        assert result["confidence_calibrated"] is True
        assert result["confidence"] == 0.85
        assert result["rejection_status"] == "accepted"

    unknown_classifier = EventLensMLPClassifier(
        engine_factory=lambda **_kw: FakeEngine({
            "Cultural": 0.04,
            "Dance": 0.04,
            "Presentation": 0.07,
            "Unknown": 0.85,
        }),
        calibration_config=calibration,
    )
    unknown_classifier._validate_video_samples = lambda _path: None
    unknown_res = unknown_classifier.predict_video("unrelated.mp4")
    assert unknown_res["classification_status"] == "learned_unknown"
    assert unknown_res["event_type"] == "Unknown"
    assert unknown_res["assigned_category"] == "Unknown"
    assert unknown_res["rejection_status"] == "rejected_as_unknown"

    ambiguous_classifier = EventLensMLPClassifier(
        engine_factory=lambda **_kw: FakeEngine({
            "Cultural": 0.48,
            "Dance": 0.42,
            "Presentation": 0.06,
            "Unknown": 0.04,
        }),
        calibration_config=calibration,
    )
    ambiguous_classifier._validate_video_samples = lambda _path: None
    ambiguous_res = ambiguous_classifier.predict_video("ambiguous.mp4")
    assert ambiguous_res["classification_status"] == "uncertain"
    assert ambiguous_res["event_type"] == "Uncertain"
    assert ambiguous_res["assigned_category"] is None
    assert ambiguous_res["rejection_status"] == "rejected_uncertain"


def test_unverified_calibration_artifact_is_ignored(tmp_path):
    import json
    from event_mlp import _load_validated_calibration

    (tmp_path / "calibration_config.json").write_text(
        json.dumps({
            "verified": False,
            "calibration_method": "temperature_scaling",
            "source_grouped_split": True,
            "temperature": 1.2,
            "confidence_threshold": 0.70,
            "margin_threshold": 0.15,
            "unknown_threshold": 0.70,
            "validation_samples": 100,
            "validation_ece_after": 0.05,
        }),
        encoding="utf-8",
    )
    assert _load_validated_calibration(tmp_path, "eventlens_mlp.keras") is None


def test_calibrated_supported_and_unknown_propagate_to_narrative_and_pdf():
    supported_result = {
        "event_type": "Presentation",
        "confidence": 0.88,
        "model_quality": "calibrated",
        "summary": 'The EventLens MLP classified this video as "Presentation".',
        "top_objects": [],
        "event_classification": {
            "classification_status": "supported",
            "predicted_category": "Presentation",
            "confidence": 0.88,
            "model_score": 0.88,
            "confidence_calibrated": True,
        },
    }
    sup_narrative = generate_report_narrative(supported_result, [], 8)
    sup_pdf = build_pdf_report({
        "event_type": supported_result["event_type"],
        "summary": supported_result["summary"],
        "narrative": sup_narrative,
        "model_quality": supported_result["model_quality"],
        "confidence": supported_result["confidence"],
        "event_classification": supported_result["event_classification"],
    })
    assert 'predicted "Presentation" with calibrated confidence 0.880' in sup_narrative
    assert sup_pdf.startswith(b"%PDF-")

    unknown_result = {
        "event_type": "Unknown",
        "confidence": 0.85,
        "model_quality": "calibrated",
        "summary": 'The EventLens MLP classified this video as "Unknown".',
        "top_objects": [],
        "event_classification": {
            "classification_status": "learned_unknown",
            "predicted_category": "Unknown",
            "confidence": 0.85,
            "model_score": 0.85,
            "confidence_calibrated": True,
        },
    }
    unk_narrative = generate_report_narrative(unknown_result, [], 8)
    unk_pdf = build_pdf_report({
        "event_type": unknown_result["event_type"],
        "summary": unknown_result["summary"],
        "narrative": unk_narrative,
        "model_quality": unknown_result["model_quality"],
        "confidence": unknown_result["confidence"],
        "event_classification": unknown_result["event_classification"],
    })
    assert "rejected this video as Unknown" in unk_narrative
    assert unk_pdf.startswith(b"%PDF-")


def test_video_grouped_split_enforces_no_leakage_and_calibration_improves_ece():
    import numpy as np
    from train_mlp_4class import (
        assign_video_grouped_splits,
        fit_temperature_scaling,
        select_rejection_thresholds,
    )

    records = []
    for category in ("Cultural", "Dance", "Presentation", "Unknown"):
        for group_idx in range(4):
            records.append({
                "video_path": f"{category}_{group_idx}_a.mp4",
                "label": category,
                "label_idx": {"Cultural": 0, "Dance": 1, "Presentation": 2, "Unknown": 3}[category],
                "source_group_id": f"{category}_session_{group_idx}",
                "sha256": f"hash_{category}_{group_idx}_a",
            })
            records.append({
                "video_path": f"{category}_{group_idx}_b.mp4",
                "label": category,
                "label_idx": {"Cultural": 0, "Dance": 1, "Presentation": 2, "Unknown": 3}[category],
                "source_group_id": f"{category}_session_{group_idx}",
                "sha256": f"hash_{category}_{group_idx}_b",
            })

    assigned = assign_video_grouped_splits(records, seed=7)
    group_to_splits = {}
    for row in assigned:
        group_to_splits.setdefault(row["source_group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_to_splits.values())
    assert {row["split"] for row in assigned} == {"train", "validation", "test"}

    # Synthetic overconfident softmax probabilities -> temperature > 1 reduces ECE
    rng = np.random.default_rng(12)
    y_val = np.tile([0, 1, 2, 3], 15)
    logits = rng.normal(0.0, 0.8, size=(len(y_val), 4))
    logits[np.arange(len(y_val)), y_val] += 1.6
    overconfident_logits = logits * 3.0
    overconfident_logits -= np.max(overconfident_logits, axis=1, keepdims=True)
    probs = np.exp(overconfident_logits)
    probs /= np.sum(probs, axis=1, keepdims=True)

    temp, cal_metrics = fit_temperature_scaling(probs, y_val)
    assert temp > 1.0
    assert cal_metrics["nll_after"] <= cal_metrics["nll_before"] + 1e-6
    thresholds = select_rejection_thresholds(probs, y_val)
    assert 0.45 <= thresholds["confidence_threshold"] <= 0.90
