import io
from types import SimpleNamespace

import app as backend
from app import EventPipeline, build_pdf_report, generate_report_narrative
import numpy as np
from sklearn.dummy import DummyClassifier


def test_event_pipeline_classifies_event_and_summary():
    pipeline = EventPipeline()

    sample_detections = [
        {"class_name": "person", "confidence": 0.95, "bbox": [10, 20, 40, 80]},
        {"class_name": "person", "confidence": 0.94, "bbox": [50, 18, 76, 82]},
        {"class_name": "laptop", "confidence": 0.87, "bbox": [30, 40, 55, 70]},
        {"class_name": "tvmonitor", "confidence": 0.82, "bbox": [80, 30, 95, 70]},
    ]

    result = pipeline.process_detections(sample_detections)

    assert result["event_type"] == "unknown"
    assert result["model_quality"] == "unavailable"
    assert "summary" in result
    assert len(result["summary"]) > 20
    assert result["confidence"] is None


def test_event_pipeline_handles_empty_detections():
    pipeline = EventPipeline()

    result = pipeline.process_detections([])

    assert result["event_type"] == "unknown"
    assert "summary" in result
    assert result["model_quality"] == "unavailable"


def test_event_pipeline_counts_peak_people_per_frame_instead_of_recounting_them():
    pipeline = EventPipeline()
    first_frame = [
        {"class_name": "person", "confidence": 0.9},
        {"class_name": "person", "confidence": 0.8},
        {"class_name": "chair", "confidence": 0.9},
    ]
    second_frame = [
        {"class_name": "person", "confidence": 0.9},
        {"class_name": "person", "confidence": 0.8},
        {"class_name": "chair", "confidence": 0.9},
    ]

    result = pipeline.process_detections(first_frame + second_frame, [first_frame, second_frame])

    assert result["peak_person_count"] == 2
    assert {item["class_name"]: item["count"] for item in result["top_objects"]}["person"] == 2


def test_event_pipeline_accepts_labels_from_512_feature_event_model():
    pipeline = EventPipeline()
    trained_classifier = DummyClassifier(
        strategy="constant",
        constant="project presentation",
    )
    trained_classifier.fit(
        [[0] * 512, [1] * 512],
        ["project presentation", "project presentation"],
    )
    pipeline.model = trained_classifier

    result = pipeline.classify_video_features(
        np.zeros(512, dtype=np.float32),
        {"top_objects": [{"class_name": "person", "count": 1}]},
    )

    assert result["event_type"] == "project presentation"
    assert '"project presentation"' in result["summary"]


def test_event_pipeline_loads_model_and_metadata_package(tmp_path, monkeypatch):
    trained_classifier = DummyClassifier(
        strategy="constant",
        constant="speech",
    )
    trained_classifier.fit(
        [[0] * 512, [1] * 512],
        ["speech", "speech"],
    )
    model_path = tmp_path / "event_classifier.joblib"
    backend.joblib.dump({
        "model": trained_classifier,
        "metadata": {
            "feature_dimension": 512,
            "class_labels": ["speech"],
        },
    }, model_path)
    monkeypatch.setenv("EVENT_MLP_MODEL_PATH", str(model_path))

    pipeline = EventPipeline()

    assert pipeline.model_quality == "trained"
    assert pipeline.model_metadata["feature_dimension"] == 512
    result = pipeline.classify_video_features(
        np.ones(512, dtype=np.float32),
        {"top_objects": []},
    )
    assert result["event_type"] == "speech"


def test_action_predictions_keep_ucf_categories_and_do_not_claim_confidence():
    recognizer = backend.VideoActionRecognizer()
    predictions = recognizer._format_predictions(
        np.array([0.3, 0.9, 0.2]),
        categories=["BandMarching", "PlayingFlute", "WritingOnBoard"],
    )

    assert [item["action_name"] for item in predictions] == [
        "PlayingFlute", "BandMarching", "WritingOnBoard"
    ]
    assert all(item["confidence"] is None for item in predictions)
    assert all("uncalibrated" in item["score_type"] for item in predictions)


def test_report_narrative_includes_observations_and_prototype_disclaimer():
    narrative = generate_report_narrative(
        {
            "summary": "A presentation was detected.",
            "confidence": 0.75,
            "top_objects": [{"class_name": "person", "count": 2}],
        },
        [{"class_name": "person", "confidence": 0.9}],
        4,
    )

    assert "sampled 4 frame(s)" in narrative
    assert "2 person(s)" in narrative
    assert "automated estimate" in narrative


def test_report_narrative_includes_action_model_predictions_as_suggestions():
    narrative = generate_report_narrative(
        {
            "summary": "The video most closely matches a workshop.",
            "confidence": 0.8,
            "top_objects": [{"class_name": "person", "count": 2}],
            "peak_person_count": 2,
        },
        [{"class_name": "person", "confidence": 0.9}],
        16,
        [{"action_name": "typing", "confidence": 0.84}],
    )

    assert 'action model suggested "typing"' in narrative


def test_generated_report_pdf_has_valid_cross_reference_offsets():
    pdf = build_pdf_report({
        "report_title": "Event report",
        "narrative": "An event was detected (prototype).",
    })
    xref_offset = int(pdf.rsplit(b"startxref\n", 1)[1].splitlines()[0])

    assert pdf.startswith(b"%PDF-1.4")
    assert pdf[xref_offset:].startswith(b"xref\n")
    assert b"An event was detected \\(prototype\\)." in pdf


def test_health_endpoint_does_not_require_authentication():
    response = backend.app.test_client().get("/api/health")

    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_api_requires_authentication():
    response = backend.app.test_client().get("/api/reports")

    assert response.status_code == 401
    assert response.json["error"] == "Authentication is required."


def test_backend_accepts_frontend_supabase_variables_without_a_token(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)
    monkeypatch.setenv("VITE_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("VITE_SUPABASE_PUBLISHABLE_KEY", "publishable-key")

    response = backend.app.test_client().get("/api/reports")

    assert response.status_code == 401
    assert response.json["error"] == "Authentication is required."


def test_reports_are_persisted_and_isolated_by_user(monkeypatch):
    from contextlib import nullcontext

    report = {
        "id": "report-1",
        "created_at": "2026-01-01T00:00:00",
        "summary": "A test report",
    }

    class FakeCursor:
     def execute(self, query, params=None):
        self.params = params

     def fetchone(self):
        import hashlib

        token_hash = self.params[0]

        if token_hash == hashlib.sha256(b"test-token").hexdigest():
            user_id = "user-1"
        else:
            user_id = "user-2"

        return {
            "id": user_id,
            "full_name": "Test User",
            "email": "test@example.com",
        }

    class FakeConnection:
        def cursor(self):
            return nullcontext(FakeCursor())

    monkeypatch.setattr(
        backend,
        "ensure_database_initialized",
        lambda: True,
    )

    monkeypatch.setattr(
        backend,
        "get_db_connection",
        lambda: nullcontext(FakeConnection()),
    )

    monkeypatch.setattr(
        backend,
        "get_user_report",
        lambda user_id, report_id: (
            report if user_id == "user-1" and report_id == "report-1"
            else None
        ),
    )

    client = backend.app.test_client()

    own_report = client.get(
        "/api/reports/report-1",
        headers={"Authorization": "Bearer test-token"},
    )

    other_user_report = client.get(
        "/api/reports/report-1",
        headers={"Authorization": "Bearer another-token"},
    )

    assert own_report.status_code == 200
    assert own_report.json["summary"] == "A test report"
    assert other_user_report.status_code == 404
    


def test_analyze_route_returns_peak_people_count_and_runs_detection_once_per_frame(
    monkeypatch,
    tmp_path,
):
    import time
    from contextlib import nullcontext

    class FakeCursor:
        def execute(self, query, params=None):
            pass

        def fetchone(self):
            return {
                "id": "user-1",
                "full_name": "Test User",
                "email": "test@example.com",
            }

    class FakeConnection:
        def cursor(self):
            return nullcontext(FakeCursor())

    monkeypatch.setattr(backend, "ensure_database_initialized", lambda: True)
    monkeypatch.setattr(
        backend,
        "get_db_connection",
        lambda: nullcontext(FakeConnection()),
    )
    monkeypatch.setattr(backend, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(backend, "GENERATED_DIR", tmp_path / "generated")
    backend.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    backend.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(backend, "save_report", lambda *_args, **_kwargs: None)

    video_file = tmp_path / "event.mp4"
    writer = backend.cv2.VideoWriter(
        str(video_file),
        backend.cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 48),
    )
    for _ in range(4):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(
        backend,
        "extract_sample_frames",
        lambda *args, **kwargs: (frames, [0.0, 3.0], 6.0),
    )

    detected = [
        {"class_name": "person", "confidence": 0.95, "bbox": [5, 5, 30, 45]},
        {"class_name": "person", "confidence": 0.92, "bbox": [35, 5, 60, 45]},
        {"class_name": "chair", "confidence": 0.89, "bbox": [20, 25, 45, 45]},
    ]

    yolo_calls = []

    def detect(_frame):
        yolo_calls.append(True)
        return detected

    monkeypatch.setattr(backend.pipeline, "run_yolo", detect)
    monkeypatch.setattr(
        backend.pipeline,
        "track_people",
        lambda _frames: {
            "unique_tracked_people": 2,
            "peak_tracked_people": 2,
            "tracking_status": "ready",
        },
    )
    monkeypatch.setattr(
        backend.action_recognizer,
        "predict_video",
        lambda _path: {
            "action_predictions": [],
            "action_clip_details": [],
            "action_timeline": [],
            "action_summary": "",
        },
    )
    monkeypatch.setattr(
        backend.event_mlp_classifier,
        "predict_video",
        lambda _path: {
            "status": "success",
            "classification_status": "uncertain",
            "event_type": "Uncertain",
            "predicted_category": "Presentation",
            "assigned_category": None,
            "confidence": None,
            "model_score": 0.78,
            "score_type": "uncalibrated_softmax_score",
            "confidence_calibrated": False,
            "model_quality": "unverified",
            "features_dimension": 5120,
            "summary": (
                'The MLP top candidate is "Presentation" with raw softmax score '
                "0.780. This score is not calibrated; no event category has been assigned."
            ),
        },
    )
    monkeypatch.setattr(backend.cv2, "imwrite", lambda *args, **_kwargs: True)
    monkeypatch.setattr(
        backend,
        "create_full_detection_video",
        lambda *args, **kwargs: None,
    )

    client = backend.app.test_client()
    with video_file.open("rb") as stream:
        response = client.post(
            "/api/analyze",
            data={"video": (stream, "event.mp4")},
            headers={"Authorization": "Bearer test-token"},
            content_type="multipart/form-data",
        )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]

    job = None
    deadline = time.time() + 10
    while time.time() < deadline:
        poll = client.get(
            f"/api/analyses/{job_id}",
            headers={"Authorization": "Bearer test-token"},
        )
        assert poll.status_code == 200
        job = poll.get_json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert job is not None
    assert job["status"] == "completed"
    assert job["result"]["peak_person_count"] == 2
    assert job["result"]["sampled_detection_count"] == 6
    assert job["result"]["event_type"] == "Uncertain"
    assert job["result"]["event_classification"]["predicted_category"] == "Presentation"
    assert "2 person(s)" in job["result"]["narrative"]
    assert "visible-person figure is a peak per-frame count" in job["result"]["narrative"]
    assert len(yolo_calls) == len(frames)


def test_analyze_route_rejects_unreadable_video_bytes_with_400(monkeypatch, tmp_path):
    from contextlib import nullcontext

    class FakeCursor:
        def execute(self, query, params=None):
            pass

        def fetchone(self):
            return {
                "id": "user-1",
                "full_name": "Test User",
                "email": "test@example.com",
            }

    class FakeConnection:
        def cursor(self):
            return nullcontext(FakeCursor())

    monkeypatch.setattr(backend, "ensure_database_initialized", lambda: True)
    monkeypatch.setattr(
        backend,
        "get_db_connection",
        lambda: nullcontext(FakeConnection()),
    )
    monkeypatch.setattr(backend, "UPLOAD_DIR", tmp_path / "uploads")
    backend.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    response = backend.app.test_client().post(
        "/api/analyze",
        data={"video": (io.BytesIO(b"test video"), "event.mp4")},
        headers={"Authorization": "Bearer test-token"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "could not be read as a video" in response.get_json()["error"]