import numpy as np

import app as backend


def test_report_uses_mlp_and_withholds_uncalibrated_action_labels(
    monkeypatch,
    tmp_path,
):
    video_path = tmp_path / "uploaded-unique.mp4"
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    passed_video_paths = []
    saved_reports = []
    job_id = "action-report-safety-test"

    monkeypatch.setattr(
        backend,
        "analysis_jobs",
        {job_id: {"id": job_id, "user_id": "test-user", "status": "processing"}},
    )
    monkeypatch.setattr(
        backend,
        "extract_sample_frames",
        lambda *_args, **_kwargs: ([frame], [0.0], 1.0),
    )
    monkeypatch.setattr(
        backend.pipeline,
        "run_yolo",
        lambda _frame: [{"class_name": "person", "confidence": 0.9}],
    )
    monkeypatch.setattr(
        backend.pipeline,
        "process_detections",
        lambda *_args, **_kwargs: {
            "event_type": "unknown",
            "confidence": None,
            "summary": "Initial detection summary.",
            "model_quality": "unavailable",
            "top_objects": [{"class_name": "person", "count": 1}],
            "peak_person_count": 1,
        },
    )
    monkeypatch.setattr(
        backend.pipeline,
        "classify_video_features",
        lambda *_args, **_kwargs: {
            "event_type": "speech",
            "confidence": 0.99,
            "summary": "Legacy event prediction.",
            "model_quality": "trained",
        },
    )
    monkeypatch.setattr(
        backend.pipeline,
        "track_people",
        lambda _frames: {
            "unique_tracked_people": 1,
            "peak_tracked_people": 1,
            "tracking_status": "ready",
        },
    )
    monkeypatch.setattr(
        backend.action_recognizer,
        "predict_video",
        lambda path: (
            passed_video_paths.append(path)
            or {
                "action_predictions": [
                    {
                        "action_name": "PlayingDhol",
                        "confidence": None,
                        "model_score": 0.91,
                        "score_type": "uncalibrated LinearSVC decision score",
                    },
                    {
                        "action_name": "PlayingPiano",
                        "confidence": None,
                        "model_score": 0.84,
                        "score_type": "uncalibrated LinearSVC decision score",
                    },
                ],
                "action_clip_details": [
                    {"predictions": [{"action_name": "WritingOnBoard"}]}
                ],
                "action_timeline": [
                    {"activity_name": "PlayingDhol", "start_seconds": 0, "end_seconds": 1}
                ],
                "action_summary": "PlayingDhol (1 clip)",
                "_video_feature": np.zeros(512, dtype=np.float32),
            }
        ),
    )
    monkeypatch.setattr(
        backend.event_mlp_classifier,
        "predict_video",
        lambda path: (
            passed_video_paths.append(path)
            or {
                "status": "success",
                "classification_status": "uncertain",
                "event_type": "Uncertain",
                "predicted_category": "Dance",
                "assigned_category": None,
                "confidence": None,
                "model_score": 0.88,
                "score_type": "uncalibrated_softmax_score",
                "confidence_calibrated": False,
                "model_quality": "unverified",
                "features_dimension": 5120,
                "summary": (
                    'The MLP top candidate is "Dance" with raw softmax score 0.880. '
                    "This score is not calibrated; no event category has been assigned."
                ),
            }
        ),
    )
    monkeypatch.setattr(backend.action_recognizer, "status", "ready")
    monkeypatch.setattr(backend, "annotate_frame", lambda image, *_args: image)
    monkeypatch.setattr(backend.cv2, "imwrite", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(backend, "create_full_detection_video", lambda *_args: None)
    monkeypatch.setattr(
        backend,
        "save_report",
        lambda report, _user_id: saved_reports.append(report),
    )

    backend.process_analysis_job(job_id, "test-user", video_path, video_path.name)

    job = backend.analysis_jobs[job_id]
    assert job["status"] == "completed"
    report = job["result"]["report"]
    assert passed_video_paths == [video_path, video_path]
    assert report["event_type"] == "Uncertain"
    assert report["event_classification"]["classification_status"] == "uncertain"
    assert report["event_classification"]["predicted_category"] == "Dance"
    assert report["event_classification"]["assigned_category"] is None
    assert report["confidence"] is None
    assert report["model_quality"] == "unverified"
    assert report["action_reporting_status"] == "uncertain"
    assert report["action_model_status"] == "ready"
    assert report["action_predictions"] == []
    assert report["action_clip_details"] == []
    assert report["action_timeline"] == []
    assert report["events"] == []
    assert "uncalibrated decision scores" in report["action_summary"]
    assert "uncalibrated decision scores" in report["narrative"]
    response = job["result"]["result"]
    assert response["event_type"] == "Uncertain"
    assert response["action_predictions"] == []
    assert response["action_timeline"] == []
    assert response["events"] == []
    assert all(
        label not in repr(report)
        for label in ("PlayingDhol", "PlayingPiano", "WritingOnBoard", "speech")
    )
    assert all(
        label not in repr(response)
        for label in ("PlayingDhol", "PlayingPiano", "WritingOnBoard", "speech")
    )
    assert saved_reports == [report]

    pdf = backend.build_pdf_report(report)
    assert pdf.startswith(b"%PDF-")
    assert all(
        label.encode("ascii") not in pdf
        for label in ("PlayingDhol", "PlayingPiano", "WritingOnBoard")
    )


def test_event_pipeline_initializes_without_legacy_warning_when_joblib_absent(
    monkeypatch,
    caplog,
):
    monkeypatch.delenv("EVENT_MLP_MODEL_PATH", raising=False)
    with caplog.at_level("WARNING"):
        pipeline = backend.EventPipeline()

    assert pipeline.model is None
    assert pipeline.model_quality == "unavailable"
    assert "No trained MLP model found" not in caplog.text


def test_modern_mlp_prediction_propagates_to_report_without_calling_legacy_classifier(
    monkeypatch,
    tmp_path,
):
    video_path = tmp_path / "cultural_clip.mp4"
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    saved_reports = []
    job_id = "mlp-propagation-test"

    monkeypatch.setattr(
        backend,
        "analysis_jobs",
        {job_id: {"id": job_id, "user_id": "test-user", "status": "processing"}},
    )
    monkeypatch.setattr(
        backend,
        "extract_sample_frames",
        lambda *_args, **_kwargs: ([frame], [0.0], 2.5),
    )
    monkeypatch.setattr(backend.pipeline, "run_yolo", lambda _frame: [])
    monkeypatch.setattr(
        backend.pipeline,
        "classify_video_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Legacy 512-D classifier should not run when modern MLP succeeds.")
        ),
    )
    monkeypatch.setattr(
        backend.pipeline,
        "track_people",
        lambda _frames: {
            "unique_tracked_people": None,
            "peak_tracked_people": 0,
            "tracking_status": "unavailable",
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
            "predicted_category": "Cultural",
            "assigned_category": None,
            "confidence": None,
            "model_score": 0.81,
            "score_type": "uncalibrated_softmax_score",
            "confidence_calibrated": False,
            "model_quality": "unverified",
            "model_variant": "production",
            "features_dimension": 5120,
            "raw_probabilities": {
                "Cultural": 0.81,
                "Dance": 0.09,
                "Presentation": 0.06,
                "Unknown": 0.04,
            },
            "rejection_status": "not_assessed",
            "rejection_reason": "No independently validated out-of-distribution rejection rule is configured.",
            "summary": (
                'The MLP top candidate is "Cultural" with raw softmax score 0.810. '
                "This score is not calibrated; no event category has been assigned."
            ),
        },
    )
    monkeypatch.setattr(backend.cv2, "imwrite", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(backend, "create_full_detection_video", lambda *_args: None)
    monkeypatch.setattr(
        backend,
        "save_report",
        lambda report, _user_id: saved_reports.append(report),
    )

    backend.process_analysis_job(job_id, "test-user", video_path, video_path.name)

    report = saved_reports[0]
    assert report["event_type"] == "Uncertain"
    assert report["event_classification"]["predicted_category"] == "Cultural"
    assert report["event_classification"]["model_variant"] == "production"
    assert report["event_classification"]["features_dimension"] == 5120
    assert report["model_quality"] == "unverified"
    assert "Cultural" in report["narrative"]
    assert "0.810" in report["narrative"]
    pdf = backend.build_pdf_report(report)
    assert pdf.startswith(b"%PDF-")
    assert b"0.810" in pdf


def test_unavailable_mlp_for_short_or_unrelated_video_does_not_fabricate_category(
    monkeypatch,
    tmp_path,
):
    video_path = tmp_path / "short_unrelated.mp4"
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    saved_reports = []
    job_id = "mlp-unavailable-test"

    monkeypatch.setattr(
        backend,
        "analysis_jobs",
        {job_id: {"id": job_id, "user_id": "test-user", "status": "processing"}},
    )
    monkeypatch.setattr(
        backend,
        "extract_sample_frames",
        lambda *_args, **_kwargs: ([frame], [0.0], 0.2),
    )
    monkeypatch.setattr(backend.pipeline, "run_yolo", lambda _frame: [])
    monkeypatch.setattr(
        backend.pipeline,
        "track_people",
        lambda _frames: {
            "unique_tracked_people": None,
            "peak_tracked_people": 0,
            "tracking_status": "unavailable",
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
            "status": "unavailable",
            "classification_status": "unavailable",
            "event_type": None,
            "predicted_category": None,
            "assigned_category": None,
            "confidence": None,
            "model_score": None,
            "confidence_calibrated": False,
            "model_quality": "unavailable",
            "message": "The MLP requires at least eight readable video frames.",
        },
    )
    monkeypatch.setattr(backend.cv2, "imwrite", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(backend, "create_full_detection_video", lambda *_args: None)
    monkeypatch.setattr(
        backend,
        "save_report",
        lambda report, _user_id: saved_reports.append(report),
    )

    backend.process_analysis_job(job_id, "test-user", video_path, video_path.name)

    report = saved_reports[0]
    assert report["event_type"] == "unknown"
    assert report["confidence"] is None
    assert report["model_quality"] == "unavailable"
    assert report["event_classification"]["classification_status"] == "unavailable"
    assert "at least eight readable video frames" in report["narrative"]
