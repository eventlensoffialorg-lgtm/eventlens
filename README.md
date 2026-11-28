# EventLens

EventLens is a React/Vite application with a Flask video-analysis API. It uses
local MySQL (`eventlens_db`) with Gmail SMTP OTP verification for accounts,
YOLOv8 for object detection, MobileNetV2 + MLP / R3D-18 models for event and
activity analysis, and offline natural-language templates + ReportLab PDF export
for reports.

## Local development

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Configure `VITE_API_BASE_URL=http://127.0.0.1:5000/api` in `frontend/.env` or `frontend/.env.local`.

Backend (from the repository root):

```powershell
.\.venv\Scripts\Activate.ps1
cd backend
python app.py
```

Configure MySQL (`MYSQL_HOST=localhost`, `MYSQL_PORT=3306`, `MYSQL_USER`,
`MYSQL_PASSWORD`, `MYSQL_DATABASE=eventlens_db`) and Gmail SMTP credentials
(`SMTP_HOST`, `SMTP_PORT`, `SMTP_EMAIL`, `SMTP_PASSWORD`, `SMTP_FROM_NAME`) in
`backend/.env`. Reports and generated media are stored under `DATA_DIR` (the
default is the `backend` directory).

The API accepts videos up to 250 MB by default (`MAX_UPLOAD_BYTES`).

## Local event classifier preparation and training

The custom event classifier is not trained in the checked-in project. The label
CSV currently contains headers only. Do not train or use it as ground truth
until valid video paths and genuine labels have been added.

Use this CSV format:

```text
video_path,event_label,start_time,end_time,source,split
```

Prepare and validate records:

```powershell
cd backend
python prepare_dataset.py ..\EventLens_Datasets\labels\event_video_labels.csv --output ..\EventLens_Datasets\processed\event_video_labels_clean.csv
```

After adding legitimate labelled videos and train/validation/test splits, train:

```powershell
python train_event_classifier.py ..\EventLens_Datasets\processed\event_video_labels_clean.csv --output event_classifier.joblib
python evaluate_event_classifier.py event_classifier.json
```

Set `EVENT_MLP_MODEL_PATH` to the resulting model if using a custom `.joblib` path. The default
YOLOv8n model detects general objects only and must not be interpreted as event
classification evidence.

## Offline local ML operation

Event video analysis is designed to run locally with OpenCV, YOLO, MobileNetV2 +
MLP, and PyTorch/R3D-18. No external AI API key is required.

The custom event classifier remains untrained until a validated labelled video
dataset is available. In that case, EventLens returns an explicitly unclassified
result rather than inventing an event category.

## Reports and limitations

Reports are generated offline from model output using natural-language
templates; this release does not call an external generative NLP service.
Predictions are estimates from sampled frames, not a verified interpretation of
an entire video.

The API accepts videos up to 250 MB by default (`MAX_UPLOAD_BYTES`).
