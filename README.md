# EventLens

EventLens is a React/Vite frontend with a Flask video-analysis API. It uses
TiDB Cloud (MySQL-compatible) for accounts, OTPs, sessions, and reports; Gmail
SMTP for email verification; YOLOv8 for object detection; an MLP for event
classification; and offline natural-language templates for reports.

## Deployment

The Flask API remains deployed on Render with persistent storage for reports and
generated media. The existing Render Blueprint is preserved.

The GitHub Actions workflow at `.github/workflows/deploy.yml` builds the
frontend and deploys only `frontend/dist` to GitHub Pages. The Vite base path is
`/eventlens/`; the expected Pages URL is
`https://eventlensoffialorg-lgtm.github.io/eventlens/`. The workflow sets
`VITE_API_BASE_URL` to `https://eventlens-xqy5.onrender.com/api`.

The Flask backend is not published through GitHub Pages. Keep TiDB and Gmail
credentials in Render's environment settings, never in frontend `VITE_`
variables or the repository.

Render environment configuration uses `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`,
`MYSQL_PASSWORD`, `MYSQL_DATABASE`, `MYSQL_SSL`, `MYSQL_SSL_VERIFY_CERT`,
`MYSQL_SSL_VERIFY_IDENTITY`, and optionally `MYSQL_SSL_CA` and the database
timeouts. Gmail delivery uses `SMTP_EMAIL`, `SMTP_PASSWORD`, and the `SMTP_*`
host, port, and TLS/SSL settings supported by the backend. Set actual values in
Render only. Verify
`/api/health` and `/api/health/database` after a backend deployment.

## Local development

Frontend:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm ci
npm run dev
```

The frontend `.env.example` points to a local Flask API for development. The
GitHub Pages workflow injects the production Render API URL at build time.

Backend:

```powershell
cd backend
python -m pip install -r requirements.txt
# Configure TiDB and Gmail SMTP environment variables outside source control.
python app.py
```

Registration verifies email by OTP before password creation. The backend
validates its own bearer tokens against TiDB-backed sessions. Reports and
generated media are stored under `DATA_DIR` (Render mounts persistent storage
at `/var/data`).

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

Set `EVENT_MLP_MODEL_PATH` to the resulting model for deployment. The default
YOLOv8n model detects general objects only and must not be interpreted as event
classification evidence.

## Offline local ML operation

Event video analysis is designed to run locally with OpenCV, YOLO, and
PyTorch/R3D-18. No external AI API key is required.

The custom event classifier remains untrained until a validated labelled video
dataset is available. In that case, EventLens returns an explicitly unclassified
result rather than inventing an event category.

## Reports and limitations

Reports are generated offline from model output using natural-language
templates; this release does not call an external generative NLP service.
Predictions are estimates from up to eight sampled frames, not a verified
interpretation of an entire video. Uploaded source videos are deleted after
analysis; annotated media and reports are kept on the Render persistent disk.

The API accepts videos up to 250 MB by default. Change `MAX_UPLOAD_BYTES` in
Render if needed, while considering processing time and disk capacity.
