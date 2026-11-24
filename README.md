# EventLens

EventLens is a React/Vite application with a Flask video-analysis API. It uses
Supabase Auth for accounts, YOLOv8 for object detection, an MLP for event
classification, and offline natural-language templates for reports.

## Deploy on Render

The root `render.yaml` defines two Render services:

- `eventlens-web`: the public React website.
- `eventlens-api`: the authenticated Flask API, with a persistent disk for
  reports and generated media.

To deploy:

1. Push this project to a GitHub repository you can access.
2. In Render, choose **New + > Blueprint** and connect that repository.
3. Review the services and costs before applying. The API uses a paid Standard
   instance and a 1 GB persistent disk so YOLO has memory and reports survive
   restarts. Render may ask you to add a payment method.
4. Enter the same Supabase project URL and publishable key for the backend
   (`SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`) and frontend
   (`VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`) when prompted.
   Never use a Supabase service-role/secret key in the frontend.
5. After the static site deploys, open its `onrender.com` URL and add it to
   Supabase Dashboard > Authentication > URL Configuration as the Site URL and
   an allowed Redirect URL (for example `https://your-site.onrender.com/**`).
6. Confirm email/password sign-in and the API health endpoint:
   `https://your-api.onrender.com/api/health`.

The frontend API host is wired to the backend host by the Render Blueprint.
The API validates Supabase access tokens and scopes reports/media to the
authenticated user.

## Local development

Frontend:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm ci
npm run dev
```

Set the Supabase project URL and publishable key in `frontend/.env.local`.

Backend:

```powershell
cd backend
python -m pip install -r requirements.txt
$env:SUPABASE_URL = "https://your-project.supabase.co"
$env:SUPABASE_PUBLISHABLE_KEY = "your-publishable-key"
python app.py
```

The browser sends the signed-in user's access token to the API. The backend
checks it with Supabase Auth. Reports and generated media are stored under
`DATA_DIR` (the default is the backend directory; Render mounts its persistent
disk at `/var/data`).

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
