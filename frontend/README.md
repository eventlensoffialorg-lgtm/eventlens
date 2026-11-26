# EventLens Frontend

React + Vite frontend for the EventLens event-video analysis project.

## Features implemented in this frontend
- Login, email OTP registration, and OTP password-reset screens using the Flask API.
- Protected application routes.
- Dashboard, analysis upload page, reports library and report detail page.
- Local video preview and basic file type/size validation.
- API client that attaches the current Flask bearer token.
- UI areas prepared for backend detection frames, bounding boxes and activity timeline.
- PDF download client for a backend endpoint.

## Setup
1. Open this `frontend` folder in VS Code.
2. Create `.env.local` by copying `.env.example`.
3. Keep the example's local API URL for local Flask development, or set
   `VITE_API_BASE_URL` to a reachable backend URL.
4. In a terminal run:
   ```bash
   npm install
   npm run dev
   ```
5. Open the URL printed by Vite, normally http://localhost:5173.

### Using another computer on the same network

`.env.local` is intentionally excluded from Git. On each computer, create its
own `frontend/.env.local` from `.env.example` and set `VITE_API_BASE_URL` to a
backend URL reachable from that computer. Restart Vite after changing it.

To use one running dev server from another device on your Wi-Fi, start Vite on
the host computer with `npm run dev`, then open
`http://<host-computer-LAN-IP>:5173` on the other device. Allow port 5173 through
the host computer's firewall for your private network if prompted. Do not expose
the development server to the public internet.

For local development on another device, set `VITE_API_BASE_URL` to an API
address that device can reach. The Pages deployment uses the repository base
path and Render API URL configured by GitHub Actions.

## Authentication
Registration sends an email OTP through the Flask API. After OTP verification,
the user creates a password; the backend stores the account and session in
TiDB. Password reset uses the same email OTP flow. Do not put database or SMTP
credentials in frontend variables.

## Backend integration contract
This frontend expects these Flask endpoints:
- `POST /api/analyze` with multipart field `video`
- `GET /api/reports`
- `GET /api/reports/:id`
- `GET /api/reports/:id/pdf` returning PDF bytes

The analysis response may contain:
```json
{
  "annotated_frame_url": "https://...",
  "detections": [
    {"class_name": "person", "confidence": 0.91, "bbox": [12, 15, 42, 80]}
  ],
  "events": [
    {"activity_name": "presentation", "start_seconds": 12, "end_seconds": 44, "confidence": 0.78}
  ]
}
```
Bounding box coordinates in this UI are percentages (0–100) relative to the image. Convert pixel coordinates to percentages in the backend or adapt the component.

## Important
The GitHub Pages workflow sets `VITE_API_BASE_URL` to
`https://eventlens-xqy5.onrender.com/api`. The API client attaches the
Flask-issued bearer token to authenticated requests. Do not commit `.env.local`.
