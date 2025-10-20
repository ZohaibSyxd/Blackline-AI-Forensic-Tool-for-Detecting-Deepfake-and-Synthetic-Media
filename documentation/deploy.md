# Deploying Blackline Forensics (Frontend + FastAPI)

This guide shows a minimal, production-leaning setup to deploy both the FastAPI backend and the Vite/React frontend.

Prereqs
- macOS/Linux server or cloud VM
- Python 3.10+
- Node 18+
- A domain or IP address
- Optional: AWS S3 bucket (recommended for asset storage)

## 1) Configure environment

Backend reads these env vars:
- DATABASE_URL (default sqlite:///./backend/data/dev.db)
- BL_JWT_SECRET (required in prod)
- STORAGE_BACKEND=local|s3 (default local)
- If s3:
  - STORAGE_BUCKET=<your-bucket>
  - AWS_REGION=<aws-region>
  - Optional: STORAGE_PREFIX=prod/

Frontend uses:
- VITE_API_BASE (e.g., https://api.example.com)

Mac zsh tip: paths with `!` (like this repo) need quoting. Prefer using absolute paths and quotes in scripts.

## 2) Backend: run behind a process manager

Create a Python venv and install:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Start with Uvicorn/Gunicorn
- Dev (hot reload):
```
.venv/bin/python -m uvicorn backend.src.api_server:app --reload --host 0.0.0.0 --port 8000
```
- Prod (Gunicorn + Uvicorn workers):
```
.venv/bin/python -m pip install gunicorn uvicorn
.venv/bin/gunicorn -k uvicorn.workers.UvicornWorker backend.src.api_server:app \
  --bind 0.0.0.0:8000 --workers 2 --timeout 120
```

Environment example (bash/zsh):
```
export DATABASE_URL="sqlite:///$(pwd)/backend/data/dev.db"
export BL_JWT_SECRET="change-me"
# S3 (optional)
# export STORAGE_BACKEND=s3
# export STORAGE_BUCKET=your-bucket
# export AWS_REGION=us-east-1
```

Static files served by the backend
- Derived overlays: /static/*
- Local raw assets (dev only): /assets/*

## 3) Frontend: build and serve

Install dependencies and build:
```
cd front-end
npm ci || npm install
VITE_API_BASE="https://api.example.com" npm run build
```
This produces `front-end/dist`.

Serve options:
- Nginx (recommended): serve `front-end/dist` and proxy /api to the backend
- Or simple static server (Caddy, `npx serve`, S3+CloudFront, etc.)

Nginx example:
```
server {
  listen 80;
  server_name example.com;

  root /var/www/blackline/front-end/dist;
  index index.html;

  location / {
    try_files $uri /index.html;
  }

  location /api/ {
    proxy_pass http://127.0.0.1:8000/api/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }

  # Optional: expose /static and /assets from backend
  location /static/ { proxy_pass http://127.0.0.1:8000/static/; }
  location /assets/ { proxy_pass http://127.0.0.1:8000/assets/; }
}
```

## 4) S3 storage (optional but recommended)

1) Set bucket CORS (replace origin):
```
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["PUT", "GET", "HEAD"],
    "AllowedOrigins": ["https://example.com"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```
2) Set env vars on backend (see step 2)
3) Frontend UploadCard will use presigned PUT -> confirm -> analyze

## 5) Systemd units (optional)

Create `/etc/systemd/system/blackline-api.service`:
```
[Unit]
Description=Blackline API
After=network.target

[Service]
WorkingDirectory=/opt/blackline
Environment=DATABASE_URL=sqlite:///./backend/data/dev.db
Environment=BL_JWT_SECRET=change-me
# Environment=STORAGE_BACKEND=s3
# Environment=STORAGE_BUCKET=your-bucket
# Environment=AWS_REGION=us-east-1
ExecStart=/opt/blackline/.venv/bin/gunicorn -k uvicorn.workers.UvicornWorker backend.src.api_server:app --bind 0.0.0.0:8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```

Serve frontend with Nginx pointing to `/opt/blackline/front-end/dist`.

## 6) Quick local run

- Backend:
```
# from repo root
/Users/nicolaiskogstad/\!PROJECTS/nskog-ITP-blacklineAIfrontend/.venv/bin/python -m uvicorn backend.src.api_server:app --reload --port 8000
```
- Frontend:
```
cd front-end
npm run dev
```
Note zsh special char: escape the `!` in the path or quote the whole path when running commands.

## 7) Troubleshooting

- 403 on S3 PUT: check bucket CORS and that the presigned URL method is PUT.
- CORS errors in browser: set VITE_API_BASE to your backend origin and configure backend CORSMiddleware allow_origins accordingly.
- Large uploads: consider putting Nginx in front with `client_max_body_size 512m;` and tune Uvicorn/Gunicorn timeouts.
- Models not available (torch, transformers): use the "stub" model or provision GPU/CPU resources and install dependencies.

## Deploy to Netlify (Frontend)

This repo includes `netlify.toml` to make Netlify builds zero-config:

- Build base: `front-end`
- Publish dir: `front-end/dist`
- Command: `npm run build`
- SPA fallback redirect (/* → /index.html)

Steps:
1) In Netlify, create a new site from Git and pick this GitHub repo.
2) Set environment variable `VITE_API_BASE` (Site settings → Build & Deploy → Environment). Example: `https://api.blackline-tech.me`.
  - Optional: If you enable the `/api/*` proxy in `netlify.toml`, set `VITE_API_BASE` to your site origin (e.g., `https://blackline-tech.me`) so requests go same-origin and are proxied.
3) Trigger a deploy. You’ll get a `*.netlify.app` URL.
4) Connect your Namecheap domain (blackline-tech.me) under Netlify → Domain management → Add domain.
  - Netlify will guide DNS setup. Typical records:
    - CNAME: `www` → your-site.netlify.app
    - Apex `@`: use Netlify DNS for easiest setup, or if staying on Namecheap DNS use ALIAS/ANAME pointing to your-site.netlify.app (if available) or Netlify’s documented apex method.
5) Once DNS propagates, Netlify issues TLS automatically.

Note: Keep backend CORS allow_origins aligned with your final site origin(s).

## Deploy backend to Render (FastAPI)

Render’s default Python (3.13 at the time of writing) is not compatible with our mediapipe constraint (requires < 0.11.x and wheels exist only up to Python 3.11). To ensure a compatible build:

1) Pin Python for Render by adding `backend/runtime.txt` with:
```
python-3.11.9
```

2) Create a new Web Service in Render
- Repository: this GitHub repo
- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn backend.src.api_server:app --host 0.0.0.0 --port $PORT`
- Instance: Standard or above (CPU is fine; GPU not required for stub models)

3) Set Environment Variables (Render → Settings → Environment)
- `BL_JWT_SECRET=change-me`
- `DATABASE_URL=sqlite:///./backend/data/dev.db` (or your Postgres URL)
- `STORAGE_BACKEND=local` (or `s3` and then set `STORAGE_BUCKET`, `AWS_REGION`, plus AWS creds/role)

4) CORS
- In production, tighten CORS in `backend/src/api_server.py` to your Netlify domain(s) instead of `"*"`.

5) Redeploy
- After saving runtime.txt and env vars, trigger a deploy. Verify `/api/health`.

Tip: If you build from the repo root by mistake, Render won’t find `runtime.txt`. Always set Root Directory to `backend` so Render picks up the correct Python version.
