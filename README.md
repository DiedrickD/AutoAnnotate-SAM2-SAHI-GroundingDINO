# Grounded-SAM2 Webapp

A small web app for running Grounded-SAM2 segmentation from a browser.

## Requirements
- Python 3.10+
- Node.js 18+

## Python dependencies
Install the Python packages listed in `requirements.txt`.

## Setup
1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Install Node dependencies:

```powershell
npm install
```

## Run (Windows)
Start the web app (Next.js) on port 3000:

```powershell
.\start-webapp.cmd
```

Open http://127.0.0.1:3000 in your browser.

## Notes
- The app may attempt to load large model files (YOLO, GroundingDINO, SAM2). Download or cache these models locally before running.
- `webapp-data/` and `public/results/` are ignored by default to avoid pushing large generated outputs. If you accidentally deleted `public/results/`, re-generate outputs by running the app and processing images.
