# FacePy

FacePy is a small Python webcam app that tracks a face with MediaPipe Face Mesh, draws line contours around the eyes and mouth, and plays a sound when the mouth opens wide.

## Setup

MediaPipe may lag behind the newest Python release, so Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Controls:

- Press `Q` or `Esc` to quit.
- If you have multiple cameras, try `python app.py --camera 1`.
- If the sound fires too easily or too rarely, adjust `--threshold`; lower values are more sensitive.

Examples:

```powershell
python app.py --threshold 0.30
python app.py --camera 1 --cooldown 1.2
```

The default sound is generated automatically at `assets/mouth_open.wav` on first run.
