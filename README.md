# AnnotationTool

A full-stack, offline-capable image annotation desktop application with
built-in model training and auto-annotation. Runs on **Windows, macOS, and Linux**.

## Architecture

```
annotation_tool/
├── backend/          FastAPI REST API (runs embedded in dev mode)
│   └── app/
│       ├── api/      Routers: auth, projects, images, annotations, export, autoannotate
│       ├── core/     Config, security (JWT/bcrypt), storage abstraction
│       ├── ml/       BaseDetector interface, CustomModelAdapter, TrainingManager
│       ├── models/   SQLAlchemy ORM
│       └── schemas/  Pydantic schemas
├── frontend/
│   ├── main.py       Entry point — embedded backend thread + PySide6 GUI
│   ├── api_client.py Typed HTTP client
│   ├── tools/        BBox, Polygon, Drag tools + Undo/Redo (Command pattern)
│   ├── ui/           Login, Signup, Project Dashboard, Annotation Canvas, Export
│   └── resources/    Dark/Light QSS themes
├── detc-core/
│   ├── model/        DETC architecture: backbone / neck / head / blocks
│   ├── utils/        Dataset loader, loss, metrics, visualisation
│   └── train.py      Training script (class-agnostic — classes come from your project)
└── build/
    └── annotation_tool.spec
```

The detection model is **fully class-agnostic**: it trains on whatever classes
you create in your project. There are no built-in class names.

---

## Quick Start (all platforms)

### 1. Prerequisites
- Python 3.11+
- (Recommended) Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

(This includes PyTorch and OpenCV, which are required for model training and
auto-annotation. On Apple Silicon Macs the standard `pip` wheels work out of
the box — no extra steps.)

### 3. Run the app

```bash
python frontend/main.py
```

The first time you run it:
- A SQLite database is created at `~/AnnotationTool/data/annotation_tool.db`
  (`%USERPROFILE%\AnnotationTool\...` on Windows)
- Images are stored in `~/AnnotationTool/storage/`
- Trained models are saved to `~/AnnotationTool/models/<project>/v<N>/best.pt`
- The first user you create is automatically an **admin**
- The embedded FastAPI backend starts on `http://127.0.0.1:8765`
- API docs: http://127.0.0.1:8765/docs (open in browser while app is running)

---

## Auto-Annotation Workflow

1. **Create a project** and add your label classes (any names you want).
2. **Upload images** in the dashboard.
3. **Manually annotate at least 50 images** and mark the annotations as reviewed.
4. Open **🤖 Auto-Annotate → Auto-Annotation Loop**:
   - Choose **epochs** (50–70) and the **train/validation split**
     (e.g. 80% train / 20% val — you decide the percentage).
   - Click **Train Model**. Training runs in the background; the live log
     streams into the dialog. The best checkpoint (`best.pt`) is kept
     automatically.
   - Click **💾 Save best.pt to Computer…** to export the trained weights
     anywhere on your machine.
   - Click **Auto-Annotate All Unannotated Images** to let the model annotate
     the rest of your dataset.
5. **Review** the auto-annotations, fix mistakes, mark as reviewed, and train
   again — each round improves accuracy.

---

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
APP_ENV=development

# Change this before deploying!
SECRET_KEY=your_very_long_random_secret_key

# Optional: override the model directory (defaults to the bundled DETC model)
# CUSTOM_MODEL_DIR=/absolute/path/to/model/directory

# Production DB (PostgreSQL)
# DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname

# Production storage (S3)
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION=us-east-1
# S3_BUCKET_NAME=my-annotation-bucket
```

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `B` | Bounding box tool |
| `P` | Polygon tool |
| `Esc` | Select/pan tool |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Delete` | Delete selected annotation |
| `Ctrl+S` | Save annotations |
| `Ctrl+T` | Toggle dark/light theme |
| `Ctrl+F` | Fit image to view |
| Mouse wheel | Zoom in/out |
| Double-click | Close polygon (polygon mode) |
| `Enter` | Close polygon (polygon mode) |
| `Backspace` | Remove last polygon vertex |

(On macOS use `Cmd` in place of `Ctrl`.)

---

## Training the Model Standalone (CLI)

The training script can also be run directly on any YOLO-format dataset:

```bash
python train.py --data /path/to/dataset --classes my_class_a,my_class_b \
    --model n --epochs 60 --batch 8 --imgsz 640 --save-dir runs/train
```

`--classes` is required — the model has no default class list.

---

## Build .exe (Windows)

```powershell
pip install pyinstaller
pyinstaller build/annotation_tool.spec
```

Output: `dist/AnnotationTool.exe` — single executable, no Python install needed.

### Installer (optional)

To also produce a proper installer wizard (license page, install location,
Start Menu + desktop shortcut, Add/Remove Programs entry) instead of
distributing the raw `.exe`, install [Inno Setup](https://jrsoftware.org/isinfo.php)
and run:

```powershell
iscc build\installer.iss
```

Output: `dist_installer\AnnotationToolSetup.exe`. Installs per-user under
`%LocalAppData%\Programs\AnnotationTool` — no admin rights required. The
CI build (`.github/workflows/build-windows.yml`) does this automatically
and uploads both artifacts.

---

## Build .app (macOS)

PyInstaller can't cross-compile — this has to run on an actual Mac (the
CI build uses a `macos-latest` runner, which is Apple Silicon/arm64; the
resulting app only runs natively on Apple Silicon Macs, not Intel ones):

```bash
pip install pyinstaller
pyinstaller build/annotation_tool.spec
```

Output: `dist/AnnotationTool.app`. The same spec file drives both
platforms — it detects macOS and wraps the build in a proper `.app`
bundle (icon, `Info.plist`) instead of a bare Windows `.exe`.

To also produce a `.dmg` (the drag-to-Applications installer most Mac
apps ship as):

```bash
brew install create-dmg
create-dmg --volname "InSiSo Model Bench" "dist/InSiSoModelBench.dmg" "dist/AnnotationTool.app"
```

The CI build (`.github/workflows/build-macos.yml`) does this automatically
and uploads both the `.app` and the `.dmg`. Like the unsigned Windows
build, launching an unsigned `.app` triggers Gatekeeper's "can't verify
developer" warning — right-click → Open (instead of double-clicking)
bypasses it; a proper fix needs an Apple Developer ID certificate for
signing and notarization.

---

## Production Deployment

1. Run the FastAPI backend on a server:
   ```bash
   APP_ENV=production uvicorn backend.app.main:app --host 0.0.0.0 --port 8765
   ```

2. Set `BACKEND_URL=https://your-server.com/api` in the `.env` distributed with the `.exe`.

3. No rebuild needed to switch environments — the `.exe` reads the config file.

---

## Export Formats

| Format | File | Notes |
|---|---|---|
| COCO JSON | `.json` | Standard for most ML frameworks |
| YOLO TXT | `.zip` of `.txt` files | One file per image + `classes.txt` |
| Pascal VOC XML | `.zip` of `.xml` files | One file per image |
| Full Dataset ZIP | `.zip` | Images + COCO annotations + class list |

---

## License

Proprietary — © 2026 INSISO TECHNOLOGIES PRIVATE LIMITED. All rights reserved.
See [LICENSE](LICENSE) for terms. The `detc-core/` model architecture ships
under its own [LICENSE](detc-core/LICENSE) and [NOTICE](detc-core/NOTICE)
(same proprietary terms).
