# Sign Language Recognition System

A real-time **ASL / FSL sign language recognition and learning system** with:
- 🤖 **ML Recognition** — CNN + Random Forest ensemble (96%+ accuracy)
- 🎯 **AR Learning Modes** — 4 augmented reality modes for practice
- 📊 **Real-time Feedback** — Accuracy scoring with visual guidance
- 🗄️ **Database-backed** — Supabase for landmark storage
- 🎨 **3D Visualization** — Interactive PyVista hand models

Built as a complete thesis project demonstrating computer vision, machine learning, and augmented reality.

---

## 📋 Table of Contents

- [Requirements](#requirements)
- [Quick Setup](#quick-setup)
- [Detailed Setup Guide](#detailed-setup-guide)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Features](#features)
- [Troubleshooting](#troubleshooting)
- [For Thesis/Academic Use](#for-thesisacademic-use)

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| **Python 3.10** | 3.9 / 3.11 may work but untested |
| **Webcam** | Any USB or built-in webcam |
| **Supabase account** | Free tier sufficient ([sign up](https://supabase.com)) |
| **Dataset** | ASL/FSL images organized by letter/sign |
| **~4 GB disk space** | For dependencies + TensorFlow |
| **OS** | Windows 10+, macOS 10.14+, or Linux |

**Optional (for ARuco mode):**
- Printer for ARuco marker (can use phone screen as alternative)

**Optional (for phone-anchored AR):**
- YOLO model: `pip install ultralytics`

---

## Quick Setup

**For the impatient:**

```bash
# 1. Clone and enter directory
git clone https://github.com/YOUR_USERNAME/sign-language-system.git
cd sign-language-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Supabase
cp .env.example .env
# Edit .env with your Supabase credentials

# 5. Setup database
# Copy schema.sql contents to Supabase SQL Editor and run

# 6. Ingest dataset
python -m tools.ingest_dataset --path ./dataset --language ASL --dry-run
python -m tools.ingest_dataset --path ./dataset --language ASL

# 7. Train models
python -m tools.train_models --language ASL

# 8. Run!
python main.py
```

Continue reading for detailed instructions...

---

## Detailed Setup Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/sign-language-system.git
cd sign-language-system
```

---

### Step 2: Create Virtual Environment

A virtual environment keeps this project's packages isolated.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

**To deactivate later:**
```bash
deactivate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**This installs:**
- OpenCV (4.8+) — Computer vision
- MediaPipe (0.10.9) — Hand tracking
- TensorFlow (2.15.0) — CNN model
- scikit-learn — Random Forest classifier
- PyVista — 3D visualization
- Supabase client — Database connection
- Ultralytics (optional) — YOLO for phone detection

**Verify installation:**
```bash
python -c "import cv2, mediapipe, tensorflow, sklearn, pyvista, supabase; print('✓ All core packages OK')"
```

**Optional (for phone AR):**
```bash
pip install ultralytics
python -c "from ultralytics import YOLO; print('✓ YOLO installed')"
```

---

#### 4c. Add Credentials to Config

```python
# config.py
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5c..."
```

---

### Step 7: Train ML Models

```bash
python -m tools.train_models --language ASL
```

**This trains:**
1. **CNN** — Convolutional Neural Network for pattern recognition
2. **Random Forest** — Ensemble classifier for final prediction

**Training process:**
```
[Training] Loading data from Supabase...
[Training] Loaded 22,570 samples (26 classes)
[Training] Normalizing landmarks (palm-scale)...
[Training] Applying mirror augmentation (2× data)...
[Training] Training CNN...
Epoch 1/50: loss=2.4123, acc=0.3421
...
Epoch 50/50: loss=0.1234, acc=0.9614
✓ CNN trained (96.14% accuracy)

[Training] Training Random Forest...
✓ Random Forest trained (97.23% accuracy)

[Training] Testing ensemble...
✓ Ensemble accuracy: 96.87%

Models saved:
  - models/cnn_asl.h5
  - models/rf_asl.pkl
```

**Training time:**
- Small dataset (1K samples): 2-5 minutes
- Medium dataset (10K samples): 10-20 minutes
- Large dataset (50K samples): 30-60 minutes

**Optional: Hyperparameter tuning**
```bash
python -m tools.train_models --language ASL --tune
```

Finds best Random Forest parameters (adds 10-15 minutes).

**Verify models exist:**
```bash
ls models/
# Should show: cnn_asl.h5, rf_asl.pkl
```

---

### Step 8: Generate ARuco Marker (Optional)

Only needed if using ARuco AR mode:

```bash
python -m tools.generate_aruco
```

**Output:**
```
✓ ARuco marker generated: aruco_marker_id0.png

Printing instructions:
  1. Open aruco_marker_id0.png
  2. Print at 10cm x 10cm size
  3. Place on flat surface
```

---

## How to Run

### Main Application

```bash
python main.py
```

**You'll see:**
```
┌─────────────────────────────────────┐
│ Sign Language Recognition System    │
├─────────────────────────────────────┤
│ 1. ASL (American Sign Language)     │
│ 2. FSL (Filipino Sign Language)     │
└─────────────────────────────────────┘
```

Choose language, then:

```
┌─────────────────────────────────────┐
│ 1. Sign → Word (Recognition)        │
│ 2. Word → Sign (Learning)           │
└─────────────────────────────────────┘
```

---

### Mode 1: Sign → Word (Recognition)

**Recognize signs in real-time**

```bash
python main.py → ASL → Sign → Word
```

**Features:**
- Live camera feed
- Real-time predictions (30 FPS with frame-skip caching)
- Top-3 prediction bars with confidence
- Caption history (holds signs for 2 seconds)
- Recording capability (R key)

**Controls:**
- **R** — Start/stop recording
- **ESC** — Exit

**Performance:**
- Frame skip: Every 3rd frame processed
- FPS: 25-28 (smooth)
- Accuracy: 96%+ (on test set)

---

### Mode 2: Word → Sign (Learning)

**Learn signs with AR guidance**

```bash
python main.py → ASL → Word → Sign
```

**Four AR modes available:**

#### A. 2D Hand AR (Fastest)
- 2D skeleton overlay
- Floats above your hand
- Yellow skeleton = reference
- Green skeleton = your hand
- 30 FPS

#### B. 3D Mesh AR (Phone-Anchored)
- Full 3D mesh
- Anchored to detected phone (YOLO)
- Smooth tracking
- WASD rotation controls
- 20-25 FPS

**Requirements:**
```bash
pip install ultralytics  # For YOLO phone detection
```

**Controls:**
- **A/D** — Rotate left/right
- **W/S** — Tilt up/down
- **Q/E** — Zoom in/out
- **R** — Reset view
- **ESC** — Exit

#### C. ARuco AR (Most Precise) [EXPERIMENTAL]
- Marker-based (print required)
- Sub-pixel accuracy (±1 pixel)
- Academic gold standard
- Same controls as phone AR

**Setup:**
```bash
python -m tools.generate_aruco  # Generate marker
# Print at 10cm × 10cm
```

#### D. 3D Viewer (Study Mode)
- Separate PyVista 3D window
- Rotatable with mouse
- Accuracy feedback in camera window
- Best for detailed study

---

### Utility Scripts

#### Check Database Connection

```bash
python -m tools.check_database
```

Shows:
- Connection status
- Table statistics
- Sample counts per language

#### Ingest Dataset

```bash
# Dry run (test)
python -m tools.ingest_dataset --path ./dataset --language ASL --dry-run

# Real ingestion
python -m tools.ingest_dataset --path ./dataset --language ASL

# Different language
python -m tools.ingest_dataset --path ./fsl_dataset --language FSL
```

#### Train Models

```bash
# Standard training
python -m tools.train_models --language ASL

# With hyperparameter tuning
python -m tools.train_models --language ASL --tune

# For FSL
python -m tools.train_models --language FSL
```

#### Generate ARuco Marker

```bash
python -m tools.generate_aruco
```

Creates `aruco_marker_id0.png` (500×500 pixels with border)

---

## Project Structure

```
sign-language-system/
│
├── main.py                      # Entry point
├── config.py                    # Supabase credentials (GITIGNORED!)
├── .env                         # Environment variables (GITIGNORED!)
├── .env.example                 # Template for .env
├── requirements.txt             # Python dependencies
├── schema.sql                   # Database schema
├── .gitignore                   # Git ignore rules
│
├── core/                        # Core logic
│   ├── caption.py               # Caption history management
│   ├── db.py                    # Database operations
│   ├── inference.py             # ML inference with caching
│   ├── models.py                # Model loading/training
│   ├── recognition.py           # Landmark normalization
│   └── recording.py             # Video recording
│
├── modes/                       # Application modes
│   ├── sign_to_word.py          # Recognition mode
│   ├── word_to_sign.py          # Learning mode router
│   ├── word_to_sign_hand_2d.py  # 2D skeleton AR
│   ├── word_to_sign_hand_3d_final.py  # 3D phone AR
│   ├── word_to_sign_aruco.py    # ARuco marker AR
│   └── __init__.py
│
├── ui/                          # User interface
│   ├── dialogs.py               # Tkinter dialogs
│   ├── overlays.py              # OpenCV overlays
│   ├── word_picker.py           # Letter/word selection
│   └── __init__.py
│
├── ar/                          # AR utilities
│   ├── aligner.py               # Similarity calculation
│   ├── mesh_renderer.py         # 3D mesh to 2D image
│   └── __init__.py
│
├── visualization/               # 3D visualization
│   ├── hand_3d_combined.py      # PyVista 3D viewer
│   └── __init__.py
│
├── tools/                       # Utility scripts
│   ├── check_database.py        # Verify DB connection
│   ├── ingest_dataset.py        # Load images to DB
│   ├── train_models.py          # Train ML models
│   ├── generate_aruco.py        # Generate marker
│   └── __init__.py
│
├── models/                      # Trained models (GITIGNORED!)
│   ├── cnn_asl.h5
│   ├── rf_asl.pkl
│   ├── cnn_fsl.h5
│   └── rf_fsl.pkl
│
├── .cache/                      # Cached data (GITIGNORED!)
│   └── dataset_*.pkl
│
└── recordings/                  # Saved recordings (GITIGNORED!)
    └── recording_*.avi
```

**Total:** ~3,800 lines of Python across 25 files

---

## Features

### Recognition (Sign → Word)

- ✅ Real-time hand detection (MediaPipe)
- ✅ CNN + Random Forest ensemble
- ✅ Frame-skip caching (3× FPS boost)
- ✅ Top-3 predictions with confidence
- ✅ Caption history (2-second hold)
- ✅ Recording capability
- ✅ 96%+ accuracy

### Learning (Word → Sign)

- ✅ 4 AR modes (2D, 3D Phone, ARuco, Viewer)
- ✅ Real-time accuracy feedback
- ✅ Cosine similarity scoring
- ✅ Hand-anchored visualization
- ✅ Interactive controls (WASD, QE, R)
- ✅ Smooth tracking algorithms

### Data Pipeline

- ✅ Supabase integration
- ✅ Automated ingestion
- ✅ Palm-scale normalization
- ✅ Mirror augmentation
- ✅ Disk caching

### ML Pipeline

- ✅ CNN architecture (Conv2D + Dense)
- ✅ Random Forest ensemble
- ✅ Hyperparameter tuning
- ✅ Confusion matrix evaluation
- ✅ Class-balanced training

---

## Troubleshooting

### Installation Issues

**"No module named 'cv2'"**
```bash
pip install opencv-python opencv-contrib-python
```

**"TensorFlow not found"**
```bash
pip install tensorflow==2.15.0
```

**"MediaPipe version conflict"**
```bash
pip install mediapipe==0.10.9
```

### Database Issues

**"Failed to connect to Supabase"**
- Check `.env` file has correct credentials
- Verify `SUPABASE_URL` starts with `https://`
- Verify `SUPABASE_KEY` is service_role key (not anon)
- Test connection: `python -m tools.check_database`

**"Table 'landmark_samples' does not exist"**
- Run `schema.sql` in Supabase SQL Editor
- Refresh browser
- Check database → Tables in dashboard

### Training Issues

**"No samples found"**
- Run ingestion first: `python -m tools.ingest_dataset`
- Check database: `python -m tools.check_database`
- Verify samples uploaded (should show count > 0)

**"Model accuracy very low (<50%)"**
- Dataset too small (need 500+ samples per class)
- Poor quality images (hands not visible)
- Inconsistent hand poses in dataset
- Try with known-good dataset first (see Step 6a)

### Runtime Issues

**"Camera not found"**
- Check webcam connected
- Check permissions (Windows: Settings → Privacy → Camera)
- Try different camera index in code (change `cv2.VideoCapture(0)` to `(1)`)

**"Low FPS / Laggy"**
- Increase frame skip: Edit `sign_to_word.py` line 49: `frame_skip=5`
- Reduce resolution: Add `frame = cv2.resize(frame, (640, 480))`
- Close other camera apps

**"Phone not detected" (3D Phone AR)**
```bash
pip install ultralytics
```

**"ARuco marker not detected"**
- Print at correct size (10cm × 10cm)
- Ensure marker flat (no curves)
- Good lighting (no shadows, glare)
- Update OpenCV: `pip install --upgrade opencv-python`

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

