# 🛑 Stop Sign Detection and Distance Estimation using Stereo Vision and YOLO

## 📌 Overview

This project implements a robust and efficient system for **real-time stop sign detection and distance estimation** using a combination of deep learning (YOLO) and classical stereo vision techniques.

Developed with **affordability and high performance** in mind, this solution is ideal for applications in autonomous driving, advanced driver-assistance systems (ADAS), and robotics — where accurate environmental perception is critical.

Unlike traditional object detection systems that only identify objects, this project leverages **stereoscopic triangulation** to provide precise depth information, giving vehicles or robots a crucial understanding of their surroundings in 3D space.

---

## 🚀 Key Features & Advantages

- **Hybrid Approach for Enhanced Perception**  
  Combines YOLO for real-time object detection with Stereo Vision for precise distance estimation.

- **High Detection Accuracy**  
  The YOLO model achieves **over 80%** accuracy in detecting stop signs, even in complex scenes.

- **Accurate Distance Estimation**  
  Uses the `Stereo SGBM (Semi-Global Block Matching)` algorithm to compute disparity maps and apply the triangulation formula:  
  `Z = (f × B) / d`

- **Cost-Effective Alternative**  
  Requires only stereo cameras — making it significantly cheaper than LIDAR or RADAR.

- **Rich Visual Data**  
  Unlike LIDAR or RADAR, stereo cameras offer full-color and texture-rich images, enabling deeper scene understanding.

- **Real-Time Performance**  
  Efficient YOLO inference + optimized stereo algorithms enable fast processing.

- **Modular and Maintainable Codebase**  
  Built around the `StereoYOLOProcessor` class for clarity, modularity, and ease of future updates.

---

## ⚙️ Technical Details

### 🔍 Models and Algorithms

- **YOLO (You Only Look Once)** – Fast and accurate object detection for bounding boxes and labels.
- **Stereo SGBM (Semi-Global Block Matching)** – Computes pixel-wise disparity between stereo pairs.
- **Triangulation Formula** – Converts disparity to distance:  
  `Z = (focal_length × baseline) / disparity`

### 📸 How It Works

1. **Stereo Image Capture** – Left and right frames are captured from a calibrated stereo camera pair.  
2. **Image Rectification** – Aligns images along epipolar lines to prepare for matching.  
3. **Stop Sign Detection (YOLO)** – YOLO processes one image (typically left) to detect and localize stop signs.  
4. **Disparity Map Generation (SGBM)** – Disparity computed from stereo image pair using SGBM.  
5. **Distance Estimation** – The average disparity inside the bounding box is used to calculate the distance.

---

## 📁 Project Structure

```
STOP_SIGN_PYTHON/
├── .pycache__/
├── .venv/                      # Python virtual environment
├── extracted_tesla_frames/     # Processed or raw frame data
├── STOP_LEFT_frames/           # Left image frames
├── STOP_RIGHT_frames/          # Right image frames
├── scene0/
├── scene1/
├── video/
│   ├── LEFT.mp4
│   └── RIGHT.mp4
├── config.txt                  # Configuration (model paths, thresholds, etc.)
├── detection.py                # Main processing script (detection + depth)
├── last.pt                     # YOLO model weights
├── packages.py                 # Custom modules
├── README.md                   # This file
└── video.py                    # Video capture/processing
.gitattributes
.gitignore
```

> **Note:** The main class `StereoYOLOProcessor` is likely defined in `detection.py`.

---

## 🛠️ Getting Started

### 1️⃣ Prerequisites

Make sure the following are installed:

- Python 3.8+
- Git

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/HodayaMoyal/Streo_Stop_Sign.git
cd STOP_SIGN_PYTHON
```

### 3️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

### 4️⃣ Activate the Environment

**Windows:**

```bash
.\.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source ./.venv/bin/activate
```

### 5️⃣ Install Dependencies

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, install manually:

```bash
pip install opencv-python numpy ultralytics configparser
```

### 6️⃣ Download Models and Calibration Files

Ensure the following:

- `last.pt` (YOLO weights) is in the project root.
- `camera_calibration.yml` is available (with intrinsic/extrinsic params).

> If not calibrated yet — see the **Calibration** section.

### 7️⃣ Configure `config.txt`

Update settings such as:

```
model_path=last.pt
confidence_threshold=0.5
calibration_file=camera_calibration.yml
num_disparities=64
block_size=11
```

### 8️⃣ Run the Project

Run the main script:

```bash
python detection.py
```

> The script will output detected stop signs, bounding boxes, and estimated distances. You can optionally modify it to save output frames.

---

## 🧪 Calibration

Stereo calibration is essential. Use OpenCV’s calibration tools or a dedicated script (e.g., `calibrate.py`) to generate `camera_calibration.yml`.

---

## 🤝 Contributing

We welcome contributions!

```bash
# Fork and clone
git checkout -b feature/my-feature
# Make your changes
git commit -m "Add new feature"
git push origin feature/my-feature
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.
