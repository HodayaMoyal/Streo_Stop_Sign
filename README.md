🛑 Stop Sign Detection and Distance Estimation using Stereo Vision and YOLO
Overview
This project implements a robust and efficient system for real-time stop sign detection and distance estimation using a combination of deep learning (YOLO) and classical stereo vision techniques. Developed with affordability and high performance in mind, this solution is ideal for applications in autonomous driving, advanced driver-assistance systems (ADAS), and robotics where accurate environmental perception is critical.

Unlike traditional object detection systems that only identify objects, this project leverages stereoscopic triangulation to provide precise depth information, giving vehicles or robots a crucial understanding of their surroundings in 3D space.

Key Features & Advantages
Hybrid Approach for Enhanced Perception: The system integrates a state-of-the-art YOLO (You Only Look Once) model for rapid and highly accurate stop sign detection with a Stereo Vision pipeline for precise distance measurement. This hybrid approach combines the power of deep learning's object recognition capabilities with the geometric accuracy of stereo vision.

High Detection Accuracy: The YOLO model is specifically trained and fine-tuned to achieve an impressive detection accuracy of over 80% for stop signs, ensuring reliable identification even in complex scenarios.

Accurate Distance Estimation: Utilizing the Stereo SGBM (Semi-Global Block Matching) algorithm, the system generates high-quality disparity maps. These maps are then translated into real-world distances using the stereoscopic triangulation formula (
Z=
fracfcdotBd
), providing critical depth information to the autonomous system.

Cost-Effective Solution: By relying on standard, readily available stereo cameras, this project offers a significantly more economical alternative to expensive LIDAR or RADAR systems, making advanced perception capabilities accessible for a wider range of applications and research.

Rich Visual Information: Unlike LIDAR (which provides sparse point clouds) or RADAR (which offers low-resolution object detection), the camera-based approach provides rich visual information (color, texture, context) essential for comprehensive scene understanding and sophisticated decision-making in autonomous systems.

Real-time Performance: Both YOLO's efficient architecture and the optimized stereo matching algorithms contribute to near real-time processing capabilities, crucial for dynamic environments.

Modular and Maintainable Codebase: The project's architecture is built around a StereoYOLOProcessor class, promoting modularity, reusability, and ease of maintenance. This structured approach simplifies future enhancements, debugging, and collaboration.

Technical Details
Models and Algorithms Used:
YOLO (You Only Look Once): For fast and accurate 2D object detection (bounding boxes and class labels for stop signs).

Stereo Semi-Global Block Matching (SGBM): An efficient algorithm for computing disparity maps from rectified stereo image pairs. This algorithm balances accuracy and computational efficiency, making it suitable for real-time applications.

Stereoscopic Triangulation Formula: The core geometric principle used to convert disparity values (in pixels) into real-world distances (e.g., meters) based on camera focal length and baseline.

How it Works:
Stereo Image Capture: The system captures synchronized left and right images from a calibrated stereo camera pair.

Image Rectification: The captured images are rectified to align their epipolar lines, simplifying the stereo matching process.

Stop Sign Detection (YOLO): The YOLO model processes one of the rectified images (typically the left) to detect and classify stop signs, providing their 2D bounding box locations.

Disparity Map Generation (SGBM): The Stereo SGBM algorithm computes a disparity map from the rectified stereo image pair. This map represents the pixel-wise difference in horizontal position for corresponding points in the left and right images.

Distance Calculation (Triangulation): For each detected stop sign, the average disparity within its bounding box is extracted from the disparity map. This disparity value, along with the pre-calibrated camera focal length and baseline, is then fed into the stereoscopic triangulation formula to accurately estimate the real-world distance to the stop sign.

Project Structure
Based on your project structure image, it appears as follows:

.
├── STOP_SIGN_PYTHON/             # Root directory of your project
│   ├── .pycache__/
│   ├── .venv/                      # Python virtual environment
│   ├── extracted_tesla_frames/     # Likely raw or processed frames
│   ├── STOP_LEFT_frames/
│   ├── STOP_RIGHT_frames/
│   ├── scene0/                     # Potentially another set of frames or scenes
│   ├── scene1/
│   ├── video/                      # Directory containing video files (e.g., LEFT.mp4, RIGHT.mp4)
│   │   ├── LEFT.mp4
│   │   └── RIGHT.mp4
│   ├── config.txt                  # Configuration file for model paths, thresholds, etc.
│   ├── detection.py                # Main script for running the detection and estimation (likely your main.py)
│   ├── last.pt                     # Pre-trained YOLO model weights (e.g., yolovX_stop_sign.pt)
│   ├── packages.py                 # Possibly a custom package or module
│   ├── README.md                   # This file
│   └── video.py                    # Script related to video processing or capture
├── .gitattributes                  # Git configuration
└── .gitignore                      # Files/directories to ignore in Git
Note: The class StereoYOLOProcessor appears to be defined within detection.py based on your screenshot.

Getting Started
To get this project up and running on your local machine, follow these steps:

1. Prerequisites
Ensure you have the following software installed:

Python 3.8+

Git (for cloning the repository)

2. Clone the Repository
First, clone this GitHub repository to your local machine using the command line:

Bash

git clone https://github.com/YourUsername/STOP_SIGN_PYTHON.git
cd STOP_SIGN_PYTHON
(Note: Replace YourUsername/STOP_SIGN_PYTHON.git with the actual path to your repository. This assumes STOP_SIGN_PYTHON is the name of your repository's root directory.)

3. Create a Virtual Environment (Recommended)
It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

Bash

python -m venv .venv
4. Activate the Virtual Environment
On Windows:

Bash

.\.venv\Scripts\activate
On macOS/Linux:

Bash

source ./.venv/bin/activate
5. Install Dependencies
Once your virtual environment is active, install all the required Python packages. Given your project structure, it's likely you'll have a requirements.txt file (if not, you'll need to create one listing all libraries like opencv-python, torch, numpy, ultralytics etc.).

Assuming you have a requirements.txt file (you might need to create it based on your installed libraries):

Bash

pip install -r requirements.txt
If you don't have a requirements.txt, you might need to install them manually:

Bash

pip install opencv-python numpy ultralytics configparser # Add any other specific libraries you use
6. Download Pre-trained Models and Calibration Files
YOLO Model Weights: Your project structure shows last.pt directly in the root, indicating it's your YOLO model. If this file is too large for GitHub, you'd typically provide a download link in a release or a separate location. Ensure this last.pt file is present in the STOP_SIGN_PYTHON/ root directory.

Camera Calibration File: Ensure your camera_calibration.yml file, containing the intrinsic and extrinsic parameters of your stereo camera setup, is present in a location accessible by your code (e.g., in the same directory as config.txt or a dedicated models/ folder if you create one). If you don't have one, you'll need to perform a camera calibration first (see "Calibration" section below). Update config.txt accordingly if the path differs.

7. Configuration
Review and potentially adjust the config.txt file located in the STOP_SIGN_PYTHON/ directory. This file contains key parameters such as:

model_path: Path to your YOLO model weights (e.g., last.pt).

confidence_threshold: The minimum confidence score for a detected object to be considered valid.

calibration_file: Path to your camera calibration file (e.g., camera_calibration.yml).

Other parameters related to stereo matching (e.g., num_disparities, block_size).

8. Run the Project
Once all prerequisites are met, dependencies are installed, and models are in place, you can run the main script, which appears to be detection.py.

To process a sample stereo video (if video.py or detection.py handles this):

Bash

python detection.py # Or video.py, depending on which script initializes the process
To process specific image files or live camera, you might need to modify detection.py or video.py or use command-line arguments if implemented (e.g., --left_image, --right_image, --live_feed as suggested in previous examples, if your detection.py supports them).

The script will display the detected stop signs, their bounding boxes, and the calculated distance. Output images and disparity maps may be saved to the data/output/ directory (if you create and configure such a directory and output logic in your code).

Calibration
Accurate camera calibration is crucial for precise distance estimation in stereo vision. You will need to calibrate your specific stereo camera setup to generate the camera_calibration.yml file. This project may include a separate calibration script (e.g., in utils.py or a dedicated calibration/ folder) or recommend using standard OpenCV calibration tools.

(You might want to add a brief explanation or a link to a calibration script/tutorial here.)

Contributing
We welcome contributions! If you'd like to improve this project, please follow these steps:

Fork the repository.

Create a new branch for your feature or bug fix (git checkout -b feature/your-feature-name).

Make your changes and commit them (git commit -m 'Add new feature').

Push to your branch (git push origin feature/your-feature-name).

Create a new Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.







