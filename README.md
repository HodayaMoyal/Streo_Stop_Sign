# Stereo Stop Sign Detection and Depth Estimation

## Project Overview

This project focuses on detecting stop signs in a stereo camera setup and estimating their depth (distance from the camera). It leverages computer vision techniques, including object detection (using a YOLOv8 model) and stereo disparity calculation, to provide a 3D understanding of the scene.

## Features

* **Stop Sign Detection:** Utilizes a custom-trained YOLOv8 model (`last.pt`) for accurate real-time stop sign identification.
* **Stereo Disparity Calculation:** Computes disparity maps from stereo camera images (left and right views).
* **Depth Estimation:** Derives depth information from the calculated disparity, providing crucial distance measurements.
* **Video Processing:** Capable of processing video streams (`LEFT.mp4`, `RIGHT.mp4`) to demonstrate the system's capabilities.
* **Configuration Flexibility:** Customizable parameters via `config.txt` for different stereo camera setups (e.g., baseline, focal length).

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3.x
* `pip` (Python package installer)
* Git and Git LFS (Large File Storage)

### Installation

1.  **Clone the repository (ensure Git LFS is installed first):**
    ```bash
    git clone [https://github.com/HodayaMoyal/Streo_Stop_Sign.git](https://github.com/HodayaMoyal/Streo_Stop_Sign.git)
    cd Streo_Stop_Sign
    git lfs pull
    ```
    *Note: `git lfs pull` will download the actual large files (like `last.pt` and `.mp4` videos) which are tracked by Git LFS.*

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have a `requirements.txt` yet, create one by running `pip freeze > requirements.txt` after installing all your project's dependencies.)*

## Usage

Describe how to use your project. For example:

* **To run the main detection and depth estimation script on pre-recorded videos:**
    ```bash
    python video.py --left_video video/LEFT.mp4 --right_video video/RIGHT.mp4
    ```
    *Adjust paths as necessary.*

* **To configure camera parameters:**
    Edit the `config.txt` file to set `baseline`, `focal_length`, or other relevant parameters for your stereo camera setup.

## Project Structure

A brief overview of the main files and directories:

* `video/`: Contains sample stereo video files (`LEFT.mp4`, `RIGHT.mp4`).
* `last.pt`: The trained YOLOv8 model for stop sign detection.
* `video.py`: The main script for processing stereo video, detecting stop signs, and estimating depth.
* `config.txt`: Configuration file for camera parameters.
* `.gitattributes`: Defines Git LFS tracking rules for large files like `.pt` models and `.mp4` videos.
* `extracted_tesla_frames/`, `scene0/`, `scene1/`: [Add descriptions for these folders if they contain important data or code]

## Contributing

Contributions are welcome! If you have suggestions or improvements, please:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the [Choose a License, e.g., MIT License] - see the `LICENSE.md` file for details.

## Acknowledgments

* Mention any libraries, frameworks, or resources you used (e.g., OpenCV, PyTorch, Ultralytics YOLOv8).
* Thank anyone who helped you.
