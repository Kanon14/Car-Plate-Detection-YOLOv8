# Car-Plate-Detection-YOLOv8

## Overview
This project focuses on detecting vehicles and recognizing license plates from video streams using deep learning models. It utilizes the YOLO (You Only Look Once) model for object detection and applies OCR (Optical Character Recognition) to read text from detected license plates.

## Project Setup
### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- Compatible cuda toolkit and cudnn installed on your machine.
- Anaconda or Miniconda installed on your machine.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Kanon14/Car-Plate-Detection-YOLOv8.git
cd Car-Plate-Detection-YOLOv8
```

2. Create and activate a Conda environment:
```bash
conda create -n carplate python=3.8 -y
conda activate carplate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Data Source
The training data is sourced from the [License Plate Recognition Computer Vision Project](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) provided by the Roboflow.


## Workflow
The project workflow is designed to facilitate a seamless transition from development to deployment:
1. `constants`: Manage all fixed variables and paths used across the project.
2. `entity`: Define the data structures for handling inputs and outputs within the system.
3. `components`: Include all modular parts of the project such as data preprocessing, model training, and inference modules.
4. `pipelines`: Organize the sequence of operations from data ingestion to the final predictions.
5. `app.py`: This is the main executable script that ties all other components together and runs the whole pipeline.
6. `main.py`: Python script for the carplate detection and recognition, processed MP4 and detection result in CSV saved.
7. `visualize.py`: A script designed to overlay detection results onto the video, demonstrating vehicle and license plate recognition visually in the processed output.

## How to Run
### Training and image detection:
```bash
python app.py
```
### For generate result for carplate detection and recognition:
```bash
python main.py
```
### To visualize the results:
```bash
python visualize.py
```