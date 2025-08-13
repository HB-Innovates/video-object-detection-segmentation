# Video Object Detection & Segmentation

![MIT License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

## Overview
This project implements real-time object detection and semantic segmentation on video streams using YOLOv5 and U-Net models. The goal is to accurately detect and segment objects in automotive video datasets. It is designed for automotive applications, but can be adapted for other domains.

## Key Contributions
- Implemented YOLOv5 for real-time object detection.
- Developed U-Net for semantic segmentation.
- Applied data augmentation techniques including random crops and color jitter.
- Fine-tuned models using PyTorch.
- Evaluated performance on a custom automotive video dataset, achieving over 85% mean Average Precision (mAP).

## Dataset
The dataset consists of automotive video streams with annotated bounding boxes and segmentation masks. Data should be placed in `data/raw` (videos) and `data/annotations` (labels). Example annotation formats: COCO for detection, PNG masks for segmentation.

## Sample Results
Below are sample outputs from the models:

| Model   | Input Frame | Output (Detection) | Output (Segmentation) |
|---------|-------------|--------------------|-----------------------|
| YOLOv5  | ![input](docs/sample_input.jpg) | ![detection](docs/sample_detection.jpg) | - |
| U-Net   | ![input](docs/sample_input.jpg) | - | ![segmentation](docs/sample_segmentation.jpg) |

> *Add your own sample images in the `docs/` folder for better presentation.*

## Project Structure
```
video-object-detection-segmentation
├── data
│   ├── raw                # Raw video data for training and evaluation
│   ├── processed          # Processed video data ready for model training
│   └── annotations        # Annotation files for video data (bounding boxes, masks)
├── models
│   ├── yolov5             # Implementation of the YOLOv5 model
│   └── unet               # Implementation of the U-Net model
├── notebooks
│   ├── data_preprocessing.ipynb  # Jupyter notebook for data preprocessing
│   ├── training_yolov5.ipynb     # Jupyter notebook for training YOLOv5
│   └── training_unet.ipynb        # Jupyter notebook for training U-Net
├── src
│   ├── detection.py       # Functions for object detection using YOLOv5
│   ├── segmentation.py     # Functions for semantic segmentation using U-Net
│   ├── utils.py           # Utility functions for data augmentation and processing
│   └── evaluation.py       # Functions for model evaluation
├── requirements.txt        # Python dependencies for the project
├── README.md               # Documentation for the project
└── config.yaml             # Configuration settings for the project
```

## Quickstart
```bash
git clone <repository-url>
cd video-object-detection-segmentation
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

Prepare your data by placing raw videos in the `data/raw` directory and annotations in the `data/annotations` directory.

## Usage Guidelines
- Use the Jupyter notebooks in the `notebooks` directory for data preprocessing and model training.
- The `src` directory contains the main code for detection, segmentation, and evaluation.
- Modify the `config.yaml` file to adjust model parameters and training settings as needed.
- For inference, use `src/detection.py` and `src/segmentation.py` scripts. Example:
   ```bash
   python src/detection.py --input data/raw/video.mp4 --output results/detections/
   python src/segmentation.py --input data/raw/frame.jpg --output results/segmentation/
   ```

## License
This project is licensed under the MIT License. See the LICENSE file for details.