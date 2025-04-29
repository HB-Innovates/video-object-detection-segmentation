# Video Object Detection & Segmentation

## Overview
This project implements real-time object detection and semantic segmentation on video streams using YOLOv5 and U-Net models. The goal is to accurately detect and segment objects in automotive video datasets.

## Key Contributions
- Implemented YOLOv5 for real-time object detection.
- Developed U-Net for semantic segmentation.
- Applied data augmentation techniques including random crops and color jitter.
- Fine-tuned models using PyTorch.
- Evaluated performance on a custom automotive video dataset, achieving over 85% mean Average Precision (mAP).

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

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd video-object-detection-segmentation
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your data by placing raw videos in the `data/raw` directory and annotations in the `data/annotations` directory.

## Usage Guidelines
- Use the Jupyter notebooks in the `notebooks` directory for data preprocessing and model training.
- The `src` directory contains the main code for detection, segmentation, and evaluation.
- Modify the `config.yaml` file to adjust model parameters and training settings as needed.

## License
This project is licensed under the MIT License. See the LICENSE file for details.