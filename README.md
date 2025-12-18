# Image Convolution & OCR Research

This Repository is to research the fundamentals of image convolution and preprocessing for Optical Character Recognition (OCR). The goal is to replicate the initial layers of a Convolutional Neural Network (CNN) in an optimized way, specifically focusing on techniques like grayscale conversion, sharpening, and edge detection to isolate text from images for machine readability.

**Update (Dec 2025):** The system has been upgraded from a simple MLP to a **Custom CNN (Convolutional Neural Network)** built with PyTorch to improve accuracy and robustness against noise and rotation.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This requires PyTorch. If you don't have a CUDA GPU, it will run on CPU.*

## Usage

### 1. Training the Model
Before running OCR, we train the CNN (using synthetic data)
```bash
python train_cnn.py
```
Creates `ocr_cnn.pth` (the model weights) and save training plots to `plots/`.

### 2. Run Processing (Inference)
To process an image, run the script with the image filename:
```bash
python main.py test_images/complex.jpg
```
The script:
1.  Preprocess the image (Grayscale, Threshold).
2.  Segment lines and characters.
3.  Use the trained CNN to predict characters.
4.  Save the text to `output.md`.

## Methodology

*   **Model:** A 3-layer Convolutional Neural Network (CNN) with Batch Normalization and Dropout.
*   **Data:** Synthetic data generated on-the-fly using `Pillow` and system fonts.
*   **Augmentation:** Training data includes random rotations, scaling, Gaussian blur, and noise to simulate real-world document artifacts.
*   **Preprocessing:** Input images are thresholded and inverted (White text on Black background) before segmentation.