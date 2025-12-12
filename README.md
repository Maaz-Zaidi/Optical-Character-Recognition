# Image Convolution & OCR Research

This Repository is to research the fundamentals of image convolution and preprocessing for Optical Character Recognition (OCR). The goal is to replicate the initial layers of a Convolutional Neural Network (CNN) in an optimized way, specifically focusing on techniques like grayscale conversion, sharpening, and edge detection to isolate text from images for machine readability.

Reference for the following is from 'CSI 4106: Intro to AI', by Marcel Turcotte ([turcotte.github.io/csi-4106](turcotte.github.io/csi-4106)). This is an extension of one of the topics, various reference codes are studied from here. 

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Place Test Images
Create a `test_images` folder and add `.png`, `.jpg`, or `.ppm` files there.

### 2. Run Processing
To process an image, run the script with the image filename:
```bash
python main.py my_image.png
```
If the image is in the `test_images` folder, just use the filename:
```bash
python main.py document.jpg
```

To run a default test pattern:
```bash
python main.py
```

The script will convert the image, apply convolution filters (Sharpen, Edge Detection), threshold it to black & white, and display the results in a window.