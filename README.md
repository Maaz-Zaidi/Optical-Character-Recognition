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



## OCR

*   **Line by Line Classification:** The system segments the image into lines and then characters.

*   **Optimization & Speed:** Uses an MLP Classifier (a simple Feedforward Neural Network) trained on synthetic data. This provides a balance of speed and accuracy compared to heavier CNNs.

*   **Binary Processing:** Images are thresholded and inverted (White text on Black background) to simplify feature extraction.

*   **Output:** Recognized text is stored in `output.md`.



### How it works

1.  **Preprocessing:** Image is converted to grayscale and thresholded to binary.

2.  **Segmentation:** Histogram projection profiles are used to isolate lines and individual characters.

3.  **Classification:** Each character is resized to 20x20 pixels and classified using the pre-trained MLP model.

4.  **Storage:** Results are appended to a Markdown file.
