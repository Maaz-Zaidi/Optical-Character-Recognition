import sys
import os
import subprocess
import tempfile 
from processor import Image
from ocr_model_cnn import OCRModelCNN as OCRModel
import matplotlib.pyplot as plt

def show_results(images, titles):
    # multiple image generations for current research 
    n = len(images)
    if n == 0:
        return

    plt.figure(figsize=(15, 5))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        # converted to 2D matrix
        data = []
        # either ints/tuples dependant on grayscale/color
        is_color = isinstance(img.pixels[0], tuple)
        
        for y in range(img.height):
            row = []
            for x in range(img.width):
                pixel_val = img.pixels[y * img.width + x]
                if is_color:
                    row.append(list(pixel_val)) 
                else:
                    row.append(pixel_val)
            data.append(row)
            
        # create subplot
        plt.subplot(1, n, i + 1)
        plt.title(title)
        
        # determine cmap and data type
        if is_color:
            plt.imshow(data)
        else:
            plt.imshow(data, cmap='gray', vmin=0, vmax=255)
            
        plt.axis('off')

    print("Displaying matplotlib...")
    plt.tight_layout()
    plt.show()

def main():
    print("Image Convolution Processor & OCR")
    print("---------------------------------")
    
    images_to_show = []
    titles_to_show = []
    
    original_input_file = None
    temp_ppm_path = None

    try:
        input_file_path = None
        if len(sys.argv) > 1:
            raw_input_arg = sys.argv[1]
            candidate_path = raw_input_arg

            # Check if the input argument is just a filename and not a path, and doesn't exist in current dir
            if not os.path.dirname(raw_input_arg) and not os.path.exists(raw_input_arg):
                # Assume it might be in test_images
                test_images_path = os.path.join('test_images', raw_input_arg)
                if os.path.exists(test_images_path):
                    candidate_path = test_images_path
                else:
                    print(f"Error: Could not find '{raw_input_arg}' in current directory or 'test_images' directory.", file=sys.stderr)
                    return
            
            original_input_file = candidate_path
            file_extension = os.path.splitext(original_input_file)[1].lower()

            if file_extension in ['.png', '.jpg', '.jpeg']:
                # Convert to PPM using the helper script
                print(f"Converting '{original_input_file}' to temporary ppm...")
                fd, temp_ppm_path = tempfile.mkstemp(suffix=".ppm")
                os.close(fd) # Close the file descriptor, will open again in subprocess

                result = subprocess.run(
                    [sys.executable, 'convert_to_ppm.py', original_input_file, temp_ppm_path],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"Error during conversion: {result.stderr}")
                    return
                print(result.stdout.strip())
                input_file_path = temp_ppm_path
            elif file_extension == '.ppm':
                input_file_path = original_input_file
            else:
                print(f"Unsupported input file format: {file_extension}. Only .png, .jpg, .jpeg, .ppm are supported.", file=sys.stderr)
                return

        if input_file_path:
            print(f"Loading {input_file_path}...")
            try:
                img = Image.load_ppm(input_file_path)
            except Exception as e:
                print(f"Error loading file: {e}")
                print("Ensure the file is a valid P3/P6 PPM image.")
                return
        else:
            print("No input file provided. Generating test pattern...")
            img = Image.generate_test_image(100, 100)
            img.save_ppm("original.ppm")
            print("Saved generated pattern to 'original.ppm'")

        images_to_show.append(img)
        titles_to_show.append("Original")

        # 2. Grayscale
        print("Convert to grayscale")
        gray_img = img.to_grayscale()
        images_to_show.append(gray_img)
        titles_to_show.append("Grayscale")

        # 3. Sharpen 
        # sharpen_kernel = [
        #     0, -1, 0,
        #     -1, 5, -1,
        #     0, -1, 0
        # ]
        # print("Applying Sharpening...")
        # sharpened = gray_img.convolve(sharpen_kernel, 3, 3)
        # images_to_show.append(sharpened)
        # titles_to_show.append("Sharpened")

        # Thresholding for OCR (Need High Contrast)
        print("Applying thresholding for OCR...")
        # Standardize
        binary_img = gray_img.threshold(90) 
        
        # Check if we need to invert 
        avg_val = sum(binary_img.pixels) / len(binary_img.pixels)
        if avg_val > 128:
            print("Detected light background, inverting for OCR...")
            ocr_ready_img = binary_img.invert()
        else:
            ocr_ready_img = binary_img

        # Save OCR ready image for debug
        if not os.path.exists("debug_output"):
            os.makedirs("debug_output")
        ocr_ready_img.save_ppm("debug_output/ocr_input.ppm")

        images_to_show.append(ocr_ready_img)
        titles_to_show.append("OCR Input")

        # --- OCR SECTION ---
        print("\n--- Starting OCR Process ---")
        
        # init model
        ocr_model = OCRModel()
        if not ocr_model.load():
             print("Model not found. Please run 'train_cnn.py' first.")
             return
             
        
        full_text = ""
        
        lines = ocr_ready_img.segment_lines()
        print(f"Found {len(lines)} lines of text.")

        # Create debug directory
        debug_dir = "debug_output"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        for i, line_img in enumerate(lines):
            # Save line debug
            line_img.save_ppm(os.path.join(debug_dir, f"line_{i}.ppm"))

            # OpenCV Contours for better segmentation
            try:
                chars = line_img.segment_contours(min_area=15, erode_iters=0)
            except Exception as e:
                print(f"Contour segmentation failed: {e}, falling back to projection.")
                chars = line_img.segment_chars()

            line_str = ""
            for j, char_img in enumerate(chars):
                # Save char debug
                char_img.save_ppm(os.path.join(debug_dir, f"line_{i}_char_{j}.ppm"))

                raw_pixels = char_img.pixels
                
                # Predict
                prediction = ocr_model.predict(raw_pixels, width=char_img.width, height=char_img.height)
                line_str += prediction
            
            full_text += line_str + "\n"
            print(f"Line {i+1}: {line_str}")
            
        # Save to MD
        output_md_path = "output.md"
        with open(output_md_path, "w") as f:
            f.write("# OCR Output\n\n")
            f.write("```text\n")
            f.write(full_text)
            f.write("```\n")
        print(f"\nOCR Results saved to {output_md_path}")
        print("----------------------------")

        
        print("\nProcessing Complete.")
        print("Outputs saved to disk.")
        
        print("\nLaunching visualization...")
        #show_results(images_to_show, titles_to_show)

    finally:
        # delete temp ppm
        if temp_ppm_path and os.path.exists(temp_ppm_path):
            try:
                os.remove(temp_ppm_path)
                print(f"Cleaned up temporary file: {temp_ppm_path}")
            except OSError:
                pass

if __name__ == "__main__":
    main()