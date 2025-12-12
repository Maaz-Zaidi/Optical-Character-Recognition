from PIL import Image as PilImage
import sys
import os

def convert_to_ppm(input_path, output_path):
    # image to .ppm using PIlImage 
    try:
        # *has to be RGB
        img = PilImage.open(input_path)
        img = img.convert("RGB") 
        
        # Save as PPM
        img.save(output_path, format='PPM')
        print(f"Converted '{input_path}' to '{output_path}'")
        return True
    except FileNotFoundError:
        print(f"Error: input file not found at '{input_path}'", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error converting image '{input_path}': {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_ppm.py <input_image_path> <output_ppm_path>", file=sys.stderr)
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_ppm_path = sys.argv[2]
    
    if convert_to_ppm(input_image_path, output_ppm_path):
        sys.exit(0)
    else:
        sys.exit(1)
