import sys
import os
from processor import Image
import subprocess
import tempfile

def ascii_art(img):
    chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
    w, h = img.width, img.height
    
    target_w = 200
    if w > target_w:
        scale = target_w / w
        new_w = target_w
        new_h = int(h * scale * 0.5) 
    else:
        new_w = w
        new_h = int(h * 0.5)
        
    new_h = max(1, new_h)
    
    resized = img.resize(new_w, new_h)
    
    print(f"Image Dimensions: {w}x{h}")
    print(f"ASCII Preview ({new_w}x{new_h}):")
    print("-" * new_w)
    
    for y in range(new_h):
        line = ""
        for x in range(new_w):
            val = resized.get_pixel(x, y)
            if isinstance(val, tuple):
                val = sum(val) // 3
            
            # val is 0..255. 0 is black (bg in OCR input). 255 is white (text)
            idx = int((val / 255) * (len(chars) - 1))
            line += chars[idx]
        print(line)
    print("-" * new_w)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Inspecting {path}...")
        
        # Helper to load standard formats if not ppm
        if not path.endswith('.ppm'):
             import tempfile
             fd, temp_ppm = tempfile.mkstemp(suffix=".ppm")
             os.close(fd)
             subprocess.run([sys.executable, 'convert_to_ppm.py', path, temp_ppm])
             try:
                img = Image.load_ppm(temp_ppm)
                ascii_art(img)
             finally:
                if os.path.exists(temp_ppm):
                    os.remove(temp_ppm)
        else:
            img = Image.load_ppm(path)
            ascii_art(img)
