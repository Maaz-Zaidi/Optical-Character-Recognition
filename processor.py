import math

class Image:
    def __init__(self, width, height, pixels, max_val=255):
        self.width = width
        self.height = height
        self.pixels = pixels 
        self.max_val = max_val

    @classmethod
    def load_ppm(cls, filepath):
        # Reads a P3 (text) or P6 (binary) PPM file.
        with open(filepath, 'rb') as f:
            magic_number = f.readline().strip()
            
            # Read dimensions
            while True:
                line = f.readline().strip()
                if line and not line.startswith(b'#'):
                    width, height = map(int, line.split())
                    break
            
            # Read max_val
            while True:
                line = f.readline().strip()
                if line and not line.startswith(b'#'):
                    max_val = int(line)
                    break
            
            pixel_data = []
            if magic_number == b'P3':
                # text format (P3)
                content = f.read().decode('ascii').split()
                data_values = [int(x) for x in content]
                for i in range(0, len(data_values), 3):
                    pixel_data.append((data_values[i], data_values[i+1], data_values[i+2]))
            elif magic_number == b'P6':
                
                # read all remaining bytes.
                raw_bytes = f.read()
                
                # if we expect RGB, then it's 3 bytes per pixel
                for i in range(0, len(raw_bytes), 3):
                    r = raw_bytes[i]
                    g = raw_bytes[i+1]
                    b = raw_bytes[i+2]
                    pixel_data.append((r, g, b))
            else:
                raise ValueError("Unsupported PPM format: Only P3 and P6 are supported.")
        
        return cls(width, height, pixel_data, max_val)

    @staticmethod
    def generate_test_image(width=100, height=100):
        # white background
        pixels = [(255, 255, 255)] * (width * height)
        
        # draw a black square in the middle
        for y in range(30, 70):
            for x in range(30, 70):
                pixels[y * width + x] = (0, 0, 0)
                
        # draw a line (simulating an edge)
        for y in range(10, 90):
            pixels[y * width + 80] = (0, 0, 0)
            
        return Image(width, height, pixels, 255)

    def save_ppm(self, filepath):
        #Saves as P3 (Text PPM) for simplicity.
        with open(filepath, 'w') as f:
            f.write(f"P3\n{self.width} {self.height}\n{self.max_val}\n")
            line_len = 0
            for i, p in enumerate(self.pixels):
                if isinstance(p, int) or isinstance(p, float):
                    s = f"{int(p)} {int(p)} {int(p)} "
                else:
                    s = f"{int(p[0])} {int(p[1])} {int(p[2])} "
                
                f.write(s)
                line_len += len(s)
                if line_len > 70: 
                    f.write("\n")
                    line_len = 0

    def to_grayscale(self):
        new_pixels = []
        for p in self.pixels:
            # r, g, b
            gray = int(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
            new_pixels.append(gray)
        return Image(self.width, self.height, new_pixels, self.max_val)
    
    def get_pixel(self, x, y, border_mode='extend'):
        # safe pixel retrieval for convolution
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixels[y * self.width + x]
        
        # Border handling
        if border_mode == 'extend':
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            return self.pixels[y * self.width + x]
        return 0

    def convolve(self, kernel, kernel_w, kernel_h):
        new_pixels = []
        
        pad_w = kernel_w // 2
        pad_h = kernel_h // 2
        
        for y in range(self.height):
            for x in range(self.width):
                acc = 0
                for ky in range(kernel_h):
                    for kx in range(kernel_w):
                        # Sample image pixel around (x, y)
                        ix = x + kx - pad_w
                        iy = y + ky - pad_h
                        
                        val = self.get_pixel(ix, iy)
                        weight = kernel[ky * kernel_w + kx]
                        
                        acc += val * weight
                
                # Clamp result
                val = max(0, min(self.max_val, int(acc)))
                new_pixels.append(val)
                
        return Image(self.width, self.height, new_pixels, self.max_val)

    def threshold(self, cutoff=128):
        new_pixels = [0 if p < cutoff else self.max_val for p in self.pixels]
        return Image(self.width, self.height, new_pixels, self.max_val)
