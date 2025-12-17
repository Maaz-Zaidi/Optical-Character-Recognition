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

    def invert(self):
        new_pixels = [self.max_val - p for p in self.pixels]
        return Image(self.width, self.height, new_pixels, self.max_val)

    def crop(self, x, y, w, h):
        new_pixels = []
        for j in range(h):
            row_start = (y + j) * self.width + x
            new_pixels.extend(self.pixels[row_start : row_start + w])
        return Image(w, h, new_pixels, self.max_val)

    def resize(self, new_w, new_h):
        # Nearest neighbor resizing for simplicity and speed on binary images
        x_scale = self.width / new_w
        y_scale = self.height / new_h
        new_pixels = []
        for y in range(new_h):
            for x in range(new_w):
                src_x = int(x * x_scale)
                src_y = int(y * y_scale)
                new_pixels.append(self.get_pixel(src_x, src_y))
        return Image(new_w, new_h, new_pixels, self.max_val)

    def resize_contain(self, target_w, target_h, bg_color=0, padding=0):
        # 1. Calculate available space
        avail_w = max(1, target_w - 2 * padding)
        avail_h = max(1, target_h - 2 * padding)

        # 2. Calculate scale to fit while maintaining aspect ratio
        scale = min(avail_w / self.width, avail_h / self.height)
        new_w = int(self.width * scale)
        new_h = int(self.height * scale)
        
        # Avoid 0 dimensions
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # 3. Resize source
        resized_img = self.resize(new_w, new_h)
        
        # 4. Create canvas
        total_pixels = target_w * target_h
        canvas_pixels = [bg_color] * total_pixels
        
        # 5. Paste centered
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        
        for y in range(new_h):
            for x in range(new_w):
                val = resized_img.get_pixel(x, y)
                target_idx = (start_y + y) * target_w + (start_x + x)
                
                # Boundary check just in case
                if 0 <= target_idx < len(canvas_pixels):
                    canvas_pixels[target_idx] = val
                    
        return Image(target_w, target_h, canvas_pixels, self.max_val)

    def segment_lines(self, threshold_density=0.01):
        # Horizontal projection to find lines
        lines = []
        in_line = False
        start_y = 0
        
        # Calculate row densities
        row_sums = []
        for y in range(self.height):
            row_sum = sum(1 for x in range(self.width) if self.get_pixel(x, y) > 0)
            row_sums.append(row_sum)

        for y, val in enumerate(row_sums):
            if val > (self.width * threshold_density) and not in_line:
                in_line = True
                start_y = y
            elif val <= (self.width * threshold_density) and in_line:
                in_line = False
                if y - start_y > 5: # Minimum line height
                     lines.append(self.crop(0, start_y, self.width, y - start_y))
        
        # Capture last line if image ends
        if in_line:
             lines.append(self.crop(0, start_y, self.width, self.height - start_y))
             
        return lines

    def segment_chars(self, threshold_density=0.01):
        # Vertical projection to find chars in a line
        chars = []
        in_char = False
        start_x = 0
        
        col_sums = []
        for x in range(self.width):
            col_sum = sum(1 for y in range(self.height) if self.get_pixel(x, y) > 0)
            col_sums.append(col_sum)
            
        for x, val in enumerate(col_sums):
            if val > (self.height * threshold_density) and not in_char:
                in_char = True
                start_x = x
            elif val <= (self.height * threshold_density) and in_char:
                in_char = False
                if x - start_x > 2:
                    chars.append(self.crop(start_x, 0, x - start_x, self.height))

        if in_char:
            chars.append(self.crop(start_x, 0, self.width - start_x, self.height))
            
        return chars
