import numpy as np
import joblib
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.neural_network import MLPClassifier

class OCRModel:
    def __init__(self, model_path="ocr_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.img_size = (20, 20)
        self.chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    def generate_training_data(self, samples_per_char=50, font_paths=None):
        print("Generating synthetic training data...")
        X = []
        y = []
        
        fonts = []
        if font_paths:
            for fp in font_paths:
                try:
                    fonts.append(ImageFont.truetype(fp, 16))
                except Exception as e:
                    print(f"Warning: Could not load font {fp}: {e}")
        
        # If no fonts loaded (or none provided), fallback to default
        if not fonts:
            try:
                fonts.append(ImageFont.truetype("arial.ttf", 16))
            except:
                fonts.append(ImageFont.load_default())

        for font in fonts:
            for char in self.chars:
                for _ in range(samples_per_char):
                    # Create a blank image
                    img = Image.new('L', self.img_size, color=0)
                    draw = ImageDraw.Draw(img)
                    
                    # Randomize position slightly for robustness
                    try:
                         # Newer Pillow versions
                        w_char = draw.textlength(char, font=font)
                    except:
                        # Older Pillow versions
                        w_char = draw.textsize(char, font=font)[0]
                        
                    # Centering logic with noise
                    x_pos = (self.img_size[0] - w_char) / 2 + np.random.randint(-2, 3)
                    y_pos = (self.img_size[1] - 16) / 2 + np.random.randint(-2, 3) # approx height
                    
                    draw.text((x_pos, y_pos), char, font=font, fill=255)
                    
                    # Convert to numpy array and flatten
                    data = np.array(img).flatten()
                    # Binarize
                    data = (data > 128).astype(int)
                    
                    X.append(data)
                    y.append(char)
                
        return np.array(X), np.array(y)

    def train(self, font_paths=None):
        # if font_paths is provided, we ignore existing model and retrain.
        if os.path.exists(self.model_path) and font_paths is None:
            print(f"Loading existing model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            return

        print("Training new OCR model (MLP Classifier)...")
        X, y = self.generate_training_data(font_paths=font_paths)
        
        # MLP (150, 100)
        self.model = MLPClassifier(hidden_layer_sizes=(150, 100), max_iter=500, alpha=1e-4, solver='adam', verbose=False, random_state=1, learning_rate_init=.005) 
        
        self.model.fit(X, y)
        print(f"Training complete. Accuracy on train set: {self.model.score(X, y):.2f}")
        
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, pixel_list_flat):
        return self.predict_enhanced(pixel_list_flat)

    def predict_enhanced(self, pixel_list_flat, aspect_ratio=None):
        if not self.model:
            self.train()
            
        # Ensure numpy array
        data = np.array(pixel_list_flat).reshape(1, -1)
        pred = self.model.predict(data)[0]
        
        # cheating for 0 vs o lol
        if aspect_ratio:
            if pred in ['O', 'o'] and aspect_ratio > 1.3:
                return '0'
            
            if pred == '0' and aspect_ratio < 1.2:
                return 'O'
                
        return pred
