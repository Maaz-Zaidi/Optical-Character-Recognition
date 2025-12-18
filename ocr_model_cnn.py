import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import os
import random
import glob
import joblib
import matplotlib.pyplot as plt

import torchvision.models as models

# Define the CNN Architecture (ResNet18 adapted for 1-channel 28x28)
class ResNetOCR(nn.Module):
    def __init__(self, num_classes):
        super(ResNetOCR, self).__init__()
        # Load standard ResNet18
        self.resnet = models.resnet18(weights=None)
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the maxpool layer to preserve spatial dim for small images
        self.resnet.maxpool = nn.Identity()
        
        # Modify the fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class SimpleCNN(nn.Module): # Deprecated but kept to avoid breaking if referenced elsewhere (replaced in usage)
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        pass

class CharacterDataset(Dataset):
    def __init__(self, samples_per_char=100, font_paths=None, chars=None, img_size=(28, 28), augment=True):
        self.samples_per_char = samples_per_char
        self.img_size = img_size
        self.chars = chars if chars else "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        self.augment = augment
        self.data = []
        self.labels = []
        
        self.generate_data(font_paths)

    def generate_data(self, font_paths):
        fonts = []
        if font_paths:
            for fp in font_paths:
                try:
                    # Use different sizes for variety
                    fonts.append(ImageFont.truetype(fp, 14))
                    fonts.append(ImageFont.truetype(fp, 16))
                    fonts.append(ImageFont.truetype(fp, 18))
                    fonts.append(ImageFont.truetype(fp, 20))
                    fonts.append(ImageFont.truetype(fp, 22))
                    fonts.append(ImageFont.truetype(fp, 24))
                    fonts.append(ImageFont.truetype(fp, 28))
                    fonts.append(ImageFont.truetype(fp, 32))
                except Exception as e:
                    pass
        
        if not fonts:
            try:
                fonts.append(ImageFont.truetype("arial.ttf", 20))
            except:
                fonts.append(ImageFont.load_default())

        print(f"Generating synthetic data with {len(fonts)} fonts/sizes x {self.samples_per_char} samples...")
        
        difficult_chars = ['i', 'l', '1', 'I', 'o', 'O', '0', 't', 'f', 'r', 'c', 'C', 'e', 's', 'S']
        
        for char_idx, char in enumerate(self.chars):
            # Oversample difficult characters
            current_samples = self.samples_per_char
            if char in difficult_chars:
                current_samples += 50 # Add extra samples for hard cases
                
            for _ in range(current_samples):
                font = random.choice(fonts)
                
                # 1. Create Base Image
                img = Image.new('L', (40, 40), color=0) # Larger canvas for rotation
                draw = ImageDraw.Draw(img)
                
                # Center text
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
                    w = right - left
                    h = bottom - top
                except:
                    w, h = draw.textsize(char, font=font)
                    
                x = (40 - w) / 2
                y = (40 - h) / 2
                draw.text((x, y), char, font=font, fill=255)
                
                if self.augment:
                    # Rotation
                    angle = random.uniform(-15, 15)
                    img = img.rotate(angle, resample=Image.BILINEAR)
                    
                    
                    # Gaussian Blur / Noise
                    if random.random() > 0.7:
                        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
                        
                # Crop and Resize to Target
                bbox = img.getbbox()
                if bbox:
                    img_cropped = img.crop(bbox)
                else:
                    img_cropped = img
                
                # Resize to target size, maintaining aspect ratio
                target_w, target_h = self.img_size
                img_final = ImageOps.contain(img_cropped, (target_w, target_h))
                
                # Paste onto black background
                bg = Image.new('L', self.img_size, color=0)
                pos = ((target_w - img_final.width) // 2, (target_h - img_final.height) // 2)
                bg.paste(img_final, pos)
                
                # Add Noise
                if self.augment:
                     np_img = np.array(bg)
                     # Gaussian Noise
                     if random.random() > 0.5:
                         noise = np.random.normal(0, 10, np_img.shape)
                         np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
                     
                     # Binarize 
                     thresh = random.randint(100, 150)
                     np_img = (np_img > thresh).astype(np.uint8) * 255
                     
                     bg = Image.fromarray(np_img)

                # Normalize to 0-1 tensor format
                img_tensor = torch.tensor(np.array(bg), dtype=torch.float32) / 255.0
                img_tensor = (img_tensor - 0.5) / 0.5 
                
                self.data.append(img_tensor.unsqueeze(0)) #
                self.labels.append(char_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class OCRModelCNN:
    def __init__(self, model_path="ocr_cnn.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        self.img_size = (28, 28)
        self.model = ResNetOCR(len(self.chars)).to(self.device)
        
    def train(self, font_paths=None, epochs=10, batch_size=64):
        print(f"Training on {self.device}...")
        
        # Dataset
        dataset = CharacterDataset(samples_per_char=50, font_paths=font_paths, chars=self.chars, img_size=self.img_size, augment=True)
        
        # Split
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        history = {'loss': [], 'acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            # Validation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            avg_loss = running_loss / len(train_loader)
            history['loss'].append(avg_loss)
            history['acc'].append(acc)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {acc:.2f}%")
            
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Plot training
        if not os.path.exists("plots"):
            os.makedirs("plots")
            
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Loss')
        plt.title('Training Loss')
        plt.subplot(1, 2, 2)
        plt.plot(history['acc'], label='Accuracy')
        plt.title('Validation Accuracy')
        plt.savefig("plots/cnn_training.png")
        plt.close()

    def load(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("Loaded CNN model weights.")
            return True
        return False

    def predict(self, pixel_list_flat, width=20, height=20):
        # Compatibility wrapper for old main.py
        
        if not hasattr(self, '_loaded'):
            if not self.load():
                 print("Model not trained yet!")
                 return "?"
            self._loaded = True
            
        # Reshape to image
        img_arr = np.array(pixel_list_flat, dtype=np.uint8).reshape(height, width)
        
        if img_arr.max() <= 1:
            img_arr = img_arr * 255
            
        img_pil = Image.fromarray(img_arr).convert('L')
        img_pil = ImageOps.contain(img_pil, self.img_size) # Resize to fit 28x28
        
        img_arr = np.array(img_pil)
        img_arr = (img_arr > 128).astype(np.uint8) * 255
        img_pil = Image.fromarray(img_arr)
        
        bg = Image.new('L', self.img_size, color=0)
        pos = ((self.img_size[0] - img_pil.width) // 2, (self.img_size[1] - img_pil.height) // 2)
        bg.paste(img_pil, pos)
        
        # Tensor
        img_tensor = torch.tensor(np.array(bg), dtype=torch.float32) / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            
        return self.chars[predicted.item()]
