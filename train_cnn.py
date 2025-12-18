from ocr_model_cnn import OCRModelCNN
import glob
import os

def get_system_fonts():
    # Windows font path
    font_dir = r"C:\Windows\Fonts"
    all_fonts = glob.glob(os.path.join(font_dir, "*.ttf"))
    
    # Filter out symbol fonts
    valid_fonts = []
    for f in all_fonts:
        name = os.path.basename(f).lower()
        if "webdings" in name or "wingding" in name or "symbol" in name or "holo" in name:
            continue
        valid_fonts.append(f)
        
    return valid_fonts

def main():
    print("Initializing ResNet OCR Training (V2 - High Accuracy)...")
    model = OCRModelCNN()
    
    fonts = get_system_fonts()
    print(f"Found {len(fonts)} system fonts.")
    
    # Priority: Serif fonts (Times, Georgia, Palatino, etc.) + Standard Sans
    priority_keywords = ["times", "georgia", "cour", "bookman", "palatino", "garamond", "century", "serif", "arial", "calibri"]
    
    selected_fonts = []
    priority_fonts = []
    
    for f in fonts:
        name = os.path.basename(f).lower()
        if any(k in name for k in priority_keywords):
            priority_fonts.append(f)
            
    selected_fonts.extend(priority_fonts)
            
    # Fill rest with random others up to 150
    for f in fonts:
        if f not in selected_fonts and len(selected_fonts) < 150:
            selected_fonts.append(f)
            
    print(f"Selected {len(selected_fonts)} fonts for training (Priority: {len(priority_fonts)}).")
    
    # Train
    
    model.train(font_paths=selected_fonts, epochs=25, batch_size=128)
    print("Training Finished.")

if __name__ == "__main__":
    main()