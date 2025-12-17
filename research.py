import glob
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from ocr_model import OCRModel

def get_system_fonts():
    fonts = glob.glob(r"C:\Windows\Fonts\*.ttf")
    return [f for f in fonts if "webdings" not in f.lower() and "wingding" not in f.lower()]

def run_research():
    print("Starting Research Benchmark... (Grab a coffee, this might take a sec)")
    
    if not os.path.exists("plots"):
        os.makedirs("plots")

    all_fonts = get_system_fonts()
    print(f"Found {len(all_fonts)} fonts. That's a decent chunk.")
    
    np.random.shuffle(all_fonts)
    
    # 70:30 split
    split_idx = int(len(all_fonts) * 0.7)
    train_fonts = all_fonts[:split_idx]
    test_fonts = all_fonts[split_idx:]
    
    print(f"Training on {len(train_fonts)} fonts, Testing on {len(test_fonts)} unseen fonts.")

    subset_sizes = [1, 5, 10, 20, 50, len(train_fonts)]
    accuracies = []
    training_times = []

    model_validator = OCRModel("temp_research_model.pkl")
    print("Generating validation set from unseen fonts...")
    X_test, y_test = model_validator.generate_training_data(samples_per_char=5, font_paths=test_fonts[:10]) 

    print("\n Optimization Curves ---")
    for size in subset_sizes:
        print(f"Training with {size} fonts...")
        current_fonts = train_fonts[:size]
        
        model = OCRModel(f"research_models/model_{size}.pkl")
        if not os.path.exists("research_models"):
            os.makedirs("research_models")
            
        start_time = time.time()
        # Train
        model.train(font_paths=current_fonts)
        duration = time.time() - start_time
        
        # Validate
        preds = model.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        accuracies.append(acc)
        training_times.append(duration)
        print(f"-> Accuracy: {acc:.2f}, Time: {duration:.2f}s")

    # Accuracy vs Font Diversity
    plt.figure(figsize=(10, 6))
    plt.plot(subset_sizes, accuracies, marker='o', linestyle='-', color='b')
    plt.title("Optimization: Accuracy vs. Number of Training Fonts")
    plt.xlabel("Number of Fonts Used")
    plt.ylabel("Accuracy on Unseen Fonts")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("plots/optimization_accuracy.png")
    plt.close()

    # Training Speed
    plt.figure(figsize=(10, 6))
    plt.plot(subset_sizes, training_times, marker='s', linestyle='--', color='r')
    plt.title("Speed: Training Time vs. Dataset Size")
    plt.xlabel("Number of Fonts Used")
    plt.ylabel("Training Time (seconds)")
    plt.grid(True)
    plt.savefig("plots/optimization_speed.png")
    plt.close()

    # Confusion Matrix
    print("\n--- Phase 2: Deep Dive Accuracy ---")
    best_model = OCRModel(f"research_models/model_{len(train_fonts)}.pkl")
    
    preds = best_model.model.predict(X_test)
    
    cm = confusion_matrix(y_test, preds, labels=list(best_model.chars))
    
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=list(best_model.chars), yticklabels=list(best_model.chars))
    plt.title("Confusion Matrix: Where does the model get confused?")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    run_research()
