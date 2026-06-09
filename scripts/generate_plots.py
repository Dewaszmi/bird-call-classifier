import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def plot_history():
    history_file = ROOT / "checkpoints" / "history.json"
    if not history_file.exists():
        print("No history.json found.")
        return

    with open(history_file, 'r') as f:
        data = json.load(f)

    epochs = [item['epoch'] for item in data]
    train_loss = [item['train_loss'] for item in data]
    val_loss = [item['val_loss'] for item in data]
    val_acc = [item['val_accuracy'] for item in data]
    val_f1 = [item['val_macro_f1'] for item in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, train_loss, label='Train Loss', marker='o')
    ax1.plot(epochs, val_loss, label='Validation Loss', marker='o')
    ax1.set_title('Learning Curve (Loss)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Metrics plot
    ax2.plot(epochs, val_acc, label='Accuracy', marker='s', color='green')
    ax2.plot(epochs, val_f1, label='Macro F1', marker='^', color='orange')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_path = ROOT / "history_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved history plot to {output_path}")

def plot_spectrogram():
    processed_dir = ROOT / "processed_data"
    npy_files = list(processed_dir.rglob("*.npy"))
    if not npy_files:
        print("No .npy files found.")
        return

    sample_file = npy_files[0]
    spec = np.load(sample_file)

    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f'Log-Mel Spectrogram\n{sample_file.parent.name}')
    plt.ylabel('Mel Frequency Bins')
    plt.xlabel('Time Frames')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    output_path = ROOT / "spectrogram_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved spectrogram plot to {output_path}")

if __name__ == "__main__":
    plot_history()
    plot_spectrogram()
