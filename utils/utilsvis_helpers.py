# utils/vis_helpers.py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_image_and_caption(image_path, caption):
    """
    Displays an image with its generated caption.
    """
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')
    plt.show()

def plot_training_history(history):
    """
    Plots the training and validation loss/accuracy from a Keras history object.
    """
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy (if applicable, e.g., if you have accuracy metrics)
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()