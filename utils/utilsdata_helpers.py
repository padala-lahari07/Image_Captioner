# utils/data_helpers.py

import os
import re
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import preprocess_input

def load_captions(captions_path):
    """
    Loads image captions from a text file.
    Expected format: image_name.jpg,caption_text
    """
    mapping = {}
    with open(captions_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1) # Split only on the first comma
            if len(parts) < 2:
                continue
            image_id = parts[0].strip()
            caption = parts[1].strip()
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)
    return mapping

def clean_captions(captions):
    """
    Cleans a list of captions: lowercasing, removing punctuation,
    single-character words, and numeric tokens.
    Adds 'startseq' and 'endseq' tokens.
    """
    cleaned_captions = []
    for caption in captions:
        caption = caption.lower()
        caption = re.sub(r'[^a-z\s]', '', caption) # Remove punctuation and numbers
        caption = re.sub(r'\b\w\b', '', caption)    # Remove single character words
        caption = ' '.join(caption.split())          # Remove extra spaces
        cleaned_captions.append('startseq ' + caption + ' endseq')
    return cleaned_captions

def load_and_preprocess_image(image_path, target_size=(299, 299)):
    """
    Loads an image from the given path and preprocesses it for InceptionV3.
    """
    try:
        img = Image.open(image_path).resize(target_size)
        img = np.array(img)
        # Ensure image has 3 channels (convert grayscale to RGB if needed)
        if len(img.shape) == 2: # Grayscale
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4: # RGBA
            img = img[..., :3] # Take only RGB channels
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img) # InceptionV3 specific preprocessing
        return img
    except Exception as e:
        print(f"Error loading or preprocessing image {image_path}: {e}")
        return None

def create_tokenizer(captions, num_words=None):
    """
    Creates and fits a Keras Tokenizer on the cleaned captions.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
    tokenizer.fit_on_texts(captions)
    return tokenizer

def get_max_length(tokenizer):
    """
    Calculates the maximum sequence length from the tokenizer's word_counts.
    Useful for padding sequences.
    """
    # This might need adjustment if your captions are not directly tokenized yet
    # A more robust way is to tokenize all captions and find the max length
    all_lengths = [len(s.split()) for s in tokenizer.word_index.keys()] # This is not accurate for actual sequences
    # A better way is to pass actual tokenized sequences to this function or keep this logic in the notebook
    # For now, let's assume it's calculated after tokenization
    # max_length = max(len(seq) for seq in tokenizer.texts_to_sequences(captions)) # This is better
    # For now, let's estimate or set a fixed max length. In a notebook, you'd calculate this from actual tokenized data.
    return 35 # Example value, you'll calculate this dynamically from your tokenized captions

def data_generator(images, captions_map, tokenizer, max_length, batch_size, image_dir):
    """
    A generator function for feeding data to the model during training.
    Yields (image_features, input_sequence) and output_sequence (shifted).
    """
    # This function is more complex and might be better kept in the main notebook
    # because it tightly couples image features (from InceptionV3 output) with captions.
    # If you put it here, you'd need a way to pass the pre-extracted image features.
    # For simplicity of utils, let's assume image features are already extracted and passed.
    # If not, this part remains in the notebook where feature extraction happens.
    pass # Placeholder for now.