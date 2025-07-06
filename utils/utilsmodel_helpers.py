# utils/model_helpers.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def word_for_id(integer, tokenizer):
    """
    Map an integer to a word using the tokenizer.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def id_for_word(word, tokenizer):
    """
    Map a word to its integer ID using the tokenizer.
    """
    return tokenizer.word_index.get(word)

def generate_caption(model, photo_features, tokenizer, max_length):
    """
    Generate a caption for a given photo using the trained model.
    """
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [id_for_word(word, tokenizer) for word in in_text.split() if id_for_word(word, tokenizer) is not None]
        sequence = pad_sequences([sequence], maxlen=max_length)[0] # Pad sequence for the model input

        yhat = model.predict([photo_features, np.array([sequence])], verbose=0)
        yhat = np.argmax(yhat) # Get the word with highest probability
        word = word_for_id(yhat, tokenizer)

        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.replace('startseq ', '').replace(' endseq', '')
    return final_caption

def decode_sequence(sequence, tokenizer):
    """
    Decodes an integer sequence back into a human-readable sentence.
    """
    words = []
    for integer in sequence:
        word = word_for_id(integer, tokenizer)
        if word is not None:
            words.append(word)
        else:
            words.append("<unknown>") # Handle unknown tokens
    return ' '.join(words)