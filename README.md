# 🖼️ Image Captioning using CNN + LSTM

This repository contains an end-to-end deep learning project that generates captions for images using a **CNN encoder** (InceptionV3 or ResNet50) and an **LSTM decoder**. The model is trained on the **Flickr8k** dataset using TensorFlow/Keras in **Google Colab**.

---

## 📂 Project Structure

```bash
image-captioning-cnn-lstm/
│
├── image_captioning_cnn_lstm.ipynb      # Main notebook (Google Colab)
│
├── dataset/
│   ├── captions.txt                     # Text file containing image IDs and captions
│   └── Images/                          # Folder containing image files (not uploaded)
│
├── tokenizer.pkl                        # Saved Keras tokenizer object
├── glove.6B.200d.txt                    # GloVe embeddings file (not uploaded)
└── README.md                            # Project documentation (this file)
```

> ⚠️ Large files like `Images/` and `glove.6B.200d.txt` are not included due to size limitations. Follow the setup instructions below to download and use them.

---

## 🚀 Workflow Summary

1. Load and clean the dataset (`captions.txt`)  
2. Extract image features using a pre-trained CNN  
3. Load GloVe embeddings and prepare the embedding matrix  
4. Tokenize and encode the captions  
5. Build and train the CNN-LSTM model  
6. Generate captions using greedy/beam search

---

## 📝 Setup Instructions (Google Colab)

### ✅ Step 1: Open the Notebook
Open the `image_captioning_cnn_lstm.ipynb` notebook in [Google Colab](https://colab.research.google.com/) for execution.

### 📦 Step 2: Upload Dataset
- Download the Flickr8k Dataset from Kaggle:  
  🔗 [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

- Upload to Colab:
  - `Flicker8k_Dataset/` → `/content/dataset/Images/`
  - `Flickr8k_text/Flickr8k.token.txt` → Rename to `captions.txt` and upload to `/content/dataset/`

### 📚 Step 3: Upload GloVe Embeddings
- Download from:  
  🔗 [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

- Upload `glove.6B.200d.txt` to `/content/`

---

## 🧠 Model Overview

- **CNN Encoder**: Pre-trained InceptionV3 or ResNet50 extracts 2048-dimensional image features.  
- **LSTM Decoder**: Takes image features and word embeddings to predict the next word in the caption sequence.  
- **Word Embeddings**: GloVe 200d vectors.  
- **Loss Function**: Sparse categorical cross-entropy  
- **Optimizer**: Adam

---

## 📈 Example Output

**Input Image**  
![Sample](https://upload.wikimedia.org/wikipedia/commons/3/3d/LARGE_elevation.jpg) <!-- Replace with your own hosted image link -->

**Generated Caption**  
```
a group of people are walking on a mountain trail
```

---

## 📦 Dependencies

> Already pre-installed in Google Colab

```bash
tensorflow
keras
numpy
pandas
matplotlib
opencv-python
Pillow
```

If running locally:
```bash
pip install tensorflow keras numpy pandas matplotlib opencv-python pillow
```

---

## 🧪 Future Improvements

- Integrate attention mechanism (Bahdanau or Luong)  
- Replace LSTM with GRU or Transformers  
- Extend to Flickr30k or MS COCO dataset  
- Add beam search decoding

---

## 🙏 Acknowledgements

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)  
- [GloVe Word Embeddings](https://nlp.stanford.edu/projects/glove/)  
- [TensorFlow Documentation](https://www.tensorflow.org/)  
- [Keras Documentation](https://keras.io/)

---

## 👤 Author

**Padala Lakshmi Sai Lahari**  

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
