<h1 align="center">CNN-LSTM-Based Bilingual Receipt Information Extraction Using Template-Based Data Generation</h1>

<div align='center'>
    <a href='https://ieeexplore.ieee.org/document/10770740' target='_blank'><strong>IEEExplore Link</strong></a><sup></sup>&emsp;
    <a href='https://github.com/pupshii' target='_blank'><strong>Poonyavee Wongwisetsuk</strong></a><sup></sup>&emsp;
    <a href='https://github.com/PhoramintTaweeros' target='_blank'><strong>Phoramint Chotwarutkit</strong></a><sup></sup>&emsp;
</div>

<div align='center'>
    King Mongkut's University of Technology Thonburi
</div>

## 🧾 Introduction

This project implements a **CNN-LSTM image captioning system** designed for **bilingual (Thai–English)** receipt information extraction.  
It automatically generates structured text (captions) describing key information (e.g., store name, items, price, total) directly from receipt images.

The approach combines **Convolutional Neural Networks (CNN)** for visual feature extraction and **Long Short-Term Memory (LSTM)** networks for text sequence generation.  
A **template-based data generation** pipeline is used to create synthetic bilingual training data to overcome limited real-world datasets.

---

## 🧠 Overview

The system aims to bridge the gap between OCR-based document reading and multimodal understanding by teaching a neural model to “describe” receipts in natural language.

**Core ideas:**
- Learn to extract key entities (items, totals, store names) visually.
- Support bilingual text (Thai + English).
- Use synthetic data generated from templates to improve generalization.

---

## 📁 Repository Structure

| File | Description |
|------|--------------|
| **train2.py** | Main training and evaluation script. Builds dataset, model, optimizer, and loss; runs training loop and test function. |
| **model2.py** | Defines the deep learning architecture: `EncoderCNNResnet50`, `DecoderRNN`, and `CNNtoRNN` wrapper. |
| **dataloader2.py** | Handles data loading, image transformations, tokenization, and batching. |
| **readfile_torch.py** | Reads receipt text from template or OCR files and formats caption pairs. |
| **Final Report-2.pdf** | Research report explaining the methodology, experiments, and evaluation results. |
| *(optional)* **utils.py**, **preprocessing_torch.py** | Helper modules for checkpointing, logging, or preprocessing (may need to be added). |

---

## 🧩 Requirements

Python ≥ 3.9  
PyTorch ≥ 2.0  
TorchVision ≥ 0.15  
Transformers ≥ 4.30  
Pillow ≥ 9.0  
NumPy, Pandas, Matplotlib  

Install dependencies:
```bash
pip install torch torchvision transformers pillow numpy pandas matplotlib spacy opencv
```

## Getting Started
### 1. Clone the code
```bash
git clone https://github.com/pupshii/NLP-ReceiptExtraction
cd NLP-ReceiptExtraction
```
### 2. Prepare the environment
```bash
# create env using conda
conda create -n NLP-ReceiptExtraction python==3.9.18
conda activate NLP-ReceiptExtraction

# install dependencies with pip
pip install -r requirements.txt
```
### 3. Inference
```bash
python inference.py
```
The default file in `data_inference/inference_image` is a sample file. 
If you want to test with your own image of receipt, before running the inference, place the image to be inferred in the folder `data_inference/inference_image`.

### 4. Generate augmented receipt images
To start generating augmented images of receipts, use the following command:
```bash
python generate.py [Number of Image to Generate] [Output Folder Name]
```
For Example: The following commands initiate the process to generate 1000 images to
‘sample_data’ folder
```bash
python generate.py 1000 sample_data
```

### 5. Inference performance evaluation
To test the model for accuracy, use the following command:
```bash
python3 validate.py [Folder Name]
```
For Example: if you want to validate the model using folder, “test_image_folder”
```bash
python3 validate.py test_image_folder
```

## Acknowledgements
We would like to thank these repositories for contributing to this research:  [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp), [Levenshtein](https://github.com/rapidfuzz/Levenshtein), [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy), [EasyOCR](https://github.com/JaidedAI/EasyOCR)


