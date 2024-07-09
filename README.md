<h1 align="center">Data Augmentation For Receipt Extraction using Natural Language Processing</h1>

<div align='center'>
    <a href='https://github.com/pupshii' target='_blank'><strong>Poonyavee Wongwisetsuk</strong></a><sup></sup>&emsp;
    <a href='https://github.com/PhoramintTaweeros' target='_blank'><strong>Phoramint Chotwarutkit</strong></a><sup></sup>&emsp;
</div>

<div align='center'>
    King Mongkut's University of Technology Thonburi
</div>

## Introduction
This is a code repository containing Pytorch implementation of the paper Data Augmentation For Receipt Extraction using Natural Language Processing. 

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


