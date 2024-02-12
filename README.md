# XTC (XAI Texture Classifier)

## Overview
The Texture Classifier project is designed to analyze and evaluate the similarity between sub-images extracted from a larger image. This project utilizes a linear classifier approach to facilitate an understanding of how decisions are made, enhancing the explainability of the classification results.

## Features
- **Sub-Image Extraction**: Efficiently extracts sub-images from a larger image for detailed analysis.
- **Linear Classifier**: Employs a linear classifier to assess the similarity between sub-images, ensuring an interpretable and straightforward decision-making process.
- **Explainability**: The linear nature of the classifier aids in explainability, allowing users to understand the basis of similarity decisions.

## Installation

To install the Texture Classifier, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/yourgithubusername/texture-classifier.git
```

2. Navigate to the project directory and dowload dataset:
cd XTC

```bash
wget -O /data/skyscraper_dataset.zip https://www.kaggle.com/datasets/mexwell/skyscraper-dataset
unzip /data/skyscraper_dataset.zip -d /data
```

3. Install requirements
```bash
pip install numpy matplotlib torch
```

5. Usage
To use the Texture Classifier, execute the following command:
```bash
python train.py
```
