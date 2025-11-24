# TF + MLflow Image Classification  
Image classification pipeline using **TensorFlow**, **PySpark** e **MLflow**, following modern Machine Learning best practices.

This project was created to serve as a real reference for MLOps + Deep Learning, including data ingestion, preprocessing, training, experiment tracking, and modular code organization.

---

## Main technologies:
- **TensorFlow** — Model training for image classification
- **PySpark** — Scalable ingestion, reading, and preprocessing
- **MLflow** — Tracking metrics, parameters, and artifacts
- **Python**
- **Jupyter Notebook** — Local execution and experimentation

---

## How to run locally

- python -m venv .venv

Windows:
- .venv\Scripts\activate

Linux:
- source .venv/bin/activate

- pip install -r requirements.txt

- Place your images in: data/raw/. Exemple: data/raw/cats, data/raw/dogs

- mlflow ui

- python src/train.py