# TF + Spark + MLflow Image Classification

Image classification pipeline using TensorFlow, PySpark, and MLflow, following modern Machine Learning and MLOps best practices.

This project was built as a practical reference for end-to-end ML pipelines, covering data ingestion, preprocessing, training, experiment tracking, and inference using modular, production-oriented code.

---

## PROBLEM

Binary image classification (cats vs dogs) using a scalable and reproducible machine learning pipeline.

---

## WHY THIS PROJECT?

This project demonstrates real-world ML engineering practices, including:

* Distributed data ingestion with Apache Spark
* Image preprocessing and training using TensorFlow
* Experiment tracking, metrics, and artifacts with MLflow
* Reproducible experiments and modular project structure
* Separation of training and inference pipelines

The focus is not on model complexity, but on pipeline design, reliability, and observability.

---

## DATASET

Source: Publicly available internet images
Classes:

* Cat -> label 0
* Dog -> label 1

Dataset size: Small (toy dataset for pipeline demonstration)

Directory structure:

data/raw/
cat/
dog/

NOTE:
This dataset is intentionally small and simple to emphasize pipeline architecture rather than model performance.

---

## ARCHITECTURE

Spark DataFrame
-> TensorFlow Dataset
-> CNN (TensorFlow / Keras)
-> MLflow (metrics, params, model artifacts)

---

## HOW TO RUN

RECOMMENDED: Linux or WSL
Windows may cause compatibility issues with Spark and TensorFlow.

1. Create virtual environment
   python3 -m venv .venv

2. Activate environment

Windows:
.venv\Scripts\activate

Linux / WSL:
source .venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

4. Prepare dataset
   Place your images in:

* data/raw/cat/
* data/raw/dog/

5. Start MLflow UI
   mlflow ui

6. Train model
   python3 src/train.py

---

## EXPERIMENTS

All experiments are tracked using MLflow, including:

* Training parameters
* Loss and validation metrics
* Trained TensorFlow models as artifacts

---

## PREDICTION

Run inference using a trained model:

python3 src/predict.py 
--run-id <MLFLOW_RUN_ID> 
--image data/sample/cat/cat.jpg

The model artifact is automatically loaded from MLflow.

---

## RESULTS

* Model converges successfully on a small dataset
* Loss decreases consistently across epochs
* Predictions are reproducible via MLflow runs

Due to the limited dataset size, results are not intended to represent production-level accuracy.

---

## LIMITATIONS

* Small dataset size
* Binary classification only
* No data augmentation
* No hyperparameter tuning
* CPU-only training

These limitations are intentional to keep the project focused on pipeline design and experiment tracking.

---

## FUTURE IMPROVEMENTS

* Add multi-class classification
* Introduce data augmentation
* Register models using MLflow Model Registry
* Containerize pipeline with Docker