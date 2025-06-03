# EEG Seizure Classifier

This repository offers a comprehensive pipeline for detecting epileptic seizures from EEG (Electroencephalogram) data using deep learning techniques.
## 📁 Project Structure

* `configs/` – Configuration files for model parameters and training settings.
* `data/` – Directory for storing raw EEG datasets, also contain data class.
* `models/` – Model definition.
* `notebooks/` – Jupyter notebooks for exploratory data analysis and experiments.
* `scripts/` – Python scripts for hyperparameter tuning and testing.
* `training/` – Training related utilities.
* `utils/` – Helper functions.
* `main.py` – Entry point for training and testing.
* `requirements.txt` – List of required Python packages.

## 🧠 Dataset

The project utilizes EEG datasets for seizure detection. Ensure that the dataset is placed in the `data/` directory.
## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cboissier77/eeg_seizure_classifier.git
   cd eeg_seizure_classifier
   ```

2. **Create a virtual environment (optional but recommended):**

   [EPFL Cluster Documentation](https://scitas-doc.epfl.ch/user-guide/software/python/python-venv/)

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Training the Model

To train the model with default settings:

```bash
python main.py --mode hyperparameter_tuning --config configs/hyper_gat.yaml
```

### Testing the Model

To test a trained model:

```bash
python main.py --mode testing --config configs/hyper_gat.yaml
```
Note: make sure that the config contain the right path to the .pth of the model you want to test!


## 🧪 Evaluation Metrics

The model's performance is assessed using the Macro F1 Score metric

