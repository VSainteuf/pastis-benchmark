# Welcome

Welcome to the **Mines x Invent 2024 Data Challenge**! This repository is designed to help Mines Paris students get started quickly with the challenge.

## 🏗️ Setup Instructions

### 1. Create a Virtual Environment

First, you'll need to create a Python virtual environment. Make sure you're using Python 3.12:

```bash
conda create -n env_challenge python=3.12
conda activate env_challenge
```

> **Note:** For those familiar with Poetry, you can by-pass the Conda step. We add it to simplify the install.

### 2. Install Dependencies

Next, install the required dependencies for the project:

```bash
pip install poetry
poetry install --with dev
```

For more information about Poetry, check out their [**documentation**](https://python-poetry.org).

## 📊 Dataset

The dataset for this challenge is quite large, but for initial experiments, we recommend starting with the mini dataset.

- **Mini Dataset**: Available on Kaggle in the [Data section](https://www.kaggle.com/competitions/data-challenge-invent-mines2024/data).
- **Full Dataset**: Coming soon on Kaggle.

## 🧪 Running the Demo

To get a quick start, check out the `demo.ipynb` notebook. It will guide you through loading and visualizing the dataset, helping you familiarize yourself with the data.

## 📚 Documentation

This static documentation website is built using MkDocs and hosted on GitHub Pages. Make sure to document your work thoroughly—your future self will thank you for presenting your project clearly! 😇