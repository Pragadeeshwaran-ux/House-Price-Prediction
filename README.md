# House Price Prediction

A simple house price prediction project using Python and scikit-learn. This repository loads training data from `data/train.csv`, performs basic preprocessing, trains a linear regression model, and reports model performance metrics.

## Project Structure

- `main.py` - Main script that loads data, cleans it, trains the model, and prints evaluation results.
- `data/train.csv` - Input dataset containing house sale information and target prices.
- `README.md` - Project documentation.
- `requirements.txt` - Python dependencies.

## Features

- Loads housing dataset from `data/train.csv`
- Drops columns with excessive missing values
- Fills missing numeric values with medians
- Fills missing categorical values with mode values
- Uses selected numerical features for model training
- Trains a `LinearRegression` model from scikit-learn
- Computes MAE, MSE, RMSE, and R² score

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is empty, install the needed packages directly:

```bash
pip install pandas scikit-learn
```

## Usage

Run the project with:

```bash
python main.py
```

The script prints dataset information, preprocessing steps, training/test shapes, and evaluation metrics.

## Notes

- Ensure `data/train.csv` exists before running the script.
- The model uses a small subset of features and serves as a starting point for further experimentation.
