# Mint-Gender-Prediction-ML

This is a Machine Learning (ML) project aimed at classifying user gender (Male/Female) based on anonymous financial transaction data.

The project focuses on building and comparing various classic and advanced classification models using the Python data science stack.

## Project Structure

The project is comprised of two core files:

1.  **`mint_gender.ipynb`**:
    * The main Jupyter Notebook containing the entire ML pipeline: data loading, cleaning, Feature Engineering, **model training**, and performance evaluation.
    * The notebook compares performance across several algorithms, including **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**.

2.  **`utils.py`**:
    * A Python module containing essential utility functions, categorical mappings (`category_mapping`), constants (`RANDOM_STATE`), and standard model evaluation routines:
        * `eval_model`: For evaluating a model on training and validation sets.
        * `test_results`: For evaluating a model on the held-out test set.

## Technologies and Dependencies

The project requires the following standard Python libraries (as imported in `utils.py`):

### Installation

All necessary libraries can be installed using `pip`:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyter
