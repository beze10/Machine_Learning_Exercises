# Machine Learning Exercises

This repository contains machine learning regression exercises using the Boston Housing dataset.

## Exercise 1: Overfitting in Action

This exercise builds a baseline Linear Regression model to predict `medv` using the other columns in the Boston Housing dataset. It uses an 80/20 training/test split with `random_state=42`, compares actual and predicted values, and reports test MAE, RMSE, and R². The current script evaluates test performance; it does not yet compare training and test errors to demonstrate overfitting.

## Exeercise 2: Ridge Regression

This exercise explores Ridge Regression and L2 regularisation using the bOSTON gHousing dataset. It tests alpha values of 0.01, 0.1, 1, 10, 100, compares train/test errors, and examines how the cofficients change.

## Exercise 3: Lasso Regression

This exercise uses the Boston Housing dataset to explore how Lasso changes feature weights as alpha increases. The notebook compares five alpha values, reports prediction errors, and plots how many features remain in use.

### Files

- [Exercise 1 script](Exercise_1_Overfitting_in_action.py) — Linear Regression training, predictions, and evaluation.
- [Exercise 2 notebook](Exercise_2_Ridge_Regression.ipynb) — code, results, and conclusions.
- [Exercise 3 notebook](Exercise_3_Lasso_Regression.ipynb) — code, results, and conclusions.
- [BostonHousing.csv](BostonHousing.csv) — the dataset used by the exercises.
- [requirements.txt](requirements.txt) — Python packages needed to run the exercises.

### Running the exercises

Open this folder in any IDE ( cursor, vscode whatever ) or Jupyter, select a Python environment with the required packages, and run the notebook from top to bottom. Keep the CSV beside the notebook; it loads directly from this folder.

To install the packages in your active environment:

```sh
python -m pip install -r requirements.txt
```

For Exercise 1, first change the absolute CSV path in `pd.read_csv(...)` to `"BostonHousing.csv"`. Then run the script from this folder:

```sh
python Exercise_1_Overfitting_in_action.py
```

You can also run its `# %%` cells individually in an IDE that supports Python cells.

### Results

### Exercise 1: Overfitting in Action

The script prints the training/test data shapes, a sample of actual and predicted `medv` values, and test MAE, RMSE, and R². Lower MAE and RMSE indicate smaller prediction errors; R² measures performance relative to predicting the test target mean. These test metrics provide a baseline, but training metrics are also needed to assess a training/test performance gap.

### Exeercise 2: Ridge Regression

As alpha increases, the Ridge Regression exercise applies stronger regularization and the coefficients generally shrink towards zero. The features rremain in the model beacuse L2 regularisation does not normalluy force coefficients to become exactly zero.

Higher alpha values apply stronger regularisation, which can eventually increase prediction error if the model becomes too constrained.

### Exeercise 3: Lasso Regression

At alpha 0.01 and 0.1, all eight features remain active. At alpha 1, Lasso drops NOX, DIS, and AGE. At alpha 10 and 100, all feature weights become zero, so every prediction equals the training target mean.

Alpha 0.01 gives the lowest test error among the values tested on this split. Increasing alpha reduces the number of features, but it also worsens prediction performance in this experiment.
