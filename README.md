# Machine Learning Exercises

This repository contains machine learning regression exercises using the Boston Housing dataset.

## Exeercise 2: Ridge Regression

This exercise explores Ridge Regression and L2 regularisation using the Boston Housing dataset. It tests alpha values of 0.01, 0.1, 1, 10, 100, compares train/test errors, and examines how the cofficients change.

## Exercise 3: Lasso Regression

This exercise uses the Boston Housing dataset to explore how Lasso changes feature weights as alpha increases. The notebook compares five alpha values, reports prediction errors, and plots how many features remain in use.

## Exercise 4: Elastic Net Regression

This exercise uses the Boston Housing dataset to explore Elastic Net, combining L1 and L2 regularisation. It tests different alpha and l1_ratio values, compares Elastic Net with Ridge and Lasso, and visualises coefficient changes.

### Files

- [Exercise 2 notebook](Exercise_2_Ridge_Regression.ipynb) — code, results, and conclusions.
- [Exercise 3 notebook](Exercise_3_Lasso_Regression.ipynb) — code, results, and conclusions.
- [Exercise 4 - Elastic Net](Exercise_4_Elastic_Net.ipynb) — Elastic Net tuning, model comparison, and coefficient analysis.
- [BostonHousing.csv](BostonHousing.csv) — the dataset used by the notebook.
- [requirements.txt](requirements.txt) — Python packages needed to run it.

### Running the notebook

Open this folder in any IDE ( cursor, vscode whatever ) or Jupyter, select a Python environment with the required packages, and run the notebook from top to bottom. Keep the CSV beside the notebook; it loads directly from this folder.

To install the packages in your active environment:

```sh
python -m pip install -r requirements.txt
```

### Results

### Exeercise 2: Ridge Regression

As alpha increases, the Ridge Regression exercise applies stronger regularization and the coefficients generally shrink towards zero. The features rremain in the model beacuse L2 regularisation does not normalluy force coefficients to become exactly zero.

Higher alpha values apply stronger regularisation, which can eventually increase prediction error if the model becomes too constrained.

### Exeercise 3: Lasso Regression

At alpha 0.01 and 0.1, all eight features remain active. At alpha 1, Lasso drops NOX, DIS, and AGE. At alpha 10 and 100, all feature weights become zero, so every prediction equals the training target mean.

Alpha 0.01 gives the lowest test error among the values tested on this split. Increasing alpha reduces the number of features, but it also worsens prediction performance in this experiment.

#### Exercise 4: Elastic Net Regression

Elastic Net combines L1 and L2 regularisation. Increasing **l1_ratio** makes the model behave more like Lasso, while lower values increase the influence of L2 regularisation. The exercise compares Elastic Net against Ridge and Lasso and shows how feature coefficients change at different **l1_ratio** settings.
