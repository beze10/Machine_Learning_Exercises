# Machine Learning Exercises

## Exercise 3: Lasso Regression

This exercise uses the Boston Housing dataset to explore how Lasso changes feature weights as alpha increases. The notebook compares five alpha values, reports prediction errors, and plots how many features remain in use.

### Files

- [Exercise 3 notebook](Exercise_3_Lasso_Regression.ipynb) — code, results, and conclusions.
- [BostonHousing.csv](BostonHousing.csv) — the dataset used by the notebook.
- [requirements.txt](requirements.txt) — Python packages needed to run it.

### Running the notebook

Open this folder in any IDE ( cursor, vscode whatever ) or Jupyter, select a Python environment with the required packages, and run the notebook from top to bottom. Keep the CSV beside the notebook; it loads directly from this folder.

To install the packages in your active environment:

```sh
python -m pip install -r requirements.txt
```



### Results

At alpha 0.01 and 0.1, all eight features remain active. At alpha 1, Lasso drops NOX, DIS, and AGE. At alpha 10 and 100, all feature weights become zero, so every prediction equals the training target mean.

Alpha 0.01 gives the lowest test error among the values tested on this split. Increasing alpha reduces the number of features, but it also worsens prediction performance in this experiment.