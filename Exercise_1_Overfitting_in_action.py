#26134667_Brian_Ezeanya

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

boston_housing = pd.read_csv("/Users/tochi/Downloads/Wk-1-Regression-Lab Activity-20260915/BostonHousing.csv")


# print(boston_housing.head())
# print(boston_housing.info())
# print(boston_housing.isnull().sum())
# boston_housing.info()


X = boston_housing.drop(columns=["medv"])

y = boston_housing["medv"]

# %%
# Split the rows into 80% training data and 20% test data.
# Each row in X stays paired with its matching answer in y.
#
# A random seed is a starting number for the computer's random-number generator.
# Think of shuffling a deck of cards: the seed lets us repeat the same shuffle.
# random_state=42 uses 42 as that seed, so the same data in the same order
# gives the same training/test split every time we run this cell.
# This makes results easier to reproduce and compare while learning.
# 42 is an arbitrary choice; another seed, such as 7, will generally give
# a different split. It does not mean 42 rows or 42%, or improve the model.
# Without a fixed seed, rerunning the cell can give a different split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train:" , X_train.shape)
print("X_test:" , X_test.shape)
print("y_train:" , y_train.shape)
print("y_test:" , y_test.shape)

# %%
# Creates model but hasnt yet learned anything yet.
model = LinearRegression()

#Train model using the training inputs and their matching answers.
model.fit(X_train, y_train)

print("Model training complete!")

# %%

y_pred = model.predict(X_test)

# Put the actual values and predictions side by side.
comparison = pd.DataFrame({
    "Actual medv": y_test,
    "Predicted medv" : y_pred
})


print(comparison.head())

# %%

# Compare the actual test answers (y_test) with our predictions (y_pred).
# MAE is the average size of the errors, ignoring whether we predicted
# too high or too low. Lower is better; 0 means every prediction is correct.
mae = mean_absolute_error(y_test, y_pred)

# MSE averages the squared errors, giving large mistakes more weight.
mse = mean_squared_error(y_test, y_pred)

# RMSE takes the square root of MSE to return to the original target units.
# Like MAE, lower is better. Both are in thousands of dollars for medv.
rmse = mse ** 0.5

# R² compares our errors with always predicting the mean of y_test.
# 1 = perfect predictions; 0 = matches that baseline; negative = worse.
# R² is not percentage accuracy.
r2 = r2_score(y_test, y_pred)

# .2f displays two decimal places; .3f displays three decimal places.
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
