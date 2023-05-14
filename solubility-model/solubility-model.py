import pandas as pd

# Read the dataset
dataset = pd.read_csv("delaney_solubility_with_descriptors.csv")

# Dividing features (X) and target (Y)
X = dataset.drop(["logS"], axis=1)
Y = dataset.iloc[:, -1]


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Create and fitting the model
model = linear_model.LinearRegression()
model.fit(X,Y)

# Target Predicted by the model
Y_pred = model.predict(X)


# Retrieving and reading the performance of the model
print("Coefficients: ", model.coef_)
print("Interceptor: ", model.intercept_)
print("Mean Squared Error (MSE): %.2f"
      % mean_squared_error(Y, Y_pred) )
print("Coefficient of determination (R^2): %.2f"
      % r2_score(Y, Y_pred))


# Model Equation
print("LogS = %.2f %.2f LogP %.4f MW + %.4f RB %.2f AP" % (model.intercept_, model.coef_[0], model.coef_[1], model.coef_[2], model.coef_[3] )) 


# Data Visualization ( Experimental vs Predicted Logs for Training Data
from matplotlib import pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=Y, y=Y_pred, c="#7CAE00", alpha=0.3)

# Add trendline

z = np.polyfit(Y, Y_pred, 1)
p = np.poly1d(z)

plt.plot(Y, p(Y), "#F8866D")
plt.ylabel("Predicted LogS")
plt.xlabel("Experimental LogS")


#Saving the model as object
import pickle
pickle.dump(model, open("solubility_model.pkl", "wb"))