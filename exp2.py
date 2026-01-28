 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
 
# Load dataset 
df = pd.read_csv("weight-height.csv") 
 
# Features and target ++
X = df[["Height"]] 
Y = df["Weight"]  
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
 
# Load dataset 
df = pd.read_csv("weight-height.csv") 
 
# Features and target ++
X = df[["Height"]] 
Y = df["Weight"] 
 
# Train-test split 
X_train, X_test, Y_train, Y_test = train_test_split( 
X, Y, test_size=0.2, random_state=42 
) 
 
# Create and train model 
model = LinearRegression() 
model.fit(X_train, Y_train) 
 
# Prediction 
Y_pred = model.predict(X_test) 
 
 
  
# Mean squared Error calculation 
mse = mean_squared_error(Y_test, Y_pred) 
print(f"Mean Squared Error: {mse}") 
 
# Plot 
plt.scatter(X_train, Y_train, color="blue", label="Training Data") 
plt.scatter(X_test, Y_test, color="green", label="Testing Data") 
plt.plot(X_test, Y_pred, color="red", label="Regression Line") 
 
plt.xlabel("Height") 
plt.ylabel("Weight") 
plt.title("Height vs Weight Regression") 
plt.legend() 
plt.show()
 
# Train-test split 
X_train, X_test, Y_train, Y_test = train_test_split( 
X, Y, test_size=0.2, random_state=42 
) 
 
# Create and train model 
model = LinearRegression() 
model.fit(X_train, Y_train) 
 
# Prediction 
Y_pred = model.predict(X_test) 
 
 
  
# Mean squared Error calculation 
mse = mean_squared_error(Y_test, Y_pred) 
print(f"Mean Squared Error: {mse}") 
 
# Plot 
plt.scatter(X_train, Y_train, color="blue", label="Training Data") 
plt.scatter(X_test, Y_test, color="green", label="Testing Data") 
plt.plot(X_test, Y_pred, color="red", label="Regression Line") 
 
plt.xlabel("Height") 
plt.ylabel("Weight") 
plt.title("Height vs Weight Regression") 
plt.legend() 
plt.show()