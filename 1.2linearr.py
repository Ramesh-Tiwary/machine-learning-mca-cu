# import numpy as np
# import pandas as pd 
# import matplotlib.pyplot as plt 
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics  import mean_squared_error

# x= df[["year of experience"]]
# y=df[["salary"]]

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# model=linearRegression()
# model.fit(x_train,y_train)
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)
# mse=mean_squared_error(y_test,y_pred)
# print(f"Mean squard Error: {mse}")

# plt.scatter(x_train,y_train, color="blue",label="training")

# plt.plot(x_test,y_pred,color="red")




import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example: load dataset (ensure df exists)
df = pd.read_csv("Heart_Disease_prediction.csv")
print(df.head())
print(df.columns)

df = pd.DataFrame({
    "year of experience": [4,2,3,4,5,6,7,8,9,10],
    "salary": [30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]
})


X = df[["year of experience"]]
y = df["salary"]   # keep y as 1D

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="green", label="Test Data")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# 2nd

# import numpy as np
# import pandas as pd 
# import matplotlib.pyplot as plt 
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# df = pd.DataFrame({
#     "year of experience": [1,2,3,4,5,6,7,8,9,10],
#     "salary": [30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]
# })
# df = pd.read_csv("salary.csv")
# print(df.head())
# print(df.columns)


# X = df[["year of experience"]]
# y = df["salary"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)

# plt.scatter(X_train, y_train)
# plt.plot(X_test, y_pred)
# plt.xlabel("Year of Experience")
# plt.ylabel("Salary")
# plt.title("Linear Regression")
# plt.show()

# 3rd

# import numpy as np
# import pandas as pd 
# import matplotlib.pyplot as plt 
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Load CSV
# df = pd.read_csv("salary.csv")
# print(df.head())
# print(df.columns)

# # Features and target (EXACT column names)
# X = df[["year of experience"]]
# y = df["salary"]

# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)

# # Evaluation
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# print("MSE:", mse)
# print("RMSE:", rmse)

# # Plot
# plt.scatter(X_train, y_train, label="Training Data")
# plt.scatter(X_test, y_test, label="Test Data")
# plt.plot(X_test, y_pred, label="Best Fit Line")
# plt.xlabel("Year of Experience")
# plt.ylabel("Salary")
# plt.legend()
# plt.title("Linear Regression")
# plt.show()


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example: load dataset (ensure df exists)
df = pd.read_csv("salary.csv")
print(df.head())
print(df.columns)

df = pd.DataFrame({
    "year of experience": [4,2,3,4,5,6,7,8,9,10],
    "salary": [30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]
})


X = df[["year of experience"]]
y = df["salary"]   # keep y as 1D

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="green", label="Test Data")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()