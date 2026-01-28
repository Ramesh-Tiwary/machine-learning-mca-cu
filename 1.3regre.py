import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load dataset

df = pd.read_csv("insurence.csv")
print(df.head(100))
#features and target    
x = df[["age"]]
y = df["smoker"]
#train-test split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=40)

#create logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_test)

#accuracy calculation
accuracy = accuracy_score(y_test, y_pred)
print("model accuracy:", accuracy)

age_range = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
probabilities = model.predict_proba(age_range)[:, 1]
plt.figure()
plt.scatter(x, y)
plt.plot(age_range, probabilities)
plt.xlabel("Age")
plt.ylabel("Bought Insurance Probability")
plt.title("Age vs Bought Insurance Probability")
plt.show()



