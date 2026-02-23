import pickle
from statistics import LinearRegression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("housing_prices_SLR.csv")

X = dataset[["AREA"]]
Y = dataset["PRICE"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,random_state=42)

model = LinearRegression()

model.fit(X_train,Y_train)

with open("siva.pkl","wb") as f:
    pickle.dump(model,f)
