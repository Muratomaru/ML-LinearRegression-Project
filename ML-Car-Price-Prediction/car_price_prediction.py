import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"C:\Users\Asus\Desktop\программирование\ML-Projects\ML-Car-Price-Prediction\car_data.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
X = df[["Mileage_km", "Age_years"]]
Y = df["Price_usd"]
model = LinearRegression()
model.fit(X,Y)
new_car = pd.DataFrame([[45000, 2]], columns=["Mileage_km", "Age_years"])
prediction = model.predict(new_car)
print("Предсказанная цена:", prediction[0])
plt.scatter(df["Mileage_km"], df["Price_usd"],color='blue', label="Данные")
plt.plot(df["Mileage_km"],model.predict(X),color='red')
plt.xlabel("Пробег машины (км)")
plt.ylabel("Цена(USD)")
plt.legend()
plt.grid(True)
plt.show()