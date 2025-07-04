#Импорт нужного инструментария
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#Созданние базы данных
data = {
    "Square_meters": [35, 50, 70, 90, 120],
    "Rooms": [1, 2, 3, 3, 4],
    "Rent_price": [300, 500, 700, 850, 1100]
}
df=pd.DataFrame(data)
#Проверка корректного заполнения данных
print(df.head())
print(df.info())
print(df.describe())
#Созданние переменных для предсказания
X = df[["Square_meters", "Rooms"]]
Y = df["Rent_price"]
#Обучи модель линейной регрессии
model = LinearRegression()
model.fit(X,Y)
#предскозание цены в зависимости от новых данных
new_rent = [[60,2]]
prediction = model.predict(new_rent)
#Вывод данных
print("Предсказанная цена:", prediction[0])
#Создание графика
plt.scatter(df["Square_meters"], df["Rent_price"], color='blue', label="данные")
plt.plot(df["Square_meters"],model.predict(X),color='red')
plt.xlabel("Площадь квартиры (кв.м)")
plt.ylabel("Цена аренды (USD)")
plt.title("Зависимость цены от площади")
plt.legend()
plt.grid(True)
plt.show()