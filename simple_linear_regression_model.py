import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_excel("monthly_income.xlsx")
print(df)
linear_regression = linear_model.LinearRegression()
linear_regression.fit(df[["Monthly_Income"]].values, df.Monthly_Savings)
print("Model's accuracy is:", linear_regression.score(df[["Monthly_Income"]].values, df.Monthly_Savings))
print("Model's co-efficient is:", linear_regression.coef_)
print("Model's intercept is:", linear_regression.intercept_)
print("Model's predicted value based on a certain input:", linear_regression.predict([[17000]]))
predicted = (0.29662536 * 17000) + -754.5155397220515
print("Predicted value as computed by the equation y = m * x + b:", predicted)
plt.xlabel("Monthly_Income", color = 'blue')
plt.ylabel("Monthly_Savings", color = 'green')
plt.scatter(df.Monthly_Income, df.Monthly_Savings)
plt.plot(df.Monthly_Income, linear_regression.predict(df[['Monthly_Income']].values), color = 'red')
plt.show()