import numpy as np

x_data = np.array([2, 2.5, 2.5, 3, 3.2, 3.5, 4, 4, 4, 4.5, 4.7, 4.8, 5, 5, 5.5])
y_data = np.array([5, 5, 6.5, 6, 7, 7.5, 7.5, 7, 8.5, 8, 8, 8.5, 10, 10.5, 11])

class LinearRegression:
    weights = []
    intercept = 0

    def __init__(self):
        self.weight = 0
        self.intercept = 0

    def fit(self, x, y):
        x_bar = np.float64(format(np.mean(x), ".2f"))
        y_bar = np.float64(format(np.mean(y), ".2f"))
        a = np.float64(format(sum(((x - x_bar) * (y - y_bar))) / sum(((x - x_bar) ** 2)), ".2f"))
        b = np.float64(format(y_bar - (a * x_bar), ".2f"))
        self.weight = a
        self.intercept = b
        return

    def predict(self, x):
        predicts = (self.weight * x + self.intercept)
        predicts = np.array(list(map(lambda s: np.float64(format(s, ".2f")), predicts)))
        return predicts
    
l1 = LinearRegression()
l1.fit(x_data, y_data)
y_hats = l1.predict(x_data)

mae = np.float64(format(sum(abs(y_data - y_hats)) /len(y_data), ".2f"))
mse = np.float64(format(sum(((y_data - y_hats) ** 2)) / len(y_data), ".2f"))
y_bar = np.float64(format(np.mean(y_data), ".2f"))
r2_score = np.float64(format(1 - (sum(((y_data - y_hats) ** 2))) / sum(((y_data - y_bar) ** 2)), ".2f"))

print(mae)
print(mse)
print(r2_score)
