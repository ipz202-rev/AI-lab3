import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

polynomial = PolynomialFeatures(degree=2, include_bias=False)
X_poly = polynomial.fit_transform(X)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_poly, y)
y_predict = poly_linear_model.predict(X_poly)

print('X[0]:', X[0])
print('X_poly:', X_poly[0])
print("Polynomial regressor coefficient:", poly_linear_model.coef_)
print("Polynomial regressor intercept:", poly_linear_model.intercept_)

X_flattened = X.flatten()
y_pred_flattened = y_predict.flatten()
sorted_indices = np.argsort(X_flattened)
X_arr = X_flattened[sorted_indices]
y_pred = y_pred_flattened[sorted_indices]

plt.figure()
plt.scatter(X, y, edgecolors=(0, 0, 0))
plt.plot(X_arr, y_pred, color="red", linewidth=3)
plt.show()
