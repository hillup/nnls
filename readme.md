## NNLS——非负最小二乘法
algorithm introduction can be seen on  https://en.wikipedia.org/wiki/Non-negative_least_squares
### test nnls.py
```
np.random.seed(42)

n_samples, n_features = 200, 50

X = np.random.randn(n_samples, n_features) 

true_coef = 3 * np.random.randn(n_features)

true_coef[true_coef < 0] = 0

y = np.dot(X, true_coef)

y += 5 * np.random.normal(size=(n_samples,))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

coef0 = nnls(X_train, y_train)

from sklearn.linear_model import LinearRegression 

reg_nnls = LinearRegression(positive=True)

reg_nnls.fit(X_train, y_train)

y_pred_nnls = reg_nnls.predict(X_test)

coef1 = reg_nnls.coef_

fig, ax = plt.subplots()

ax.plot(coef0, coef1, linewidth=0, marker=".")

low_x, high_x = ax.get_xlim()

low_y, high_y = ax.get_ylim()

low = max(low_x, low_y)

high = min(high_x,high_y)

ax.plot([low, high], [low, high], ls="--", c=".3", alpha=.5)

ax.set_xlabel("self nnls regression coefficients", fontweight="bold")

ax.set_ylabel("NNLS regression coefficients", fontweight="bold")

plt.show()
```




