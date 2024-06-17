import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

random_state = 42

iris = load_iris()

def get_incorrect(actual, predict):
    incorrect = 0
    for i in range(len(predict)):
        predict_index = np.argmax(predict[i])
        actual_index = np.argmax(actual[i])
        if predict_index != actual_index:
            incorrect += 1
    return incorrect

def transform_output(x):
    if x == 0:
        return [1.0, 0.0, 0.0]
    elif x == 1:
        return [0.0, 1.0, 0.0]
    elif x == 2:
        return [0.0, 0.0, 1.0]
    else:
        raise NotImplemented()
output = np.array(np.matrix(list(map(transform_output, iris.target.tolist()))))

scalar = StandardScaler()
input = scalar.fit_transform(iris.data)

train_input, test_input, train_output, test_output = train_test_split(input, output, test_size=0.2, random_state=random_state)

models = [
    LinearRegression(),
    Ridge(random_state=random_state),
    RidgeClassifier(random_state=random_state),
    GaussianProcessRegressor(random_state=random_state),
    MLPClassifier(max_iter=1000, random_state=random_state),
    RandomForestClassifier(random_state=random_state),
]

lowest_error = math.inf
best_model = None
for i, model in enumerate(models):
    print(model)
    model.fit(train_input, train_output)

    predict = model.predict(test_input)
    error = root_mean_squared_error(test_output, predict)
    print(f"Test Error: {error}")
    print(f"Test Incorrect: {get_incorrect(test_output, predict)}")

    predict = model.predict(input)
    error = root_mean_squared_error(output, predict)
    print(f"Total Error: {error}")
    print(f"Total Incorrect: {get_incorrect(output, predict)}")

    if error < lowest_error:
        lowest_error = error
        best_model = i

    print()

print(f"The best model is {models[best_model]}")
