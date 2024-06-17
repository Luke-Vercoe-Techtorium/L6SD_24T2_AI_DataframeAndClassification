import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# import dataset
data = pd.read_excel("Car_Purchasing_Data.xlsx")

# print first 5 rows
# print(data.head(5))

# print last 5 rows
# print(data.tail(5))

# determine shape
# print(data.shape)

# display summary of dataset
# print(data.describe())

# check null values
# data.info()

# create input dataset
data_input = data.drop(["Customer Name", "Customer e-mail", "Country", "Car Purchase Amount"], axis=1)
data_output = data.drop(["Customer Name", "Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth", "Customer e-mail", "Country"], axis=1)

# print(data_input.head())
# print(data_output.head())

input_scaler = MinMaxScaler()
input_scaled = input_scaler.fit_transform(data_input)

# print(input_scaled[:5])

output_scaler = MinMaxScaler()
scaled_output = output_scaler.fit_transform(data_output.values.reshape(-1,1))

# print(output_scaled[:5])

train_input, test_input, train_output, test_output = train_test_split(input_scaled, scaled_output, test_size=0.2, random_state=42)

# print(input_train.shape, input_test.shape)
# print(output_train.shape, output_test.shape)

model = LinearRegression()

model.fit(train_input, train_output)
predict = model.predict(test_input)
error = root_mean_squared_error(test_output, predict)
print(error)

model.fit(input_scaled, scaled_output)
predict = model.predict(input_scaled)
error = root_mean_squared_error(scaled_output, predict)
print(error)
